"""Persistent frozen VPP2 teacher, run separately in the native VPP2 environment."""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import logging
from pathlib import Path

import torch

from .contracts import seeded_int, tensor_digest, write_json
from .online import ONLINE_FORMAT, MAX_MESSAGE_BYTES, pack, unpack, validate_request, validate_targets


class TeacherService:
    def __init__(self, teacher, data, *, seed=100008, parity_fixture=None):
        self.teacher = teacher
        # Keep teacher state local to this process; it never enters the optimizer.
        teacher.model.eval().requires_grad_(False)
        self.norms = {**teacher.identity, "weights": torch.stack([
            b.self_attn.norm_k.weight.detach().cpu() for b in teacher.model.video_expert.blocks])}
        example = data[0]
        with torch.no_grad():
            cache = teacher.prefill(example, seeded_int(seed, "online-parity-video"))
            parity = teacher.check_parity(example, cache, seeded_int(seed, "online-parity-action"))
            if parity_fixture is not None:
                path = Path(parity_fixture)
                if path.exists():
                    raise FileExistsError(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(teacher.parity_fixture(example, cache, seeded_int(seed, "online-parity-action")), path)
        del cache
        if not parity["passed"] or parity["checked_steps"] != 10 or parity["teacher_identity"] != teacher.identity:
            raise ValueError("Native Action2B parity must pass before serving online targets")
        self.metadata = dict(format=ONLINE_FORMAT, teacher_identity=teacher.identity,
                             dataset_sha256=data.manifest["sha256"], native_parity=parity,
                             key_norms_sha256=tensor_digest(self.norms["weights"]))

    @torch.no_grad()
    def targets(self, request):
        validate_request(request, self.metadata)
        result = self.teacher.online_targets(request["example"], video_seed=request["video_seed"],
                                             query_seed=request["query_seed"], query=request["query"])
        validate_targets(result)
        return result


class SerialTeacherServer(HTTPServer):
    # Requests from multiple training ranks queue; there is only ONE live native
    # video memory per worker. Add worker replicas for throughput, not threads.
    request_queue_size = 128


def make_server(service, host="127.0.0.1", port=8060):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # Avoid a log line for every microbatch.

        def respond(self, status, payload, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/metadata":
                self.respond(200, json.dumps(service.metadata).encode(), "application/json")
            elif self.path == "/key-norms":
                self.respond(200, pack(service.norms), "application/octet-stream")
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/targets":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_MESSAGE_BYTES:
                    raise ValueError("Invalid online teacher request size")
                payload = self.rfile.read(length)
                if len(payload) != length:
                    raise ValueError("Truncated online teacher request")
                request = unpack(payload)
                result = service.targets(request)
                response = dict(result, format=ONLINE_FORMAT, teacher_identity=service.metadata["teacher_identity"],
                                request_sha256=hashlib.sha256(payload).hexdigest())
                self.respond(200, pack(response), "application/octet-stream")
            except (ValueError, KeyError, TypeError) as exc:
                self.respond(400, json.dumps({"error": str(exc)}).encode(), "application/json")
            except Exception as exc:
                logging.exception("Online teacher request failed")
                self.respond(500, json.dumps({"error": str(exc)}).encode(), "application/json")

    return SerialTeacherServer((host, port), Handler)


def audit_native_contexts(teacher, data, report_path, *, required=False):
    if not report_path and not required:
        return None
    report = teacher.audit_contexts(data)
    if report_path:
        write_json(report_path, report)
    if required and (not report.get("all_bitwise_equal") or report.get("prompts", 0) < 1):
        raise RuntimeError("Prepared text contexts must exactly match native GPU prompt encoding")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vpp2-root", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--wan", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8060)
    parser.add_argument("--seed", type=int, default=100008)
    parser.add_argument("--context-audit", help="Report native-device versus prepared UMT5 numerical differences")
    parser.add_argument("--require-native-contexts", action="store_true", help="Fail before serving if any prepared prompt differs from native encoding")
    parser.add_argument("--parity-fixture", help="Optional one-example diagnostic for student-runtime parity; not training targets")
    parser.add_argument("--instance-id", help="Launcher nonce used to identify this worker on its assigned port")
    parser.add_argument("--metadata-output", help="Optional small identity/parity JSON; no training targets are saved")
    args = parser.parse_args()
    # Import only in this worker, never in the student process.
    from .dataset import Demonstrations
    from .teacher import Teacher
    data = Demonstrations(args.data, "train")
    args.stats = str(Path(args.data) / "dataset_stats.json")
    teacher = Teacher(args, data.normalizer.identity)
    context_audit = audit_native_contexts(teacher, data, args.context_audit, required=args.require_native_contexts)
    service = TeacherService(teacher, data, seed=args.seed, parity_fixture=args.parity_fixture)
    if context_audit is not None:
        service.metadata["context_audit"] = context_audit
    if torch.cuda.is_available():
        service.metadata["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    if args.instance_id:
        service.metadata["instance_id"] = args.instance_id
    server = make_server(service, args.host, args.port)
    if args.metadata_output:
        write_json(args.metadata_output, service.metadata)
    print(json.dumps({"ready": True, "host": args.host, "port": server.server_port,
                      "teacher_identity": service.metadata["teacher_identity"],
                      "native_parity": service.metadata["native_parity"], "stores_target_cache": False}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
