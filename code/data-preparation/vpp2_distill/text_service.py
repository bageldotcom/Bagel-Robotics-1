"""Original UMT5 for unseen prompts, in the VPP2 environment; never loads Wan video."""
from __future__ import annotations
import argparse
import base64
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import torch
from .contracts import file_digest, native_text_context
from .prepare import require_upstream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vpp2-root", required=True)
    parser.add_argument("--wan", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8059)
    args = parser.parse_args()
    require_upstream(args.vpp2_root)
    from vpp2.models.wan21_14b.helpers.io import load_state_dict
    from vpp2.models.wan21_14b.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer
    wan = Path(args.wan)
    checkpoint = wan / "models_t5_umt5-xxl-enc-bf16.pth"
    identity = file_digest(checkpoint)
    tokenizer = HuggingfaceTokenizer(name=str(wan / "google/umt5-xxl"), seq_len=128, clean="whitespace")
    encoder = WanTextEncoder()
    encoder.load_state_dict(load_state_dict(str(checkpoint), torch_dtype=torch.bfloat16), strict=True)
    encoder = encoder.to(device=args.device, dtype=torch.bfloat16).eval().requires_grad_(False)
    cache = OrderedDict()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("Invalid request length")
                prompt = json.loads(self.rfile.read(length))["prompt"]
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("A nonempty prompt is required")
                if prompt not in cache:
                    ids, mask = tokenizer([prompt], return_mask=True, add_special_tokens=True)
                    mask = mask.to(device=args.device, dtype=torch.bool)
                    with torch.inference_mode():
                        context = encoder(ids.to(args.device), mask)[0].cpu().to(torch.bfloat16).contiguous()
                    context, output_mask = native_text_context(context, mask[0].cpu())
                    cache[prompt] = {"prompt":prompt, "encoder_sha256":identity,
                                     "context_bf16":base64.b64encode(context.view(torch.uint8).numpy().tobytes()).decode(),
                                     "mask":output_mask.tolist()}
                    while len(cache) > 64:
                        cache.popitem(last=False)
                cache.move_to_end(prompt)
                response = json.dumps(cache[prompt]).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)
            except (ValueError, KeyError) as exc:
                self.send_error(400, str(exc))

    HTTPServer((args.host,args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
