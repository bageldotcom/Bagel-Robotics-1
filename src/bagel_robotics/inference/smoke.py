"""Check real-model inference with synthetic observations, without a simulator."""
import argparse
import json
from pathlib import Path
import time


def rgb_transport(image, *, encoded):
    """Exercise decoded RGB or the supported unmarked OpenCV PNG byte boundary."""
    import numpy as np

    image = np.ascontiguousarray(image)
    if not encoded:
        return image.copy()
    import cv2

    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise ValueError("PNG observation encoding failed")
    payload = buffer.tobytes()
    # Unmarked OpenCV buffers are decoded directly, without an RGB/BGR swap.
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or not np.array_equal(decoded, image):
        raise ValueError("PNG observation decoding changed RGB pixels")
    return decoded


def synthetic_observation(*, encoded):
    import numpy as np

    row, column = np.indices((240, 320))
    image = np.stack((column % 256, row % 256, (column * 3 + row * 5) % 256), axis=-1).astype(np.uint8)
    state = np.zeros(16, dtype=np.float32)
    state[[3, 11]] = 1
    return {"env_idx": 0,
            "observation": {name: {"rgb": rgb_transport(np.roll(image, index * 17, axis=1), encoded=encoded)}
                            for index, name in enumerate(("head_camera", "left_camera", "right_camera"))},
            "joint_action": {"vector": state}}


def check_reset(policy):
    policy.reset()
    if any(getattr(policy, name) for name in policy._PER_ENV_STATE):
        raise ValueError("Episode reset left observation or action history behind")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--wan-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--instruction", default="stack the blocks")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, help="Optional JSON check result")
    args = parser.parse_args(argv)

    import numpy as np
    import torch
    from .assets import MANIFEST, load_manifest, validate_settings, verify_bundle, verify_wan
    from .backend import configure_backend
    from .policy import get_model
    from .temporal_ensemble import TemporalEnsemble

    manifest = load_manifest(args.manifest) if args.manifest else MANIFEST
    verification = {"checkpoint": verify_bundle(args.bundle, manifest=manifest),
                    "encoders": verify_wan(args.wan_root, manifest=manifest)}
    backend = configure_backend(args.seed)
    cfg = validate_settings({"seed": args.seed})
    cfg.update(checkpoint_path=str(args.bundle / "action.pt"),
               video_checkpoint_path=str(args.bundle / "video.pt"),
               dataset_stats_path=str(args.bundle / "dataset_stats.json"),
               wan_model_dir=str(args.wan_root), device=args.device)
    started = time.perf_counter()
    policy = get_model(cfg)
    parameters = sum(parameter.numel() for parameter in policy.model.video_expert.parameters())
    if parameters != manifest["video_parameters"]:
        raise ValueError("Loaded video expert does not match the configured artifact")
    transports = {}
    reference = None
    for name, encoded in (("raw_rgb", False), ("png_bytes", True)):
        check_reset(policy)
        configure_backend(args.seed)
        observation = synthetic_observation(encoded=encoded)
        policy.observe(observation)
        inputs = (policy._latest_image_by_env[0].detach().cpu().clone(),
                  policy._normalize_state(observation["joint_action"]["vector"]).detach().cpu().clone())
        raw = np.asarray(policy._infer_action_chunk(observation, args.instruction))
        if raw.shape != (cfg["action_horizon"], 16) or not np.isfinite(raw).all():
            raise ValueError(f"Invalid action predictions for {name}")
        actions = TemporalEnsemble().apply(raw)
        if actions.shape != (cfg["replan_steps"], 16) or not np.isfinite(actions).all():
            raise ValueError(f"Invalid executable actions for {name}")
        if reference is None:
            reference = (inputs, raw.copy(), actions.copy())
        else:
            if not all(torch.equal(left, right) for left, right in zip(reference[0], inputs)):
                raise ValueError("Raw and PNG observations produced different model inputs")
            if not np.array_equal(reference[1], raw) or not np.array_equal(reference[2], actions):
                raise ValueError("Raw and PNG observations produced different deterministic actions")
        transports[name] = {"passed": True, "predicted_shape": list(raw.shape),
                            "executed_shape": list(actions.shape)}
        check_reset(policy)
    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize()
    report = {"passed": True,
              "scope": "Real-model raw RGB/PNG codec-boundary inference; no simulator or live websocket",
              "verification": verification, "backend": backend, "device": args.device,
              "codec": "unmarked_opencv_png", "video_parameters": parameters,
              "transports": transports, "model_inputs_equal": True, "actions_equal": True,
              "seconds": time.perf_counter() - started}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
