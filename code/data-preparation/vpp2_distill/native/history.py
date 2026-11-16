from collections import deque
from typing import Any, Dict, Optional
import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
ROBODOJO_TSHAPE_VIDEO_SIZE = (240, 416)
ROBODOJO_HORIZONTAL_VIDEO_SIZE = (240, 960)
ROBODOJO_HORIZONTAL_PER_CAMERA_SIZE = (240, 320)

def _normalized_to_uint8(image: torch.Tensor) -> torch.Tensor:
    """Keep one composed RGB frame compactly in the CPU history buffer."""
    frame = image.detach().float().cpu()
    if frame.ndim == 4:
        if frame.shape[0] != 1:
            raise ValueError(f"Expected one image batch, got {tuple(image.shape)}")
        frame = frame[0]
    if frame.ndim != 3 or frame.shape[0] != 3:
        raise ValueError(f"Expected normalized [3,H,W], got {tuple(image.shape)}")
    return frame.clamp(-1, 1).add(1.0).mul(127.5).round().to(torch.uint8).contiguous()

def _uint8_to_normalized(frames: list[torch.Tensor]) -> torch.Tensor:
    """Stack compact CPU RGB frames into normalized ``[3,T,H,W]`` form."""
    if not frames:
        raise ValueError("Cannot build a history condition from no frames.")
    stacked = torch.stack(frames, dim=1).to(torch.float32)
    return stacked.div(255.0).mul(2.0).sub(1.0)

def _as_uint8_hwc(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"RoboDojo RGB view must be HWC, got {array.shape}")
    if array.shape[0] in {1, 3} and array.shape[-1] not in {1, 3, 4}:
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] < 3:
        raise ValueError(f"RoboDojo RGB view needs three channels, got {array.shape}")
    array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(array)) <= 1.5 else 1.0
        array = np.clip(array * scale, 0, 255).astype(np.uint8)
    else:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)

def compose_training_condition(
    head: Any,
    left_wrist: Any,
    right_wrist: Any,
    *,
    layout: str = "robodojo_tshape",
    video_size: tuple[int, int] = ROBODOJO_TSHAPE_VIDEO_SIZE,
    per_camera_video_size: Optional[tuple[int, int]] = None,
) -> torch.Tensor:
    """Reproduce the checkpoint's camera composition and normalization."""
    views = []
    for image in (head, left_wrist, right_wrist):
        tensor = torch.from_numpy(np.array(_as_uint8_hwc(image), copy=True))
        views.append(tensor.permute(2, 0, 1).to(torch.float32).div_(255.0))
    layout = str(layout).strip().lower()
    target_size = tuple((int(value) for value in video_size))
    if layout == "horizontal":
        camera_size = tuple(
            (int(value) for value in per_camera_video_size or ROBODOJO_HORIZONTAL_PER_CAMERA_SIZE)
        )
        resized_views = [
            transforms_F.resize(
                view,
                size=list(camera_size),
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            for view in views
        ]
        condition = torch.cat(resized_views, dim=2)
        if tuple(condition.shape[-2:]) != target_size:
            condition = transforms_F.resize(
                condition,
                size=list(target_size),
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
    elif layout == "robodojo_tshape":
        if target_size != ROBODOJO_TSHAPE_VIDEO_SIZE:
            raise ValueError(
                f"RoboDojo T-shape deployment requires video_size={ROBODOJO_TSHAPE_VIDEO_SIZE}, got {target_size}."
            )
        head_tensor = transforms_F.resize(
            views[0],
            size=[480, 640],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
        left_tensor = transforms_F.resize(
            views[1],
            size=[240, 240],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
        right_tensor = transforms_F.resize(
            views[2],
            size=[240, 240],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
        raw_layout = torch.cat([head_tensor, torch.cat([left_tensor, right_tensor], dim=1)], dim=2)
        condition = transforms_F.resize(
            raw_layout,
            size=list(target_size),
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
    else:
        raise ValueError(f"Unsupported RoboDojo camera layout: {layout!r}.")
    return condition.mul(2.0).sub(1.0).unsqueeze(0).contiguous()

class NativeHistory:
    _PER_ENV_STATE = ("_history_by_env", "_history_anchor_by_env", "_history_step_by_env", "_latest_image_by_env", "_inference_index_by_env")

    def _compose_observation_image(self, observation: Dict[str, Any]) -> torch.Tensor:
        images = observation["observation"]
        return compose_training_condition(
            images["head_camera"]["rgb"],
            images["left_camera"]["rgb"],
            images["right_camera"]["rgb"],
            layout=self.visual_layout,
            video_size=self.video_size,
            per_camera_video_size=self.per_camera_video_size,
        ).to(device=self.model.device, dtype=self.model.torch_dtype)


    def observe(self, observation: Dict[str, Any]) -> None:
        """Consume one native simulator observation for history conditioning.

        The official RoboDojo client calls ``update_obs`` for every action
        inside a returned chunk.  Recording here, rather than only when a new
        chunk is requested, preserves the native 25-Hz history timeline.
        """
        env_idx = int(observation.get("env_idx", 0))
        image = self._compose_observation_image(observation)
        self._latest_image_by_env[env_idx] = image
        if self.history_condition_frames > 0:
            self._record_history_frame(image, env_idx)


    def _record_history_frame(self, image: torch.Tensor, env_idx: int) -> None:
        """Append one simulator observation to the per-environment history."""
        env_idx = int(env_idx)
        history = self._history_by_env.get(env_idx)
        if history is None:
            history = deque(maxlen=self.history_buffer_size)
            self._history_by_env[env_idx] = history
            step = 0
            self._history_anchor_by_env[env_idx] = _normalized_to_uint8(image)
        else:
            step = int(self._history_step_by_env[env_idx]) + 1
        frame = _normalized_to_uint8(image)
        history.append((step, frame))
        self._history_step_by_env[env_idx] = step


    def _build_history_condition(
        self, image: torch.Tensor, env_idx: int
    ) -> tuple[Optional[torch.Tensor], Optional[dict[str, Any]]]:
        """Build the exact anchor+history RGB condition expected by aligned history."""
        if self.history_condition_frames <= 0:
            return (None, None)
        env_idx = int(env_idx)
        history = self._history_by_env.get(env_idx)
        if not history:
            self._record_history_frame(image, env_idx)
            history = self._history_by_env[env_idx]
        current_step = int(self._history_step_by_env[env_idx])
        anchor = self._history_anchor_by_env.get(env_idx)
        if anchor is None:
            anchor = history[0][1]
            self._history_anchor_by_env[env_idx] = anchor
        requested_offsets = [
            -(self.history_condition_frames - 1 - index) * self.history_stride
            for index in range(self.history_condition_frames)
        ]
        step_to_frame = {int(step): frame for (step, frame) in history}
        selected_frames: list[torch.Tensor] = []
        selected_steps: list[int] = []
        for offset in requested_offsets:
            requested_step = current_step + int(offset)
            if requested_step <= 0:
                selected_step = 0
                selected_frame = anchor
            elif requested_step in step_to_frame:
                selected_step = requested_step
                selected_frame = step_to_frame[requested_step]
            else:
                prior = [
                    (int(step), frame) for (step, frame) in history if int(step) <= requested_step
                ]
                if prior:
                    (selected_step, selected_frame) = prior[-1]
                else:
                    (selected_step, selected_frame) = (0, anchor)
            selected_steps.append(int(selected_step))
            selected_frames.append(selected_frame)
        condition_frames = [anchor, *selected_frames]
        condition = _uint8_to_normalized(condition_frames).unsqueeze(0)
        condition = condition.to(device=image.device, dtype=image.dtype)
        metadata = {
            "enabled": True,
            "env_idx": env_idx,
            "current_step": current_step,
            "requested_offsets": requested_offsets,
            "selected_steps": [0, *selected_steps],
            "rgb_frames": len(condition_frames),
            "latent_frames": (len(condition_frames) - 1) // 4 + 1,
            "stride_native_steps": self.history_stride,
        }
        return (condition.contiguous(), metadata)


    def _seeds(self, env_idx: int = 0) -> tuple[Optional[int], Optional[int]]:
        env_idx = int(env_idx)
        replan_index = int(self._inference_index_by_env.get(env_idx, 0))
        self._inference_index_by_env[env_idx] = replan_index + 1
        self._inference_index += 1
        if self.base_seed is None:
            return (None, None)
        action_seed = int(self.base_seed + replan_index)
        video_seed = int(action_seed + self.video_seed_offset)
        return (action_seed, video_seed)


    def reset(self) -> None:
        self._inference_index = 0
        for name in self._PER_ENV_STATE:
            state = getattr(self, name, None)
            if state is not None:
                state.clear()


    def reset_env(self, env_idx: int) -> None:
        """Clear the episode state of one environment only."""
        for name in self._PER_ENV_STATE:
            state = getattr(self, name, None)
            if state is not None:
                state.pop(int(env_idx), None)

