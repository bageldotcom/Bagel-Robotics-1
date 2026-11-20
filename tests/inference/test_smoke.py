"""Check lossless RGB transport inputs without model tensors or a GPU."""
import numpy as np
import pytest

from bagel_robotics.inference.smoke import check_reset, rgb_transport, synthetic_observation


def test_png_transport_preserves_pixels_and_channel_order():
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[..., 0] = 231
    image[..., 1] = 79
    image[..., 2] = 17
    assert np.array_equal(rgb_transport(image, encoded=True), image)
    assert np.array_equal(rgb_transport(image, encoded=False), image)


def test_raw_and_encoded_observations_match_exactly():
    raw = synthetic_observation(encoded=False)
    encoded = synthetic_observation(encoded=True)
    for name in raw["observation"]:
        np.testing.assert_array_equal(raw["observation"][name]["rgb"], encoded["observation"][name]["rgb"])
    np.testing.assert_array_equal(raw["joint_action"]["vector"], encoded["joint_action"]["vector"])


def test_smoke_reset_rejects_remaining_episode_state():
    class Policy:
        _PER_ENV_STATE = ("history",)
        history = {0: "stale"}

        def reset(self):
            pass

    with pytest.raises(ValueError, match="history"):
        check_reset(Policy())
