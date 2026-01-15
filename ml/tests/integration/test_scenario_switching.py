import platform
import subprocess

import pytest

from envs.convoy_env import ConvoyEnv


@pytest.mark.integration
@pytest.mark.skipif(
    platform.system() != "Linux",
    reason="pgrep not available on non-Linux systems",
)
def test_multiple_resets_no_process_leaks(dataset_dir):
    """Verify 10 consecutive resets don't leak SUMO processes."""
    env = ConvoyEnv(dataset_dir=str(dataset_dir), scenario_mode="train")

    try:
        for _ in range(10):
            obs, info = env.reset()
            assert "scenario_id" in info
            assert obs["ego"].shape == (4,)
    finally:
        env.close()

    result = subprocess.run(
        ["pgrep", "-c", "sumo"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, f"Found zombie SUMO processes: {result.stdout}"
