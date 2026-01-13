import os

import pytest


@pytest.fixture
def scenario_path():
    """Path to base SUMO scenario."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(base_dir, "scenarios", "base", "scenario.sumocfg")
