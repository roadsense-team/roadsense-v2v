# Repository Guidelines

## Project Structure & Module Organization
- ml/: Python simulation, RL, and SUMO integration
  - envs/, espnow_emulator/, models/, policies/, training/, scripts/, tests/, scenarios/
- hardware/: ESP32 firmware (PlatformIO)
  - src/, include/, test/, lib/
- Key configs: ml/pytest.ini, ml/requirements.txt, hardware/platformio.ini

## Build, Test, and Development Commands
- Python setup: `python -m venv .venv && . .venv/Scripts/Activate.ps1 && pip install -r ml/requirements.txt`
- ML tests (root): `pytest -c ml/pytest.ini -m "unit and not slow"`
- ML integration tests: `pytest -c ml/pytest.ini -m integration`
- ML Docker workflows (from ml/): `./run_docker.sh test | demo | train`
- Run demos: `python ml/examples/demo_emulator.py` or `python ml/demo_convoy_gui.py`
- Training: `python ml/training/train_convoy.py`
- Firmware build (from hardware/): `pio run -e esp32dev`
- Firmware tests: `pio test -e esp32dev` (all) or `pio test -e esp32dev_integration`
- Upload (set upload_port in platformio.ini): `pio run -e esp32dev -t upload`

## Coding Style & Naming Conventions
- Python: 4‑space indent, PEP8; type hints preferred. Modules/files: `snake_case.py`; classes: `PascalCase`; functions/vars: `snake_case`.
- C++ (Arduino/PlatformIO): 4‑space indent; headers in `include/`, sources in `src/`. Classes `PascalCase`, methods `camelCase`, constants `UPPER_SNAKE`, macros `UPPER_SNAKE`.
- Tests: file names `test_*.py` (pytest) and `hardware/test/*/test_main.cpp` (Unity).

## Testing Guidelines
- Pytest config: see `ml/pytest.ini` (markers: unit, integration, statistical, slow). Example: `-m "unit and not slow"`.
- Coverage: enabled via pytest-cov; HTML report in `htmlcov/`. Maintain or improve coverage for `espnow_emulator` and core envs.
- Hardware tests: Unity via `pio test`. Default env skips integration tests; use `esp32dev_integration` to include them.

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise subject, scope where helpful (e.g., `hardware: add QMC5883L driver`). Reference issues: `#123`.
- PRs: clear description, linked issues, test plan/output, and relevant screenshots/logs. Include impact on ML training, SUMO assets, or firmware.

## Security & Configuration Tips
- SUMO versions match Docker image (traci/sumolib ~1.25.0). Prefer Docker for integration tests.
- Do not commit secrets or local upload ports. Large artifacts (models/logs) should be excluded.
