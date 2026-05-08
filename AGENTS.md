# Repository Guidelines

These guidelines help contributors work consistently across the ML simulation and ESP32 firmware code.

## Project Structure & Module Organization
- `ml/`: Machine learning and simulation (SUMO + Gymnasium). Tests in `ml/tests` with `pytest.ini`.
- `hardware/`: ESP32 firmware (PlatformIO/Arduino). Unit and integration tests under `hardware/test`.
- `config/`: Repository-wide config placeholders; extend as needed.
- `results/`: Local outputs, logs, and artifacts (don’t commit large binaries).
- `tests/`: Top-level placeholder; most tests live in `ml/tests` and `hardware/test`.

## Build, Test, and Development Commands
- ML (Dockerized):
  - `cd ml && ./run_docker.sh test`: Run unit tests headless with coverage.
  - `cd ml && ./run_docker.sh demo`: Launch SUMO GUI demo (requires X11/WSLg/XQuartz).
  - `cd ml && ./run_docker.sh train`: Start RL training.
  - `cd ml && ./run_docker.sh gui`: Interactive shell inside container.
- Firmware (PlatformIO):
  - `cd hardware && pio run -e esp32dev`: Build for ESP32 DevKit.
  - `cd hardware && pio test -e native`: Run C++ unit tests on host.
  - `cd hardware && pio device monitor -b 115200`: Serial monitor.

## Coding Style & Naming Conventions
- Python (ml/): PEP8, 4 spaces, `snake_case` for functions/variables, `CapWords` for classes, type hints where reasonable. Files: `lower_snake_case.py`.
- C++/Arduino (hardware/): 4 spaces; braces on same line. Classes `PascalCase`, functions/variables `camelCase`, constants/macros `UPPER_SNAKE_CASE`. Shared protocol structs (e.g., `hardware/src/network/protocol/V2VMessage.h`) must not change layout without coordination.

## Testing Guidelines
- Python: `pytest` with markers (`unit`, `integration`, `slow`, `statistical`). Naming: `test_*.py`. Coverage is collected for `espnow_emulator`; HTML report is produced in container.
- Firmware: PlatformIO Unity tests in `hardware/test/*`. Use `-e native` for host tests; device-specific suites via dedicated envs (see `platformio.ini`).

## Commit & Pull Request Guidelines
- Commits: Imperative mood, concise subject (≤72 chars), meaningful body when needed. Reference issues (e.g., `#123`). Example: `fix: correct hop count handling in MeshRelayPolicy`.
- Branching: Feature branches targeting `develop` (merge or squash on approval).
- PRs: Clear description, scope (ml/ or hardware/), linked issues, test plan (`./run_docker.sh test` output, or `pio test`), and artifacts/screenshots when UI/plots are affected.

## Security & Configuration Tips
- Do not commit credentials or large datasets/logs; `.gitignore` and `.dockerignore` are in place—extend as needed.
- GUI demos require X11/WSLg/XQuartz per `ml/run_docker.sh` notes.
