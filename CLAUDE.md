# Project 4 - UGOT "Bob" bottle experiments

## Python
Always use **Python 3.11**: `py -3.11` to run and test (not 3.8).

## Commands (PowerShell, from this folder)
- Geometry tests: `py -3.11 -m pytest test_geometry.py -v`
- Equidistant experiment: `py -3.11 equidistant.py` (= `--stage full`: REAL DRIVING)
  - `--dry-run` (wheels never move), `--stage search` / `--stage measure` (turns only, no driving)
  - other options: `--ip`, `--port`, `--show`, `--no-browser`
  - results: one row per full run in `equidistant_results.csv`, full log in `logs/equidistant_*.log`
- Dashboard: http://127.0.0.1:5000
- Older experiments: `py -3.11 bottle_detect.py`, `py -3.11 bottle_orbit.py`
  (their file headers still say 3.8 - those files are not edited)

## Robot
Bob, UGOT mecanum, IP 172.20.10.12.

## Bottles + camera (config section of equidistant.py)
- Two identical small Heineken beer bottles: BOTTLE_H_CM = 22.5, BOTTLE_D_CM = 5.5, BOTTLE_R_CM = 2.75
- FOCAL_PX = 480 (camera property, calibrated in bottle_orbit - not tied to the bottle)
- test_geometry.py uses its own fixed values (32 / 8 / 4) on purpose: hand-checked math,
  independent of the bottles in use.

## Rules
- Do NOT modify bottle_detect.py, bottle_orbit.py, bottle_hunter.py, dashboard.html,
  orbit_dashboard.html - import / subclass instead.
- Math lives in geometry.py (no cv2/ugot imports) and is tested in test_geometry.py.
- Constants (bottle size, focal, limits) live in the config section of equidistant.py.
- Work one step at a time and wait for the user's OK. Reply in English, short and simple.
- Safety: nothing moves before START; STOP stops the wheels at once; Ctrl+C / crash -> wheels stop first.
