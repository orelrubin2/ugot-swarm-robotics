# equidistant.py
# Equidistant-point experiment with Bob (UGOT mecanum, see README in the chat).
#
# STEP 4 (this version): the full experiment
#   SEARCH_PAIR -> CENTER_PAIR -> MEASURE -> (safety checks) -> RETURN to heading 0
#   -> DRIVE s cm along the axis -> SEARCH_PAIR -> CENTER_PAIR -> MEASURE -> DONE
#   Result: ratio = d_max / d_min, error %, compared to the predicted r.
#   One row per full run in equidistant_results.csv + a full log in logs/.
#
# Place Bob at the origin facing the x axis: his heading at START = "heading 0".
#
# Run (from the project folder), dashboard: http://127.0.0.1:5000
#   py -3.11 equidistant.py --dry-run               flow test, wheels never move
#   py -3.11 equidistant.py --stage search          real: find + center the pair, stop
#   py -3.11 equidistant.py --stage measure         real: ... + measure, compute s, stop
#   py -3.11 equidistant.py                         real: FULL experiment (drives!)
#   options: --ip 172.20.10.12  --port 5000  --show  --no-browser
#
# Reuses (unchanged): bottle_detect (Bob, BobCamera, Recorder, Hub, make_handler),
# bottle_orbit (Drive, RunLog, MissionSignals), geometry (all math, test_geometry.py).

import argparse
import csv
import math
import statistics
import threading
import time
import webbrowser
from datetime import datetime
from http.server import ThreadingHTTPServer
from pathlib import Path

import cv2
from ultralytics import YOLO

import bottle_detect as bd
import bottle_orbit as bo
import geometry as geo

# ------------------------------------------------------------------ config ---
ROBOT_IP = bd.ROBOT_IP          # 172.20.10.12
IMGSZ = 480                     # YOLO input size (same as bottle_orbit)
CONFIDENCE = 0.25               # same as bottle_orbit (keeps far bottles)

# Camera + bottles (cm, px)
FOCAL_PX = 480.0                # calibrated in bottle_orbit
# Both bottles are identical small Heineken beer bottles
BOTTLE_H_CM = 22.5              # height
BOTTLE_D_CM = 5.5               # diameter (also used for distance from box width)
BOTTLE_R_CM = 2.75              # radius: front surface -> bottle center
CAM_FWD_OFFSET_CM = 0.0         # camera distance ahead of the turning center (TO MEASURE)

# Experiment safety limits
MAX_TRAVEL_CM = 150.0           # |s| larger than this -> abort
MIN_DX_CM = 10.0                # |x1 - x2| smaller than this -> abort (no stable solution)
PATH_CLEARANCE_CM = 20.0        # a bottle this close to the driving path -> abort

# Detection checks
EDGE_CUT_PX = 5                 # box within 5 px of the left/right edge -> "cut", not usable
TOP_CUT_PX = bd.TOP_CUT_PX      # box top within 5 px of the frame top -> distance from width
PAIR_CENTER_TOL_PX = 40         # midpoint of the pair "centered" within +-40 px

# SEARCH_PAIR (stop-and-look, like bottle_orbit SEARCH)
SEARCH_STEP_DEG = 30            # turn RIGHT 30 deg between look positions
SEARCH_TURN_SPEED = 45          # deg/s (bottle_orbit / bottle_hunter)
SEARCH_MAX_TURNS = 12           # 12 x 30 = full circle, then STOPPED
PAIR_LOOK_FRAMES = 6            # fresh frames per look...
PAIR_MIN_HITS = 3               # ...both bottles fully visible in >= 3 of them

# CENTER_PAIR (small turns until the pair midpoint is near the image center)
CENTER_TURN_SPEED = 30          # bottle_orbit ALIGN_TURN_SPEED
CENTER_MAX_TRIES = 6            # then measure anyway (centering only reduces lens error)
MIN_TURN_STEP_DEG = 2           # smaller turns don't move the wheels
NO_PROGRESS_PX = 5              # midpoint moved less than this -> next turn +1 deg

# MEASURE (Bob is still and settled: Drive waits SETTLE_SEC after every turn)
MEASURE_MAX_FRAMES = 15         # look at up to 15 fresh frames...
MEASURE_ENOUGH = 10             # ...stop early at 10 good ones
MEASURE_MIN_GOOD = 6            # fewer than 6 good frames -> STOPPED

# DRIVE along the heading-0 axis (timed move, like bottle_hunter.Robot.move)
DRIVE_SPEED_CM_S = 10           # cm/s
DRIVE_EXTRA_SEC = 0.5           # margin on top of distance/speed before the safety stop
MOVE_FORWARD, MOVE_BACKWARD, UNIT_CM = 0, 1, 1      # ugot constants (bottle_hunter.py)

STATE_LED = {"SEARCH_PAIR": (255, 0, 0), "CENTER_PAIR": (255, 255, 0),
             "MEASURE": (255, 255, 0), "RETURN": (255, 255, 0), "DRIVE": (0, 0, 255),
             "DONE": (0, 255, 0), "STOPPED": (255, 0, 0)}
STATE_SOUND = {"SEARCH_PAIR": "Searching", "CENTER_PAIR": "Centering",
               "MEASURE": "Measuring", "RETURN": "Turning back", "DRIVE": "Driving",
               "DONE": "Done", "STOPPED": "Stopped"}

CSV_FIELDS = ["time", "theta_deg", "d1_cm", "d2_cm", "x1_cm", "y1_cm", "x2_cm", "y2_cm",
              "s_cm", "predicted_r_cm", "final_theta_deg", "final_d1_cm", "final_d2_cm",
              "ratio", "error_pct",
              # extras after the requested columns
              "status", "drove_cm", "mean_d_minus_r_cm", "drift_b1_cm", "drift_b2_cm",
              "log", "message"]

SCRIPT_DIR = bd.SCRIPT_DIR
DASHBOARD_FILE = SCRIPT_DIR / "equidistant_dashboard.html"
LOGS_DIR = SCRIPT_DIR / "logs"
RESULTS_CSV = SCRIPT_DIR / "equidistant_results.csv"     # written from Step 4

GEO = dict(focal_px=FOCAL_PX, bottle_h_cm=BOTTLE_H_CM,
           bottle_d_cm=BOTTLE_D_CM, bottle_r_cm=BOTTLE_R_CM)

# Box colors on the video (B, G, R) - same colors as on the dashboard map
BOTTLE_BGR = [(238, 211, 34), (182, 114, 244)]   # B1 cyan, B2 pink
CUT_BGR = (0, 165, 255)                          # orange = cut by the image edge


# ------------------------------------------------------------- detection ---
def find_bottles(model, frame):
    """YOLO on one frame -> (boxes, n_raw).
    boxes: up to 2 bottles as (x1, y1, x2, y2, conf), the 2 most confident,
    sorted LEFT to RIGHT. n_raw: how many bottle boxes YOLO found in total."""
    result = model(frame, classes=[bd.BOTTLE_CLASS_ID], conf=CONFIDENCE,
                   imgsz=IMGSZ, verbose=False)[0]
    boxes = [(int(x1), int(y1), int(x2), int(y2), float(c))
             for (x1, y1, x2, y2), c in zip(result.boxes.xyxy.tolist(),
                                            result.boxes.conf.tolist())]
    best = sorted(boxes, key=lambda b: b[4], reverse=True)[:2]
    best.sort(key=lambda b: b[0] + b[2])                  # by center x
    return best, len(boxes)


def boxes_overlap(a, b):
    """True if two (x1, y1, x2, y2, ...) boxes intersect at all."""
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def describe_bottle(box, label, frame_w, theta_deg):
    """One box -> pixel numbers + distance/angle/position (dict, JSON friendly)."""
    x1, y1, x2, y2, conf = box
    cx = (x1 + x2) / 2.0
    b = {
        "label": label, "box": [x1, y1, x2, y2], "conf": round(conf, 2),
        "cx": round(cx, 1), "offset": round(cx - frame_w / 2.0, 1),   # + = right of center
        "w": x2 - x1, "h": y2 - y1,
        "top_cut": y1 <= TOP_CUT_PX,
        "edge_cut": x1 <= EDGE_CUT_PX or x2 >= frame_w - 1 - EDGE_CUT_PX,
        "d": None,
    }
    try:
        m = geo.measure_bottle(b["h"], b["w"], b["top_cut"], b["offset"], **GEO)
    except ValueError:
        return b                                          # empty box: no distance
    p = geo.bottle_position(m.d_cm, m.beta_deg, theta_deg, CAM_FWD_OFFSET_CM)
    b.update(method=m.method, z=round(m.z_cm, 1), beta=round(m.beta_deg, 2),
             d=round(m.d_cm, 1), x=round(p.x, 1), y=round(p.y, 1), phi=round(p.phi_deg, 2))
    return b


def analyze(boxes, n_raw, frame_w, theta_deg):
    """All boxes of one frame -> the 'view' shown on the dashboard and in the log."""
    bottles = [describe_bottle(bx, f"B{i + 1}", frame_w, theta_deg)
               for i, bx in enumerate(boxes)]
    overlap = len(boxes) == 2 and boxes_overlap(boxes[0], boxes[1])
    pair_full = (len(bottles) == 2
                 and all(not b["edge_cut"] and b["d"] is not None for b in bottles))
    view = {"bottles": bottles, "n_raw": n_raw, "overlap": overlap,
            "pair_full": pair_full, "mid_offset": None, "preview": None,
            "warnings": []}
    if len(bottles) == 2:
        view["mid_offset"] = round((bottles[0]["cx"] + bottles[1]["cx"]) / 2.0
                                   - frame_w / 2.0, 1)

    w = view["warnings"]
    if not bottles:
        w.append("no bottle seen")
    elif len(bottles) == 1:
        w.append("only 1 bottle seen - need both in one frame")
    if n_raw > 2:
        w.append(f"{n_raw} boxes found - using the 2 most confident")
    if overlap:
        w.append("B1/B2 boxes overlap - YOLO may merge close bottles")
    for b in bottles:
        if b["edge_cut"]:
            w.append(f"{b['label']} cut by the image edge - not usable")
        if b["top_cut"]:
            w.append(f"{b['label']} top cut - distance from box width")

    # LIVE PREVIEW of the target point from this one frame (not a measurement)
    if pair_full:
        p1, p2 = [(b["x"], b["y"]) for b in bottles]
        eq = geo.equidistant_point(p1, p2, min_dx_cm=MIN_DX_CM, max_travel_cm=MAX_TRAVEL_CM)
        blocked = (geo.path_blocked([p1, p2], eq.s, clearance_cm=PATH_CLEARANCE_CM)
                   if eq.ok else None)
        view["preview"] = {
            "ok": eq.ok and blocked is None,
            "s": None if eq.s is None else round(eq.s, 1),
            "r": None if eq.r is None else round(eq.r, 1),
            "reason": eq.reason or blocked or "",
        }
    return view


# ------------------------------------------------------------- annotation ---
def annotate(frame, view, footer):
    """Center line, both boxes (B1/B2 + distance + angle), pair midpoint, warnings."""
    fh, fw = frame.shape[:2]
    mid = fw // 2
    cv2.line(frame, (mid, 0), (mid, fh), (200, 200, 200), 1)
    for x in (mid - PAIR_CENTER_TOL_PX, mid + PAIR_CENTER_TOL_PX):
        cv2.line(frame, (x, 0), (x, fh), (110, 110, 110), 1)

    for i, b in enumerate(view["bottles"]):
        x1, y1, x2, y2 = b["box"]
        color = CUT_BGR if b["edge_cut"] else BOTTLE_BGR[i]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cx = int(b["cx"])
        cv2.line(frame, (cx, y1), (cx, y2), color, 1)
        text = b["label"]
        if b["d"] is not None:
            text += f" {b['d']:.0f}cm {b['beta']:+.1f}deg"
        if b["edge_cut"]:
            text += " EDGE CUT"
        cv2.putText(frame, text, (max(x1, 2), max(y1 - 8, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(frame, f"{b['conf']:.2f}", (max(x1, 2), min(y2 + 18, fh - 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    if view["mid_offset"] is not None:                    # midpoint of the pair
        mx = int(mid + view["mid_offset"])
        cv2.drawMarker(frame, (mx, fh // 2), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 18, 2)

    for k, msg in enumerate(view["warnings"][:4]):
        cv2.putText(frame, msg, (10, 24 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 255) if "overlap" in msg or "cut by" in msg else (0, 220, 255), 2)
    cv2.putText(frame, footer, (10, fh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)


# --------------------------------------------------------------- logging ---
class EqLog(bo.RunLog):
    """bottle_orbit.RunLog, but the file is logs/equidistant_YYYY-MM-DD_HH-MM-SS.log"""

    def __init__(self, folder):
        Path(folder).mkdir(exist_ok=True)
        self.path = Path(folder) / datetime.now().strftime("equidistant_%Y-%m-%d_%H-%M-%S.log")
        self.f = open(self.path, "a", encoding="utf-8", buffering=1)
        self.t0 = time.time()
        self.lock = threading.Lock()


def log_view(log, frame_id, state, view):
    parts = []
    for b in view["bottles"]:
        p = (f"{b['label']} cx={b['cx']:.0f} off={b['offset']:+.0f} w={b['w']} h={b['h']} "
             f"conf={b['conf']:.2f} top={int(b['top_cut'])} edge={int(b['edge_cut'])}")
        if b["d"] is not None:
            p += (f" d={b['d']:.1f} beta={b['beta']:+.2f} x={b['x']:.1f} y={b['y']:.1f}"
                  f" ({b['method']})")
        parts.append(p)
    line = f"DET frame={frame_id:<6} {state:<11} n={len(view['bottles'])} raw={view['n_raw']}"
    line += " | " + (" | ".join(parts) if parts else "none")
    if view["overlap"]:
        line += " | OVERLAP"
    pv = view["preview"]
    if pv:
        line += (f" | preview s={pv['s']} r={pv['r']} ok={int(pv['ok'])}"
                 + (f" ({pv['reason']})" if pv["reason"] else ""))
    log.write(line)


# ------------------------------------------------------------ measurement ---
def combine_frames(views, frames_seen, theta_deg):
    """MEASURE: median of the RAW box numbers over the good frames, then the math.
    views: frames where both bottles were fully visible (sorted left -> right)."""
    bottles = []
    for i in (0, 1):
        bs = [v["bottles"][i] for v in views]
        h = statistics.median(b["h"] for b in bs)
        w = statistics.median(b["w"] for b in bs)
        off = statistics.median(b["offset"] for b in bs)
        top_cut = 2 * sum(b["top_cut"] for b in bs) > len(bs)       # majority vote
        m = geo.measure_bottle(h, w, top_cut, off, **GEO)
        p = geo.bottle_position(m.d_cm, m.beta_deg, theta_deg, CAM_FWD_OFFSET_CM)
        ds = [b["d"] for b in bs]
        bottles.append({
            "label": f"B{i + 1}", "h": h, "w": w, "offset": round(off, 1), "top_cut": top_cut,
            "method": m.method, "z": round(m.z_cm, 1), "beta": round(m.beta_deg, 2),
            "d": round(m.d_cm, 1), "phi": round(p.phi_deg, 2),
            "x": round(p.x, 1), "y": round(p.y, 1),
            "d_spread": round(max(ds) - min(ds), 1),          # noise over the frames
        })
    return {"theta": round(theta_deg, 1), "frames": frames_seen, "good": len(views),
            "overlap_frames": sum(1 for v in views if v["overlap"]), "bottles": bottles}


def compute_target(m):
    """Measurement -> target point + all safety checks (dict for the dashboard)."""
    p1, p2 = [(b["x"], b["y"]) for b in m["bottles"]]
    eq = geo.equidistant_point(p1, p2, min_dx_cm=MIN_DX_CM, max_travel_cm=MAX_TRAVEL_CM)
    blocked = (geo.path_blocked([p1, p2], eq.s, clearance_cm=PATH_CLEARANCE_CM)
               if eq.ok else None)
    return {"ok": eq.ok and blocked is None,
            "s": None if eq.s is None else round(eq.s, 1),
            "r": None if eq.r is None else round(eq.r, 1),
            "reason": eq.reason or blocked or ""}


def append_csv(results, status, message, log_name):
    """One row per full run (also for aborted runs - 'status' says how it ended)."""
    m1, m2 = results.get("m1"), results.get("m2")
    t, f = results.get("target") or {}, results.get("final") or {}
    drift = results.get("drift") or []

    def bot(m, i, key):
        return m["bottles"][i][key] if m else ""

    row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "theta_deg": m1["theta"] if m1 else "",
        "d1_cm": bot(m1, 0, "d"), "d2_cm": bot(m1, 1, "d"),
        "x1_cm": bot(m1, 0, "x"), "y1_cm": bot(m1, 0, "y"),
        "x2_cm": bot(m1, 1, "x"), "y2_cm": bot(m1, 1, "y"),
        "s_cm": t.get("s", ""), "predicted_r_cm": t.get("r", ""),
        "final_theta_deg": m2["theta"] if m2 else "",
        "final_d1_cm": bot(m2, 0, "d"), "final_d2_cm": bot(m2, 1, "d"),
        "ratio": f.get("ratio", ""), "error_pct": f.get("error_pct", ""),
        "status": status, "drove_cm": results.get("drove_cm", ""),
        "mean_d_minus_r_cm": f.get("r_error_cm", ""),
        "drift_b1_cm": drift[0]["drift"] if len(drift) > 0 else "",
        "drift_b2_cm": drift[1]["drift"] if len(drift) > 1 else "",
        "log": log_name, "message": message,
    }
    new_file = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(row)


# ------------------------------------------------------------------ wheels ---
class AxisDrive(bo.Drive):
    """bottle_orbit.Drive (timed turns, safety stop, watchdog, STOP) + a timed
    straight drive along Bob's current heading, with the same safety pattern:
      mecanum_move_speed_times -> wait distance/speed (+margin) -> mecanum_stop
      -> settle -> busy flag off.   STOP cancels it at once.
    It also keeps Bob's position on the axis from the drive commands."""

    def __init__(self, bob, dry_run, log):
        super().__init__(bob, dry_run, log)
        self.pos_cm = 0.0              # finished drives (cm, + = forward)
        self._job = None               # running drive: (t_start, signed_cm, speed, gen)

    def _partial(self, job, now):
        t0, signed, speed, _ = job
        return math.copysign(min(max(now - t0, 0.0) * speed, abs(signed)), signed)

    def position_cm(self):
        """Position on the axis (cm): finished drives + the part of a running one."""
        with self.lock:
            p = self.pos_cm
            if self._job:
                p += self._partial(self._job, time.time())
            return p

    def drive_cm(self, cm, speed, reason=""):
        """Drive cm along the heading (> 0 forward, < 0 backward), then stop + settle.
        Returns False if nothing was sent (disabled or 0 cm)."""
        amount = int(round(abs(cm)))                   # ugot wants whole cm
        if amount == 0:
            return False
        direction = MOVE_FORWARD if cm > 0 else MOVE_BACKWARD
        signed = amount if cm > 0 else -amount
        with self.lock:
            if not self.enabled:
                return False
            self.moving = False
            self.turning = True                        # "busy": the mission ignores frames
            self.turn_gen += 1
            gen = self.turn_gen
            self.last_sent = (speed if cm > 0 else -speed, 0, 0)
            t_start = time.time()
            self._job = (t_start, signed, speed, gen)
        side = "FORWARD" if cm > 0 else "BACKWARD"
        self.log.write(f"DRIVE {side} {amount} cm @ {speed} cm/s  ({reason})")
        print(f"[drive] {side} {amount} cm @ {speed} cm/s  ({reason})"
              + ("  [dry-run, not sent]" if self.dry_run else ""))

        def worker():
            if not self.dry_run:
                try:
                    self.got.mecanum_move_speed_times(direction, speed, amount, UNIT_CM)
                except Exception as e:
                    print("drive failed:", e)
                    self.log.write(f"DRIVE FAILED: {e}")
            end = t_start + amount / float(speed) + DRIVE_EXTRA_SEC
            while time.time() < end:                   # wait even if the call returned at once
                if gen != self.turn_gen:               # STOP pressed: stop() did the rest
                    return
                time.sleep(0.02)
            with self.lock:
                if gen != self.turn_gen:
                    return
                self.pos_cm += signed
                self._job = None
            if not self.dry_run:
                try:
                    self.got.mecanum_stop()            # safety stop at the end of every drive
                except Exception as e:
                    print("mecanum_stop failed:", e)
            time.sleep(bo.SETTLE_SEC)
            with self.lock:
                if gen != self.turn_gen:
                    return
                self.turning = False
                self.last_sent = (0, 0, 0)
            self.log.write(f"DRIVE done + settled ({time.time() - t_start:.2f} s)")

        threading.Thread(target=worker, daemon=True).start()
        return True

    def stop(self, quiet=False, disable=False, reason=""):
        with self.lock:                                # count the part already driven
            if self._job:
                part = self._partial(self._job, time.time())
                self.pos_cm += part
                self._job = None
                self.log.write(f"DRIVE interrupted after ~{part:+.0f} cm (estimated from time)")
        super().stop(quiet=quiet, disable=disable, reason=reason)


# --------------------------------------------------------------- mission ---
class EqSignals(bo.MissionSignals):
    """bottle_orbit.MissionSignals with this experiment's states (LED + voice)."""

    def _loop(self):
        while True:
            state = self.q.get()
            if state is None:
                return
            try:
                if state in STATE_LED:
                    self.bob.led(*STATE_LED[state])
                if state in STATE_SOUND:
                    self.bob.say(STATE_SOUND[state])
            except Exception as e:
                print("LED/sound failed:", e)


ACTIVE = ("SEARCH_PAIR", "CENTER_PAIR", "MEASURE", "RETURN", "DRIVE")


class Mission:
    """Round 1 (at the origin):  SEARCH_PAIR -> CENTER_PAIR -> MEASURE -> safety checks
    -> RETURN (turn back to heading 0) -> DRIVE s cm
    Round 2 (at (s, 0)):         SEARCH_PAIR -> CENTER_PAIR -> MEASURE -> DONE
    STOPPED on STOP / any abort.  --stage search|measure end early (no driving).

    Every decision uses only frames that arrived AFTER Bob stopped and settled
    (a 'look window', same idea as bottle_orbit)."""

    def __init__(self, drive, signals, log, stage):
        self.drive = drive
        self.signals = signals
        self.log = log
        self.stage = stage                 # "search" | "measure" | "full"
        self.lock = threading.Lock()
        self.state = "IDLE"
        self.message = "Place Bob at the origin facing heading 0, then press START"
        self.phase = ""                    # what the current look window is doing
        self.theta0 = 0.0                  # commanded heading at START (= heading 0)
        self.pos0 = 0.0                    # drive position at START (= x 0)
        self.round = 1                     # 1 = at the origin, 2 = at (s, 0)
        self.csv_written = False
        self.results = None
        self.look = None
        self.search_turns = 0
        self.center_tries = 0
        self.center_prev = None
        self.center_boost = 0
        self.last_center_turn = None
        self.center_note = ""

    def theta(self):
        """Heading (deg, + = left) since START, from the timed turn commands."""
        return self.drive.heading_cmd() - self.theta0

    def robot_x(self):
        """Bob's position on the axis (cm) since START, from the drive commands."""
        return self.drive.position_cm() - self.pos0

    # --- called from the web server thread ---
    def start(self):
        with self.lock:
            if self.state in ACTIVE:
                return
            self.theta0 = self.drive.heading_cmd()      # here = heading 0
            self.pos0 = self.drive.position_cm()        # here = x 0
            self.round = 1
            self.csv_written = False
            self.results = {}
            self.drive.enable()
            self.log.write(f"START pressed: heading 0 + x 0 set here (stage {self.stage})")
            self._enter_search("Mission started - looking for both bottles")

    def stop_button(self):
        self.drive.stop(disable=True, reason="STOP button")      # wheels first
        with self.lock:
            active = self.state in ACTIVE
            self.look = None
            self.phase = ""
            self._set("STOPPED", "STOP button pressed - press START to restart")
            if active:
                self._save_csv("STOPPED", "STOP button")

    # --- helpers (caller holds self.lock) ---
    def _set(self, state, msg):
        self.message = msg
        if state != self.state:
            print(f">>> {self.state} -> {state}: {msg}")
            self.log.write(f"STATE {self.state} -> {state}: {msg}")
            self.state = state
            self.signals.set_state(state)
        else:
            self.log.write(f"MSG {msg}")

    def _finish(self, state, msg):
        self.drive.stop(disable=True, reason=f"{state}: {msg}")
        self.look = None
        self.phase = ""
        self._set(state, msg)
        self._save_csv(state, msg)

    def _save_csv(self, status, msg):
        """One CSV row per FULL run that got at least the first measurement."""
        if self.stage != "full" or not self.results or "m1" not in self.results \
                or self.csv_written:
            return
        self.csv_written = True
        try:
            append_csv(self.results, status, msg, self.log.path.name)
            self.log.write(f"CSV row added to {RESULTS_CSV.name} ({status})")
            print(f"results row added to {RESULTS_CSV.name}")
        except OSError as e:                           # e.g. the CSV is open in Excel
            self.log.write(f"CSV write FAILED: {e!r}")
            print("WARNING: could not write the CSV (open in Excel?):", e)
            self.message += f" - CSV NOT saved ({e.__class__.__name__}, file open in Excel?)"

    def _begin_look(self, purpose, frames):
        self.look = {"purpose": purpose, "need": frames, "marker": None,
                     "frames": 0, "hits": []}

    def _turn(self, degrees, speed, reason):
        """Timed turn (ends with a safety stop + settle). Not sent if disabled."""
        if not self.drive.turn_degrees(degrees, speed, reason):
            self.log.write(f"turn {degrees:+.1f} deg not sent ({reason})")

    # --- called once per processed frame from the main loop ---
    def step(self, view, frame_id):
        with self.lock:
            lk = self.look
            if lk is None or self.state not in ACTIVE:
                return
            if self.drive.turning:                     # moving or settling: blurry frames
                self.phase = "turning / settling"
                return
            if lk["marker"] is None:                   # Bob just became still
                lk["marker"] = frame_id                # only NEWER frames count
                return
            if frame_id <= lk["marker"]:
                return
            lk["frames"] += 1
            if view["pair_full"]:
                lk["hits"].append(view)
            {"search": self._look_search, "center": self._look_center,
             "center_undo": self._look_center_undo,
             "measure": self._look_measure,
             "return": self._look_return, "drive": self._look_drive}[lk["purpose"]](lk)

    # ---------------------------------------------------------- SEARCH_PAIR ---
    def _enter_search(self, msg):
        self.search_turns = 0
        self._begin_look("search", PAIR_LOOK_FRAMES)
        self._set("SEARCH_PAIR", msg)

    def _look_search(self, lk):
        hits, left = len(lk["hits"]), lk["need"] - lk["frames"]
        self.phase = (f"look {lk['frames']}/{lk['need']}: pair seen {hits}x "
                      f"(need {PAIR_MIN_HITS}) · turns {self.search_turns}/{SEARCH_MAX_TURNS}")
        if hits >= PAIR_MIN_HITS:
            self.log.write(f"SEARCH_PAIR: pair seen in {hits}/{lk['frames']} fresh frames")
            self._enter_center(lk["hits"])
        elif hits + left < PAIR_MIN_HITS:              # can't reach 3 here any more
            if self.search_turns >= SEARCH_MAX_TURNS:
                self._finish("STOPPED", "Both bottles never fully visible in one frame after a "
                             "full turn - move them closer together or move Bob back")
                return
            self.search_turns += 1
            self.message = f"Pair not in view - turning {SEARCH_STEP_DEG} deg right"
            self._begin_look("search", PAIR_LOOK_FRAMES)
            self._turn(-SEARCH_STEP_DEG, SEARCH_TURN_SPEED, f"search turn {self.search_turns}")

    # ---------------------------------------------------------- CENTER_PAIR ---
    def _enter_center(self, hits):
        self.center_tries = 0
        self.center_prev = None
        self.center_boost = 0
        self.last_center_turn = None
        self._set("CENTER_PAIR", "Both bottles found - centering their midpoint")
        self._center_decide(hits)

    def _center_done(self, note):
        self.center_note = note
        self.log.write(f"CENTER_PAIR done: {note}, theta={self.theta():+.1f} deg")
        if self.stage == "search":
            self._finish("DONE", f"Stage 'search' done: {note}, θ = {self.theta():+.1f}°")
        else:
            self._enter_measure(f"Measuring ({note})")

    def _center_decide(self, hits):
        mid = statistics.median(v["mid_offset"] for v in hits)     # px, + = right
        if abs(mid) <= PAIR_CENTER_TOL_PX:
            self._center_done(f"midpoint centered at {mid:+.0f} px")
            return
        if self.center_tries >= CENTER_MAX_TRIES:
            self._center_done(f"NOT fully centered after {CENTER_MAX_TRIES} tries ({mid:+.0f} px)")
            return
        # bottle_hunter's trick: last turn did nothing -> one more degree next time
        if (self.center_prev is not None and abs(mid - self.center_prev) < NO_PROGRESS_PX
                and (mid > 0) == (self.center_prev > 0)):
            self.center_boost += 1
        elif self.center_prev is not None:
            self.center_boost = max(0, self.center_boost - 1)
        self.center_prev = mid
        self.center_tries += 1
        angle = math.degrees(math.atan(mid / FOCAL_PX))            # + = midpoint right
        size = max(abs(angle), MIN_TURN_STEP_DEG) + self.center_boost
        turn = -math.copysign(size, angle)                         # right -> turn RIGHT
        self.last_center_turn = turn
        self.message = (f"Center try {self.center_tries}/{CENTER_MAX_TRIES}: midpoint "
                        f"{mid:+.0f} px -> turn {abs(turn):.0f} deg {'right' if turn < 0 else 'left'}")
        self._begin_look("center", PAIR_LOOK_FRAMES)
        self._turn(turn, CENTER_TURN_SPEED, f"center try {self.center_tries}: mid={mid:+.0f}px "
                   f"angle={angle:+.1f} boost={self.center_boost}")

    def _look_center(self, lk):
        hits, left = len(lk["hits"]), lk["need"] - lk["frames"]
        self.phase = f"look {lk['frames']}/{lk['need']}: pair seen {hits}x"
        if hits >= PAIR_MIN_HITS:
            self._center_decide(lk["hits"])
        elif hits + left < PAIR_MIN_HITS:              # the turn lost a bottle: undo it
            back = -self.last_center_turn if self.last_center_turn else 0
            self.log.write(f"CENTER_PAIR: pair lost after the turn -> undo {back:+.0f} deg")
            self.message = "Pair lost after centering turn - turning back"
            self.last_center_turn = None
            self._begin_look("center_undo", PAIR_LOOK_FRAMES)
            if back:
                self._turn(back, CENTER_TURN_SPEED, "undo last center turn")

    def _look_center_undo(self, lk):
        hits, left = len(lk["hits"]), lk["need"] - lk["frames"]
        self.phase = f"after undo: look {lk['frames']}/{lk['need']}: pair seen {hits}x"
        if hits >= PAIR_MIN_HITS:
            mid = statistics.median(v["mid_offset"] for v in lk["hits"])
            self._center_done(f"NOT fully centered - pair lost when centering ({mid:+.0f} px)")
        elif hits + left < PAIR_MIN_HITS:
            self._enter_search("Pair lost while centering - searching again")

    # -------------------------------------------------------------- MEASURE ---
    def _enter_measure(self, msg):
        self._begin_look("measure", MEASURE_MAX_FRAMES)
        self._set("MEASURE", msg)

    def _look_measure(self, lk):
        good, left = len(lk["hits"]), lk["need"] - lk["frames"]
        self.phase = (f"frame {lk['frames']}/{lk['need']}: {good} good "
                      f"(need {MEASURE_MIN_GOOD}, stop at {MEASURE_ENOUGH})")
        if good + left < MEASURE_MIN_GOOD:
            self._finish("STOPPED", f"Only {good} good frames out of {lk['frames']} "
                         f"(need {MEASURE_MIN_GOOD}) - see log")
            return
        if good < MEASURE_ENOUGH and left > 0:
            return
        theta = self.theta()
        m = combine_frames(lk["hits"], lk["frames"], theta)
        m["center_note"] = self.center_note
        self.results["m1" if self.round == 1 else "m2"] = m
        for b in m["bottles"]:
            self.log.write(f"MEASURE{self.round} {b['label']}: h={b['h']} w={b['w']} "
                           f"off={b['offset']:+.1f} ({b['method']}) Z={b['z']} "
                           f"beta={b['beta']:+.2f} d={b['d']} phi={b['phi']:+.2f} "
                           f"x={b['x']} y={b['y']} spread={b['d_spread']}")
        self.log.write(f"MEASURE{self.round} theta={theta:+.1f} x_robot={self.robot_x():+.1f} "
                       f"good={m['good']}/{m['frames']} overlap_frames={m['overlap_frames']}")
        if self.round == 1:
            self._after_first_measure(m)
        else:
            self._after_second_measure(m)

    def _after_first_measure(self, m):
        t = compute_target(m)
        self.results["target"] = t
        self.log.write(f"TARGET s={t['s']} r={t['r']} ok={t['ok']} {t['reason']}")
        if not t["ok"]:
            self._finish("STOPPED", f"Not safe to drive: {t['reason']}")
        elif self.stage != "full":
            self._finish("DONE", f"s = {t['s']:+.1f} cm, r = {t['r']:.1f} cm "
                                 f"(stage '{self.stage}': not driving)")
        else:
            self._enter_return()

    # ---------------------------------------------------- RETURN + DRIVE ---
    def _enter_return(self):
        theta = self.theta()
        s = self.results["target"]["s"]
        self._set("RETURN", f"Target s = {s:+.1f} cm - turning back to heading 0 "
                            f"({-theta:+.0f} deg)")
        self._begin_look("return", 1)                  # 1 still frame after the turn = done
        if abs(theta) >= 0.5:
            speed = SEARCH_TURN_SPEED if abs(theta) >= 10 else CENTER_TURN_SPEED
            self._turn(-theta, speed, "return to heading 0")

    def _look_return(self, lk):
        self.log.write(f"RETURN done: theta={self.theta():+.1f} deg")
        s = self.results["target"]["s"]
        self._set("DRIVE", f"Driving {abs(s):.0f} cm {'forward' if s > 0 else 'backward'} "
                           f"at {DRIVE_SPEED_CM_S} cm/s")
        self._begin_look("drive", 1)                   # 1 still frame after the drive = done
        if not self.drive.drive_cm(s, DRIVE_SPEED_CM_S, f"to the equidistant point s={s:+.1f}"):
            self.log.write("DRIVE: nothing sent (|s| < 0.5 cm) - measuring here")

    def _look_drive(self, lk):
        x = self.robot_x()
        self.results["drove_cm"] = round(x, 1)
        self.log.write(f"DRIVE finished: robot at x={x:+.1f} cm "
                       f"(target s={self.results['target']['s']:+.1f})")
        self.round = 2
        self._enter_search(f"At x = {x:+.0f} cm - looking for both bottles again")

    # ------------------------------------------------------- final result ---
    def _after_second_measure(self, m):
        xa = self.results["drove_cm"]
        for b in m["bottles"]:                         # back into the start frame
            b["x_start"] = round(geo.to_start_frame((b["x"], b["y"]), xa)[0], 1)
        d1, d2 = (b["d"] for b in m["bottles"])
        r = self.results["target"]["r"]
        f = geo.final_result(d1, d2, r)
        self.results["final"] = {"ratio": round(f.ratio, 4), "error_pct": round(f.error_pct, 2),
                                 "r_error_cm": round(f.r_error_cm, 1)}
        first = [(b["x"], b["y"]) for b in self.results["m1"]["bottles"]]
        second = [(b["x_start"], b["y"]) for b in m["bottles"]]
        self.results["drift"] = [
            {"label": f"B{i + 1}", "matched": f"B{j + 1}", "drift": round(dd, 1)}
            for i, j, dd in geo.match_bottles(first, second)]
        for dr in self.results["drift"]:
            self.log.write(f"DRIFT {dr['label']} (1st) <-> {dr['matched']} (2nd): "
                           f"{dr['drift']} cm")
        self.log.write(f"RESULT d1={d1} d2={d2} r={r} ratio={f.ratio:.4f} "
                       f"error={f.error_pct:.2f}% mean_d-r={f.r_error_cm:+.1f}")
        self._finish("DONE", f"ratio {f.ratio:.3f} (error {f.error_pct:.1f} %) - "
                             f"d = {d1:.1f} / {d2:.1f} cm, predicted r = {r:.1f} cm")

    def status(self):
        with self.lock:
            return {"mission": self.state, "message": self.message, "stage": self.stage,
                    "phase": self.phase if self.state in ACTIVE else "",
                    "round": self.round,
                    "theta": round(self.theta(), 1), "robot_x": round(self.robot_x(), 1),
                    "results": self.results, "log": self.log.path.name}


# --------------------------------------------------------------- dashboard ---
def make_eq_handler(hub, recorder, mission):
    """bottle_detect's handler (/status, /stream, /frame.jpg, /record/*)
    + this page and the START/STOP routes."""
    Base = bd.make_handler(hub, recorder)

    class EqHandler(Base):
        def do_GET(self):
            if self._route() == "/":
                self._send(200, DASHBOARD_FILE.read_bytes(), "text/html; charset=utf-8")
            else:
                super().do_GET()

        def do_POST(self):
            route = self._route()
            if route == "/mission/stop":
                mission.stop_button()
                print("[web] STOP pressed")
                self._send(200, b'{"ok": true}', "application/json")
            elif route == "/mission/start":
                mission.start()
                print("[web] START pressed")
                self._send(200, b'{"ok": true}', "application/json")
            else:
                super().do_POST()

    return EqHandler


# -------------------------------------------------------------------- main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default=ROBOT_IP)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--dry-run", action="store_true", help="wheels never move (Step 3+)")
    parser.add_argument("--stage", choices=["search", "measure", "full"], default="full",
                        help="stop after this stage (used from Step 3)")
    parser.add_argument("--show", action="store_true", help="also show an OpenCV window")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    print("connecting to Bob at", args.ip)
    bob = bd.Bob(args.ip)
    bob.got.mecanum_stop()                             # start from a known, stopped state

    print("loading", bd.YOLO_MODEL, f"(imgsz={IMGSZ}, conf={CONFIDENCE})")
    model = YOLO(bd.YOLO_MODEL)

    log = EqLog(LOGS_DIR)
    log.write(f"RUN start STEP4 ip={args.ip} stage={args.stage} dry_run={args.dry_run} "
              f"imgsz={IMGSZ} conf={CONFIDENCE} focal={FOCAL_PX} H={BOTTLE_H_CM} "
              f"D={BOTTLE_D_CM} R={BOTTLE_R_CM} cam_off={CAM_FWD_OFFSET_CM}")
    print("log file:", log.path)
    drive = AxisDrive(bob, args.dry_run, log)          # the ONLY place that moves wheels
    signals = EqSignals(bob)
    mission = Mission(drive, signals, log, args.stage)

    default_hook = threading.excepthook               # crash in any thread -> wheels stop
    def crash_hook(hook_args):
        drive.stop(disable=True, reason=f"CRASH in thread: {hook_args.exc_value!r}")
        log.write(f"CRASH in thread: {hook_args.exc_value!r}")
        default_hook(hook_args)
    threading.excepthook = crash_hook

    cam = bd.BobCamera(bob)
    hub = bd.Hub()
    recorder = bd.Recorder(bd.RECORDINGS_DIR, bd.RECORD_FPS)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_eq_handler(hub, recorder, mission))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{args.port}"
    mode = "DRY-RUN (wheels never move)" if args.dry_run else (
        "REAL WHEELS (turns + DRIVING)" if args.stage == "full" else "REAL WHEELS (turns only)")
    print(f"dashboard: {url}   mode: {mode}   stage: {args.stage}")
    print("Bob will not move until you press START. Ctrl+C here = stop + quit.")
    if not args.no_browser:
        webbrowser.open(url)

    last_id = 0
    fps, t_prev = 0.0, time.time()
    try:
        while True:
            frame_id, frame = cam.latest()
            if frame is None or frame_id == last_id:
                if args.show:
                    if cv2.waitKey(5) & 0xFF == ord("q"):
                        break
                else:
                    time.sleep(0.005)
                continue
            last_id = frame_id

            frame = frame.copy()
            boxes, n_raw = find_bottles(model, frame)
            view = analyze(boxes, n_raw, frame.shape[1], mission.status()["theta"])
            mission.step(view, frame_id)               # state machine -> timed turns
            ms = mission.status()

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-3))
            t_prev = now

            annotate(frame, view, f"{ms['mission']}  theta={ms['theta']:+.1f}  {fps:.1f} fps")
            log_view(log, frame_id, ms["mission"], view)
            recorder.write(frame)

            status = {
                "frame_w": frame.shape[1], "frame_h": frame.shape[0], "fps": round(fps, 1),
                "dry_run": args.dry_run, "imgsz": IMGSZ, "conf_threshold": CONFIDENCE,
                "config": {"focal_px": FOCAL_PX, "bottle_r_cm": BOTTLE_R_CM,
                           "max_travel_cm": MAX_TRAVEL_CM, "clearance_cm": PATH_CLEARANCE_CM,
                           "center_tol_px": PAIR_CENTER_TOL_PX,
                           "cam_fwd_offset_cm": CAM_FWD_OFFSET_CM},
            }
            status.update(view)
            status.update(ms)
            hub.publish(frame, status)

            if args.show:
                cv2.imshow("Bob equidistant", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\nCtrl+C - stopping")
        log.write("Ctrl+C")
    except Exception as e:
        log.write(f"CRASH in main loop: {e!r}")
        raise
    finally:
        drive.stop(disable=True, reason="shutdown")    # 1) WHEELS FIRST
        recorder.stop()                                # 2) finish the MP4
        signals.stop()
        for step in (bob.led_off, cam.stop, bob.close_camera):
            try:
                step()
            except Exception as e:
                print("shutdown step failed:", e)
        hub.shutdown()
        server.shutdown()
        if args.show:
            cv2.destroyAllWindows()
        log.write("RUN end")
        log.close()
        print("log saved:", log.path)
        print("bye")


if __name__ == "__main__":
    main()
