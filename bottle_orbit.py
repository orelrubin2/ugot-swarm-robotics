# bottle_orbit.py
# Bob finds the bottle, approaches to 60 cm, then circles around it while
# keeping 60 cm and keeping the bottle centered in the camera.
#
# Reuses camera / YOLO helpers / recorder / web server from bottle_detect.py
# (that file is not changed).
#
# Run (from the project folder):
#   py -3.8 bottle_orbit.py --dry-run            everything runs, wheels never move
#   py -3.8 bottle_orbit.py --stage search       real: search + align, then stop
#   py -3.8 bottle_orbit.py --stage approach     real: ... + approach to 60 cm, then stop
#   py -3.8 bottle_orbit.py                      real: full mission incl. orbit
#   py -3.8 bottle_orbit.py --axis-test          1-second moves: forward, strafe left, turn left
#
# LOG: every run writes logs/orbit_YYYY-MM-DD_HH-MM-SS.log (state changes,
# turn commands with angle, every detection or "none", with time).
#
# AXES: all code uses fwd / left / turn-left. The only translation to ugot's
# mecanum_move_xyz (x = right, y = forward, z = turn left) is in to_ugot().
# Bob never moves until you press START on the dashboard (http://127.0.0.1:5000).
#
# SAFETY layers:
#   - nothing moves before START; STOP button stops the wheels from the web thread
#   - SEARCH/ALIGN only make fixed-angle turns that end with a safety stop
#   - ALIGN lost the bottle -> stand still (3 s total), undo half the last
#     align turn, look 2 s, then SEARCH
#   - APPROACH lost > 1.5 s -> wheels stop, LED red, back to SEARCH
#   - ORBIT (arc through ARC_WAYPOINTS): bottle > 60 px off -> stop strafing, re-center
#     in place to 25 px; bottle not seen -> wheels stop AT ONCE, look 3 s, one
#     recovery turn, look 2 s, then SEARCH
#   - watchdog: if no new speed command for 1 s (program stuck), wheels stop
#   - Ctrl+C / any crash (main thread or other threads) -> wheels stop FIRST

import argparse
import math
import queue
import statistics
import threading
import time
import webbrowser
from collections import deque
from datetime import datetime
from http.server import ThreadingHTTPServer
from pathlib import Path

import cv2
from ultralytics import YOLO

import bottle_detect as bd   # Bob, BobCamera, measure, annotate, Recorder, Hub, make_handler

# ------------------------------------------------------------------ config ---
IMGSZ = 480                    # YOLO input size: 320 was fast but missed the bottle at ~100 cm
CONFIDENCE = 0.25              # lower than bottle_detect.py (0.4) to keep far bottles

# Distance from the box (to the FRONT of the bottle)
FOCAL_PX = 480
BOTTLE_H_CM = 32.0
BOTTLE_W_CM = 9.0
DIST_HISTORY = 5               # median of the last 5 detections
DIST_MIN_SAMPLES = 3           # need at least 3 before driving on distance

TARGET_CM = 60.0
DIST_TOL_CM = 5.0              # 60 +- 5 cm counts as "arrived"
CENTER_TOL_PX = 40             # same as bottle_detect.py
BOTTLE_RADIUS_CM = 4.5         # half of 9 cm -> orbit radius = 60 + 4.5

# STOP-AND-LOOK for SEARCH + ALIGN (FPS is low and frames blur while turning):
# turn a fixed angle with mecanum_turn_speed_times (bottle_hunter.py's call),
# stop, wait SETTLE_SEC, then judge only frames that arrived after that.
SETTLE_SEC = 0.5               # wait after every turn before looking
TURN_EXTRA_SEC = 0.4           # margin on top of angle/speed before the safety stop
SEARCH_STEP_DEG = 30
SEARCH_TURN_SPEED = 45         # deg/s (bottle_hunter TURN_SPEED)
SEARCH_LOOK_FRAMES = 4         # look at 4 fresh frames per position; >= 1 hit -> ALIGN
ALIGN_TURN_SPEED = 30          # bottle_hunter CENTER_TURN_SPEED ("too slow + too small = no move")
# YOLO sees the bottle in only ~81% of still frames (conf 0.29-0.45 at 80 cm),
# and the turns are accurate -> look longer, and trust the turn.
ALIGN_LOOK_FRAMES = 6          # up to 6 fresh frames (~2 s) after each align turn
ALIGN_ENOUGH_HITS = 3          # ...but decide early once 3 hits are in (median offset)
                               # (after 6 frames, even 1 hit is enough)
ALIGN_MAX_TRIES = 6
MIN_TURN_STEP_DEG = 2          # bottle_hunter MIN_CENTER_STEP_DEG
NO_PROGRESS_PX = 5             # offset changed less than this -> last turn did nothing...
NO_PROGRESS_BOOST_DEG = 1      # ...so add 1 deg to the next turn (bottle_hunter's trick)
ALIGN_LOST_TOTAL_SEC = 3.0     # nothing seen after an align turn: stand still, look up to 3 s
LOST_BACK_FRACTION = 0.5       # then undo HALF of the last align turn (opposite direction)
LOST_BACK_LOOK_SEC = 2.0       # look 2 s there; still nothing -> SEARCH

LOGS_DIR = bd.SCRIPT_DIR / "logs"

# Continuous controller (APPROACH + ORBIT) gains + limits (cm/s and deg/s)
K_TURN = 0.12                  # deg/s per px of x-offset
MAX_TURN = 30                  # |z| limit
K_DIST = 0.5                   # cm/s per cm of distance error (APPROACH)
MAX_APPROACH = 10              # |x| limit in APPROACH (asked: max 10 cm/s)
MIN_LINEAR = 4                 # smallest forward/back speed that still moves Bob
K_DIST_ORBIT = 0.3             # gentler distance correction while orbiting
MAX_ORBIT_X = 5                # |x| limit in ORBIT

# ORBIT = an ARC through waypoints, so the black background stays behind the bottle.
# Degrees around the bottle, relative to where ORBIT starts:
#   + = to the RIGHT of the start point (as Bob sees it), - = to the LEFT.
ARC_WAYPOINTS = [+45, -45, 0]
ARC_MAX_SPEED = 5.0            # cm/s strafe (max)
ARC_MIN_SPEED = 3.0            # cm/s - slower may not move the wheels at all
ARC_SLOW_ZONE_DEG = 12.0       # closer than this to the waypoint -> slow down linearly
WAYPOINT_TOL_DEG = 3.0         # waypoint reached when within +-3 deg
ORBIT_RADIUS_CM = TARGET_CM + BOTTLE_RADIUS_CM   # 64.5 cm: turn feedforward = v / R

# KEEP-CENTERED RULE during the arc
RECENTER_START_PX = 60         # |offset| > 60 px -> stop strafing, turn in place only
RECENTER_DONE_PX = 25          # ...until |offset| <= 25 px, then strafe again
RECENTER_MAX_TRIES = 6         # timed re-centering turns before giving up (STOPPED)
ORBIT_LOST_LOOK_SEC = 3.0      # not seen: wheels stop at once, stand still, look 3 s
ORBIT_BACK_LOOK_SEC = 2.0      # after the one recovery turn: look 2 s, then SEARCH
ORBIT_RECOVER_MIN_DEG = 5      # smallest recovery turn toward where it was last seen

# Heading (= angle travelled around the bottle while facing it).
# Source: Bob's IMU yaw via ugot read_gyro_data(). Checked on every timed turn
# (those are accurate on Bob): if the IMU disagrees with the commanded angle
# twice in a row, it is not trusted and the integrated turn commands are used.
# (NOT checked against the command total: bumps/slip would wrongly fail the IMU.)
IMU_YAW_SIGN = +1              # +1 = yaw grows when Bob turns LEFT (check with --axis-test)
IMU_PERIOD_SEC = 0.2           # read the IMU 5x per second
IMU_TURN_CHECK_MIN_DEG = 5     # only check on timed turns of at least 5 deg
IMU_RATIO_MIN, IMU_RATIO_MAX = 0.5, 2.0   # IMU/commanded outside this = a bad check
IMU_BAD_CHECKS = 2             # this many bad checks in a row -> IMU not trusted

APPROACH_OK_FRAMES = 3         # at 60+-5 and centered this many frames -> APPROACH done

LOST_COAST_SEC = 0.5           # bottle missing: keep last command this long, then 0
LOST_TIMEOUT_SEC = 1.5         # APPROACH/ORBIT: missing longer -> STOP + back to SEARCH
CMD_TIMEOUT_SEC = 1.0          # watchdog: no new command this long -> STOP

TURN_LEFT = 2                  # same constants as bottle_hunter.py
TURN_RIGHT = 3
UNIT_DEGREES = 2

ORBIT_DASHBOARD = bd.SCRIPT_DIR / "orbit_dashboard.html"

# Mission state -> LED (r, g, b) and the word Bob says on entering it
STATE_LED = {
    "SEARCH": (255, 0, 0), "ALIGN": (255, 255, 0), "APPROACH": (255, 255, 0),
    "ORBIT": (0, 255, 0), "STOPPED": (255, 0, 0), "DONE": (0, 255, 0),
}
STATE_SOUND = {
    "SEARCH": "Searching", "ALIGN": "Aligning", "APPROACH": "Approaching",
    "ORBIT": "Orbiting", "STOPPED": "Stopped", "DONE": "Done",
}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def with_min(v, minimum):
    """Push a non-zero speed up to the smallest value that really moves the wheels."""
    if v == 0:
        return 0
    return math.copysign(max(abs(v), minimum), v)


# ------------------------------------------------------------------- wheels ---
def to_ugot(fwd, left, turn):
    """THE ONE PLACE where my axes become ugot's mecanum_move_xyz axes.

    Measured on Bob with --axis-test (ugot docs don't say):
        ugot x = +right (sideways), ugot y = +forward, ugot z = +turn left
    My code uses: fwd (+forward), left (+strafe left), turn (+turn left).
    """
    return -left, fwd, turn            # (ugot_x, ugot_y, ugot_z)


def send_xyz(got, fwd, left, turn, dry_run=False):
    """Send a continuous move in MY axes (cm/s, cm/s, deg/s). Returns the ugot args."""
    ux, uy, uz = to_ugot(int(round(fwd)), int(round(left)), int(round(turn)))  # ugot fields are ints
    if dry_run:
        print(f"[dry-run] fwd={fwd:+.0f} left={left:+.0f} turn={turn:+.0f}  ->  "
              f"mecanum_move_xyz(x={ux}, y={uy}, z={uz})")
    else:
        got.mecanum_move_xyz(ux, uy, uz)
    return ux, uy, uz


class RunLog:
    """Line-buffered text log in logs/ - every line is on disk right away,
    so the log survives Ctrl+C and crashes."""

    def __init__(self, folder):
        Path(folder).mkdir(exist_ok=True)
        self.path = Path(folder) / datetime.now().strftime("orbit_%Y-%m-%d_%H-%M-%S.log")
        self.f = open(self.path, "a", encoding="utf-8", buffering=1)
        self.t0 = time.time()
        self.lock = threading.Lock()

    def write(self, msg):
        stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        line = f"{stamp} +{time.time() - self.t0:8.3f}s  {msg}"
        with self.lock:
            if not self.f.closed:
                self.f.write(line + "\n")

    def close(self):
        with self.lock:
            self.f.close()


class Drive:
    """The ONLY place that talks to Bob's wheels.

    - enabled=False (before START / after STOP) -> every move is ignored
    - dry_run -> print what would be sent instead of moving (stop is still real)
    - watchdog thread stops the wheels if continuous commands stop arriving
    - timed turns always end with a safety mecanum_stop + SETTLE_SEC wait
    """

    def __init__(self, bob, dry_run, log):
        self.got = bob.got
        self.dry_run = dry_run
        self.log = log
        self.lock = threading.Lock()
        self.enabled = False
        self.moving = False            # a non-zero continuous command is active
        self.last_cmd_time = 0.0
        self.last_sent = (0, 0, 0)     # (x, y, z) logical values, for the dashboard
        self.turning = False           # a timed turn (incl. settle wait) is in progress
        self.turn_gen = 0              # bumped by stop() to cancel a running turn
        # Heading from the commands we sent (deg, + = left): integral of turn rate
        self.cmd_heading = 0.0
        self.cur_rate = 0.0            # turn rate currently commanded (deg/s)
        self.rate_since = time.time()
        self.turn_listener = None      # Heading: checks the IMU on every timed turn
        threading.Thread(target=self._watchdog, daemon=True).start()

    def enable(self):
        with self.lock:
            self.enabled = True

    def _integrate(self, t):
        """(lock held) add the commanded turn since the last change, up to time t."""
        if t > self.rate_since:
            self.cmd_heading += self.cur_rate * (t - self.rate_since)
            self.rate_since = t

    def heading_cmd(self):
        """Heading estimate from commands only (deg, + = left)."""
        with self.lock:
            return self.cmd_heading + self.cur_rate * max(0.0, time.time() - self.rate_since)

    def move(self, x, y, z):
        """Continuous move in MY axes: x forward, y left (cm/s), z turn-left (deg/s).
        The swap to ugot's axes happens only in to_ugot()."""
        x, y, z = int(round(x)), int(round(y)), int(round(z))
        with self.lock:
            if not self.enabled:
                return
            now = time.time()
            self._integrate(now)
            self.cur_rate = float(z)
            self.last_cmd_time = now
            self.last_sent = (x, y, z)
            self.moving = (x, y, z) != (0, 0, 0)
            send_xyz(self.got, x, y, z, self.dry_run)

    def turn_degrees(self, degrees, speed, reason=""):
        """Turn in place by a fixed angle, then stop and settle.
        degrees > 0 = LEFT, < 0 = RIGHT (same sign as 'turn' everywhere else).

        Runs in a helper thread so the video never freezes:
          mecanum_turn_speed_times -> wait angle/speed (+margin) -> mecanum_stop
          -> wait SETTLE_SEC -> self.turning = False
        Returns False if nothing was sent (disabled or 0 degrees)."""
        amount = int(round(abs(degrees)))              # ugot wants whole degrees
        if amount == 0:
            return False
        direction = TURN_LEFT if degrees > 0 else TURN_RIGHT
        side = "LEFT" if degrees > 0 else "RIGHT"
        with self.lock:
            if not self.enabled:
                return False
            self.moving = False
            self.turning = True
            self.turn_gen += 1
            gen = self.turn_gen
            self.last_sent = (0, 0, speed if degrees > 0 else -speed)
            t_start = time.time()
            self._integrate(t_start)
            self.cur_rate = float(speed if degrees > 0 else -speed)   # heading integral
        self.log.write(f"TURN {side} {amount} deg @ {speed} deg/s  ({reason})")
        print(f"[turn] {side} {amount} deg @ {speed} deg/s  ({reason})"
              + ("  [dry-run, not sent]" if self.dry_run else ""))
        signed = amount if degrees > 0 else -amount
        listener = self.turn_listener
        if listener:
            listener.turn_started()

        def worker():
            t0 = time.time()
            if not self.dry_run:
                try:
                    self.got.mecanum_turn_speed_times(direction, speed, amount, UNIT_DEGREES)
                except Exception as e:
                    print("turn failed:", e)
                    self.log.write(f"TURN FAILED: {e}")
            # Wait as long as the turn needs (in case the call returned at once)
            t_exact = t_start + amount / float(speed)  # the turn itself is over here
            end = t_exact + TURN_EXTRA_SEC
            rate_cleared = False
            while time.time() < end:
                if gen != self.turn_gen:               # STOP pressed -> turn cancelled
                    return
                if not rate_cleared and time.time() >= t_exact:
                    with self.lock:                    # heading integral: exactly 'amount'
                        if gen == self.turn_gen:
                            self._integrate(t_exact)
                            self.cur_rate = 0.0
                    rate_cleared = True
                time.sleep(0.02)
            with self.lock:
                if gen == self.turn_gen and not rate_cleared:
                    self._integrate(t_exact)
                    self.cur_rate = 0.0
            if not self.dry_run:
                try:
                    self.got.mecanum_stop()            # safety stop at the end of every turn
                except Exception as e:
                    print("mecanum_stop failed:", e)
            time.sleep(SETTLE_SEC)                     # let the image settle (no blur)
            with self.lock:
                if gen != self.turn_gen:
                    return
                self.turning = False
                self.last_sent = (0, 0, 0)
            self.log.write(f"TURN done + settled ({time.time() - t0:.2f} s)")
            if listener:
                listener.turn_finished(signed)

        threading.Thread(target=worker, daemon=True).start()
        return True

    def stop(self, quiet=False, disable=False, reason=""):
        """Stop the wheels NOW. Always real, also in dry-run (stopping is safe)."""
        with self.lock:
            if disable:
                self.enabled = False
            self._integrate(time.time())               # count the turning done so far
            self.cur_rate = 0.0
            self.moving = False
            self.turning = False
            self.turn_gen += 1                         # cancels a running turn
            self.last_sent = (0, 0, 0)
        self.log.write("WHEELS STOP" + (f" ({reason})" if reason else ""))
        if not quiet:
            print("[wheels] STOP" + (" (dry-run)" if self.dry_run else ""))
        for _ in range(2):             # send twice - a lost packet must not keep Bob driving
            try:
                self.got.mecanum_stop()
            except Exception as e:
                print("mecanum_stop failed:", e)

    def _watchdog(self):
        while True:
            time.sleep(0.1)
            with self.lock:
                stale = (self.moving and time.time() - self.last_cmd_time > CMD_TIMEOUT_SEC)
            if stale:
                print("[watchdog] no speed command for "
                      f"{CMD_TIMEOUT_SEC} s -> stopping wheels")
                self.stop(quiet=True, reason="watchdog")


class ImuReader:
    """Reads Bob's IMU yaw with ugot read_gyro_data() -> [pitch, roll, yaw, ...]
    a few times per second and unwraps it (no jump at +-180).
    ugot returns all zeros when it gets no data -> counted as a failure."""

    def __init__(self, bob, log):
        self.bob = bob
        self.log = log
        self.lock = threading.Lock()
        self.yaw = None                # unwrapped yaw (deg), raw sign
        self._prev = None
        self.ok = False
        self.fails = 0
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            try:
                with self.bob.lock:                    # shares the connection with the camera
                    data = self.bob.got.read_gyro_data()
                if not data or not any(data):
                    raise ValueError("read_gyro_data returned no data (all zeros)")
                yaw = float(data[2])
                with self.lock:
                    if self._prev is None:
                        self.yaw = yaw
                        self.log.write(f"IMU ok, first yaw={yaw:.2f}")
                    else:
                        self.yaw += (yaw - self._prev + 180.0) % 360.0 - 180.0
                    self._prev = yaw
                    self.ok = True
                    self.fails = 0
            except Exception as e:
                with self.lock:
                    self.fails += 1
                    if self.fails == 5:
                        self.ok = False
                        self.log.write(f"IMU not available ({e}) - using turn commands")
                        print("IMU not available - heading from turn commands")
            time.sleep(IMU_PERIOD_SEC)

    def get(self):
        with self.lock:
            return self.yaw if self.ok else None


class Heading:
    """Heading change since mark() (deg, + = LEFT). Prefers the IMU, falls back
    to the integrated turn commands if the IMU is missing or disagrees."""

    def __init__(self, drive, imu, log, use_imu):
        self.drive = drive
        self.imu = imu
        self.log = log
        self.use_imu = use_imu         # False in --dry-run (Bob doesn't really turn)
        self.imu_bad = False
        self.bad_checks = 0
        self.good_checks = 0           # IMU is only USED after >= 1 good turn check
        self.cmd0 = 0.0
        self.imu0 = None
        self.last_delta = 0.0
        self._src = None
        self._turn_yaw0 = None
        drive.turn_listener = self

    def imu_usable(self):
        return self.use_imu and not self.imu_bad and self.good_checks >= 1

    # --- IMU check on every timed turn (called from Drive's turn thread) ---
    def turn_started(self):
        self._turn_yaw0 = self.imu.get() if self.use_imu else None

    def turn_finished(self, commanded):
        y0, y1 = self._turn_yaw0, (self.imu.get() if self.use_imu else None)
        if y0 is None or y1 is None or abs(commanded) < IMU_TURN_CHECK_MIN_DEG or self.imu_bad:
            return
        d = IMU_YAW_SIGN * (y1 - y0)
        ratio = d / commanded
        good = IMU_RATIO_MIN <= ratio <= IMU_RATIO_MAX
        self.bad_checks = 0 if good else self.bad_checks + 1
        if good:
            self.good_checks += 1
        self.log.write(f"IMU check: commanded {commanded:+d} deg, IMU {d:+.1f} deg -> "
                       f"{'ok' if good else 'BAD'}")
        if self.bad_checks >= IMU_BAD_CHECKS or (self.good_checks == 0 and ratio < 0):
            # (a wrong SIGN before any good check is never trusted - fail at once)
            self.imu_bad = True
            if ratio < 0:
                why = "IMU turned the OPPOSITE way - IMU_YAW_SIGN looks wrong"
            else:
                why = f"IMU failed {self.bad_checks} turn checks in a row (ratio {ratio:.2f})"
            msg = f"{why}; heading from turn commands from now on"
            self.log.write("HEADING " + msg)
            print("WARNING:", msg)

    def mark(self):
        """Zero the heading change here (ORBIT start)."""
        self.last_delta = 0.0
        self._src = None
        # raw yaw at ORBIT start - kept even before the IMU is verified, so a
        # later switch to the IMU uses its TRUE change since here (not the
        # command estimate, which may already include slip)
        self.imu_at_mark = self.imu.get() if self.use_imu else None
        d, src = self.delta()
        self.log.write(f"HEADING mark: source={src} (IMU good checks={self.good_checks}, "
                       f"bad={self.imu_bad})")

    def delta(self):
        """Heading change since mark(), (degrees, 'imu' | 'cmd').
        When the source switches (IMU gets verified / fails), the new source is
        re-based to continue from the last value - no jump in the arc angle."""
        cmd_now = self.drive.heading_cmd()
        y = self.imu.get() if self.imu_usable() else None
        if y is not None:
            if self._src != "imu":
                if self.imu_at_mark is not None:          # true change since ORBIT start
                    self.imu0 = self.imu_at_mark
                else:                                     # no IMU at start: continue smoothly
                    self.imu0 = y - IMU_YAW_SIGN * self.last_delta
                if self._src:
                    new = IMU_YAW_SIGN * (y - self.imu0)
                    self.log.write(f"HEADING source -> IMU yaw (verified on a turn): "
                                   f"{self.last_delta:+.1f} -> {new:+.1f} deg")
            d, src = IMU_YAW_SIGN * (y - self.imu0), "imu"
        else:
            if self._src != "cmd":
                self.cmd0 = cmd_now - self.last_delta
                if self._src:
                    self.log.write("HEADING source -> turn commands")
            d, src = cmd_now - self.cmd0, "cmd"
        self._src = src
        self.last_delta = d
        return d, src


class MissionSignals:
    """LED + voice for mission states, in its own thread, only on state change."""

    def __init__(self, bob):
        self.bob = bob
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def set_state(self, state):
        self.q.put(state)

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

    def stop(self):
        self.q.put(None)
        self.thread.join(timeout=2.0)


# ------------------------------------------------------------- perception ---
def find_bottle(model, frame):
    """Like bottle_detect.find_bottle, but with its own IMGSZ and CONFIDENCE."""
    result = model(frame, classes=[bd.BOTTLE_CLASS_ID], conf=CONFIDENCE,
                   imgsz=IMGSZ, verbose=False)[0]
    if len(result.boxes) == 0:
        return None
    best = int(result.boxes.conf.argmax())
    x1, y1, x2, y2 = result.boxes.xyxy[best].tolist()
    return int(x1), int(y1), int(x2), int(y2), float(result.boxes.conf[best])


def raw_distance_cm(m):
    """Pinhole distance: height if the whole bottle is visible, else width."""
    if m["top_cut"]:
        return (FOCAL_PX * BOTTLE_W_CM / m["w"], "width") if m["w"] > 0 else (None, None)
    return (FOCAL_PX * BOTTLE_H_CM / m["h"], "height") if m["h"] > 0 else (None, None)


# ---------------------------------------------------------------- mission ---
class Mission:
    """State machine: IDLE -> SEARCH -> ALIGN -> APPROACH -> ORBIT
    (STOPPED after the STOP button, DONE when --stage is finished)."""

    def __init__(self, drive, signals, stage, log, heading):
        self.drive = drive
        self.signals = signals
        self.stage = stage             # "search", "approach" or "orbit"
        self.log = log
        self.heading = heading
        # ORBIT (arc) state
        self.arc_phase = "arc"         # "arc" | "recenter" | "lost" | "back"
        self.arc_angle = 0.0           # deg travelled around the bottle (+ = right of start)
        self.arc_beta0 = 0.0           # bottle angle in the camera at ORBIT start
        self.heading_src = "-"
        self.wp_index = 0
        self.recenter_tries = 0
        self.last_orbit_turn = None    # signed deg of the last re-centering turn
        self.orbit_lost_after_turn = False
        self.lock = threading.Lock()
        self.state = "IDLE"
        self.message = "Press START to begin"
        self.phase = ""                # what SEARCH/ALIGN is doing right now (dashboard + log)
        self.dist_hist = deque(maxlen=DIST_HISTORY)
        self.distance = None
        self.dist_method = None
        self.last_seen = 0.0
        self.last_offset = 0           # where the bottle was last seen (+ = right)
        self.last_cmd = (0, 0, 0)
        self.ok_frames = 0
        self.look = None               # current stop-and-look window (see _begin_look)
        self.search_turns = 0
        self.align_tries = 0
        self.align_prev_off = None     # for the "turn did nothing -> +1 deg" boost
        self.align_boost = 0
        self.last_align_turn = None    # signed degrees of the last ALIGN turn (+ = left)

    # --- called from the web server thread ---
    def start(self):
        with self.lock:
            if self.state not in ("IDLE", "STOPPED", "DONE"):
                return
            self.drive.enable()
            self.dist_hist.clear()
            self.log.write("START pressed")
            self._enter_search("Mission started")

    def stop_button(self):
        self.drive.stop(disable=True, reason="STOP button")   # wheels first
        with self.lock:
            self.look = None
            self.phase = ""
            self._set("STOPPED", "STOP button pressed - press START to restart")

    # --- helpers (caller holds self.lock) ---
    def _set(self, state, msg):
        self.message = msg
        if state != self.state:
            print(f">>> {self.state} -> {state}: {msg}")
            self.log.write(f"STATE {self.state} -> {state}: {msg}")
            self.state = state
            self.ok_frames = 0
            self.signals.set_state(state)

    def _send(self, x, y, z):
        self.last_cmd = (x, y, z)
        self.drive.move(x, y, z)

    def _finish_stage(self, msg):
        self.drive.stop(disable=True, reason="stage finished")
        self.look = None
        self.phase = ""
        self._set("DONE", msg)

    def _begin_look(self, purpose, frames=0):
        """Open a look window. It starts only once Bob has stopped and settled
        (drive.turning False); only frames that arrive AFTER that are counted."""
        self.look = {"purpose": purpose, "need": frames, "marker": None,
                     "t0": None, "frames": 0, "hits": []}

    def _enter_search(self, msg):
        self.search_turns = 0
        self._begin_look("search", SEARCH_LOOK_FRAMES)   # look here first, then turn
        self._set("SEARCH", msg)

    def _enter_align(self, hits, msg):
        self.align_tries = 0
        self.align_prev_off = None
        self.align_boost = 0
        self.last_align_turn = None
        self._set("ALIGN", msg)
        self._align_decide(hits)

    # --- called once per processed frame from the main loop ---
    def step(self, m, frame_id, now):
        with self.lock:
            seen = m["state"] != "NO_BOTTLE"
            if seen:
                self.last_seen = now
                self.last_offset = m["offset"]
                d, method = raw_distance_cm(m)
                if d is not None:
                    self.dist_hist.append(d)
                    self.dist_method = method
            self.distance = (statistics.median(self.dist_hist)
                             if len(self.dist_hist) >= DIST_MIN_SAMPLES else None)

            st = self.state
            if st == "ORBIT" and seen and not self.drive.turning:
                self._update_arc_angle(m["offset"])
            if st in ("SEARCH", "ALIGN") or (st == "ORBIT" and self.arc_phase != "arc"):
                self._stop_and_look(m, seen, frame_id, now)
            elif st == "APPROACH":
                self._track(m, seen, now)
            elif st == "ORBIT":
                self._orbit_arc(m, seen)
            self._log_detection(m, seen, frame_id, d if seen else None)

    def _log_detection(self, m, seen, frame_id, raw_d):
        where = f"{self.state:<8} {self.phase:<22}"
        if self.state == "ORBIT":
            where += (f" arc={self.arc_angle:+.1f}deg wp={self.wp_index + 1}/{len(ARC_WAYPOINTS)}"
                      f" src={self.heading_src}")
        if seen:
            med = f"{self.distance:.0f}" if self.distance is not None else "-"
            raw = f"{raw_d:.0f}" if raw_d is not None else "-"
            self.log.write(f"DET  frame={frame_id:<6} {where} cx={m['cx']} off={m['offset']:+d} "
                           f"w={m['w']} h={m['h']} conf={m['conf']:.2f} "
                           f"topcut={int(m['top_cut'])} dist={raw}cm med={med}cm "
                           f"cmd={self.drive.last_sent}")
        else:
            self.log.write(f"DET  frame={frame_id:<6} {where} none  cmd={self.drive.last_sent}")

    # ---------------------------------------------------- stop-and-look ---
    def _stop_and_look(self, m, seen, frame_id, now):
        lk = self.look
        if lk is None:
            return
        if self.drive.turning:                         # moving or settling: frames are blurry
            self.phase = "turning / settling"
            return
        if lk["marker"] is None:                       # Bob just became still
            lk["marker"] = frame_id                    # only frames NEWER than this count
            lk["t0"] = now
            return
        if frame_id <= lk["marker"]:
            return
        lk["frames"] += 1
        if seen:
            lk["hits"].append(m)

        purpose = lk["purpose"]
        hits = lk["hits"]
        waited = now - lk["t0"]                        # time since Bob became still

        if purpose == "lost_wait":
            # Trust the turn: the bottle should be near the center. Stand still and
            # keep looking until ALIGN_LOST_TOTAL_SEC since the look started.
            self.phase = f"not seen: look {waited:.1f}/{ALIGN_LOST_TOTAL_SEC:.0f}s"
            if seen:
                self.log.write(f"ALIGN: bottle seen again after {waited:.1f}s standing still")
                self._align_decide(hits)
            elif waited >= ALIGN_LOST_TOTAL_SEC:
                self._undo_half_turn()
            return

        if purpose == "lost_back":                     # after undoing half the turn: look 2 s
            self.phase = f"undo half: look {waited:.1f}/{LOST_BACK_LOOK_SEC:.0f}s"
            if seen:
                self.log.write("LOST-RECOVERY: found after undoing half the last turn")
                self.align_prev_off = None             # big jump is not "no progress"
                self._align_decide(hits)
            elif waited >= LOST_BACK_LOOK_SEC:
                self._enter_search("Bottle lost in ALIGN (waited 3 s + undid half turn) - searching")
            return

        # ---- ORBIT pauses (wheels are stopped in all of them) ----
        if purpose == "orbit_lost":                    # stand still, look up to 3 s
            self.phase = f"not seen: look {waited:.1f}/{ORBIT_LOST_LOOK_SEC:.0f}s"
            if seen:
                self.log.write(f"ORBIT: bottle seen again after {waited:.1f}s standing still")
                self._orbit_look_result(hits)
            elif waited >= ORBIT_LOST_LOOK_SEC:
                self._orbit_recovery_turn()
            return
        if purpose == "orbit_back":                    # after the one recovery turn: 2 s
            self.phase = f"recovery: look {waited:.1f}/{ORBIT_BACK_LOOK_SEC:.0f}s"
            if seen:
                self.log.write("ORBIT: found again after the recovery turn")
                self._orbit_look_result(hits)
            elif waited >= ORBIT_BACK_LOOK_SEC:
                self._enter_search("Bottle lost in ORBIT (waited 3 s + recovery turn) - searching")
            return
        if purpose == "orbit_recenter":                # after a re-centering turn
            self.phase = f"PAUSED re-center look {lk['frames']}/{lk['need']} hits={len(hits)}"
            if len(hits) >= ALIGN_ENOUGH_HITS:
                self._orbit_look_result(hits)
            elif lk["frames"] >= lk["need"]:
                if hits:
                    self._orbit_look_result(hits)
                else:                                  # 0 hits: keep standing, keep looking
                    self.log.write(f"ORBIT: 0 hits after re-center turn -> look up to "
                                   f"{ORBIT_LOST_LOOK_SEC:.0f}s total (no turn)")
                    self.arc_phase = "lost"
                    lk["purpose"] = "orbit_lost"       # same window: same t0
            return

        self.phase = f"look {lk['frames']}/{lk['need']} hits={len(hits)}"
        if purpose == "search":
            if seen:                                   # >= 1 hit is enough: go now
                self.log.write(f"SEARCH: bottle seen in fresh frame {lk['frames']}/{lk['need']}")
                self._enter_align(hits, "Bottle found - centering it")
            elif lk["frames"] >= lk["need"]:           # 4 frames, nothing: next position
                self.search_turns += 1
                self.message = f"No bottle - turning {SEARCH_STEP_DEG} deg (turn {self.search_turns})"
                self._begin_look("search", SEARCH_LOOK_FRAMES)
                self.drive.turn_degrees(-SEARCH_STEP_DEG, SEARCH_TURN_SPEED,
                                        f"search turn {self.search_turns}")
            return

        if purpose == "align":
            if len(hits) >= ALIGN_ENOUGH_HITS:         # good median already: decide early
                self._align_decide(hits)
            elif lk["frames"] >= lk["need"]:
                if hits:                               # 1-2 hits in 6 frames is enough
                    self._align_decide(hits)
                else:                                  # 0 hits: keep standing, keep looking
                    self.log.write(f"ALIGN: 0 hits in {lk['need']} frames -> keep looking "
                                   f"up to {ALIGN_LOST_TOTAL_SEC:.0f}s total (no turn)")
                    self.message = "Not seen after the turn - standing still, still looking"
                    lk["purpose"] = "lost_wait"        # same window: same t0, same marker
            return

    def _undo_half_turn(self):
        """3 s without the bottle after an align turn: undo HALF of that turn."""
        if not self.last_align_turn:
            self._enter_search("Bottle lost in ALIGN (no turn to undo) - searching")
            return
        back = -self.last_align_turn * LOST_BACK_FRACTION
        self.message = f"Still not seen - undoing half the last turn ({abs(back):.0f} deg)"
        self.log.write(f"ALIGN: not seen for {ALIGN_LOST_TOTAL_SEC:.0f}s -> undo half of last "
                       f"turn ({self.last_align_turn:+.0f} deg -> {back:+.0f} deg)")
        self.last_align_turn = None                    # only undo once
        self._begin_look("lost_back")
        if not self.drive.turn_degrees(back, ALIGN_TURN_SPEED, "undo half of last align turn"):
            self.log.write("ALIGN: undo turn too small/not sent - just looking")

    def _align_decide(self, hits):
        """Look result in ALIGN: centered -> next stage, else turn by the exact angle."""
        if not hits:                                   # (safety net - normally not reached)
            self.message = "Bottle not seen - standing still, looking"
            self._begin_look("lost_wait")
            return

        off = statistics.median(h["offset"] for h in hits)    # px, + = bottle right
        if abs(off) <= CENTER_TOL_PX:
            self.log.write(f"ALIGN: centered (median off={off:+.0f}px, {len(hits)} hits)")
            self.look = None
            self.phase = ""
            if self.stage == "search":
                self._finish_stage(f"Stage 'search' done: bottle centered ({off:+.0f} px)")
            else:
                self.last_seen = time.time()
                self._set("APPROACH", f"Centered - driving to {TARGET_CM:.0f} cm")
            return

        if self.align_tries >= ALIGN_MAX_TRIES:
            self.drive.stop(disable=True, reason="ALIGN failed")
            self.look = None
            self.phase = ""
            self._set("STOPPED", f"ALIGN failed after {ALIGN_MAX_TRIES} tries "
                                 f"(still {off:+.0f} px) - see log, press START")
            return

        # bottle_hunter's trick: if the last turn didn't change the offset, add 1 deg
        if (self.align_prev_off is not None and abs(off - self.align_prev_off) < NO_PROGRESS_PX
                and (off > 0) == (self.align_prev_off > 0)):
            self.align_boost += NO_PROGRESS_BOOST_DEG
        elif self.align_prev_off is not None:
            self.align_boost = max(0, self.align_boost - NO_PROGRESS_BOOST_DEG)
        self.align_prev_off = off

        self.align_tries += 1
        angle = math.degrees(math.atan(off / FOCAL_PX))       # + = bottle right of center
        size = max(abs(angle), MIN_TURN_STEP_DEG) + self.align_boost
        turn = -math.copysign(size, angle)                    # bottle right -> turn RIGHT
        self.message = (f"Align try {self.align_tries}/{ALIGN_MAX_TRIES}: "
                        f"{off:+.0f} px -> turn {abs(turn):.0f} deg {'right' if turn < 0 else 'left'}")
        self.last_align_turn = turn
        self._begin_look("align", ALIGN_LOOK_FRAMES)
        self.drive.turn_degrees(turn, ALIGN_TURN_SPEED,
                                f"align try {self.align_tries}: off={off:+.0f}px "
                                f"angle={angle:+.1f} boost={self.align_boost}")

    # ------------------------------------------------ continuous tracking ---
    def _track(self, m, seen, now):
        st = self.state
        if not seen:
            lost = now - self.last_seen
            if lost > LOST_TIMEOUT_SEC:
                self.drive.stop(reason=f"{st}: bottle lost > {LOST_TIMEOUT_SEC}s")  # wheels first
                self.dist_hist.clear()
                self.last_cmd = (0, 0, 0)
                self._enter_search(f"Bottle lost > {LOST_TIMEOUT_SEC} s - wheels stopped")
            elif lost > LOST_COAST_SEC:
                self._send(0, 0, 0)                # hold still while it may come back
            else:
                self._send(*self.last_cmd)         # short blink: keep going
            return

        offset = m["offset"]
        d = self.distance
        centered = abs(offset) <= CENTER_TOL_PX
        z_corr = clamp(-K_TURN * offset, -MAX_TURN, MAX_TURN)   # bottle right -> turn right

        if st == "APPROACH":
            if d is None:                          # not enough distance samples yet
                self._send(0, 0, z_corr)
                return
            err = d - TARGET_CM                    # + = too far -> forward
            arrived = abs(err) <= DIST_TOL_CM
            if arrived and centered:
                self.ok_frames += 1
                self._send(0, 0, 0)
                if self.ok_frames >= APPROACH_OK_FRAMES:
                    if self.stage == "approach":
                        self._finish_stage(f"Stage 'approach' done at {d:.0f} cm")
                    else:
                        self._enter_orbit(m, f"At {d:.0f} cm - starting arc {ARC_WAYPOINTS}")
            else:
                self.ok_frames = 0
                x = 0 if arrived else with_min(clamp(K_DIST * err, -MAX_APPROACH, MAX_APPROACH),
                                               MIN_LINEAR)
                self.message = f"Distance {d:.0f} cm, target {TARGET_CM:.0f} +- {DIST_TOL_CM:.0f}"
                self._send(x, 0, z_corr)

    # ------------------------------------------------------------ ORBIT arc ---
    def _enter_orbit(self, m, msg):
        self.heading.mark()
        self.arc_beta0 = math.degrees(math.atan(m["offset"] / FOCAL_PX))
        self.arc_angle = 0.0
        self.wp_index = 0
        self.arc_phase = "arc"
        self.recenter_tries = 0
        self.last_orbit_turn = None
        self.look = None
        self._set("ORBIT", msg)
        self.log.write(f"ORBIT start: waypoints={ARC_WAYPOINTS} beta0={self.arc_beta0:+.1f}deg")

    def _update_arc_angle(self, offset):
        """Angle travelled around the bottle.
        Facing the bottle, it equals the heading change. The bottle's angle in the
        camera (beta, + = right) corrects for not facing it exactly:
            arc = heading_change - beta + beta0"""
        dh, self.heading_src = self.heading.delta()
        beta = math.degrees(math.atan(offset / FOCAL_PX))
        self.arc_angle = dh - beta + self.arc_beta0

    def _orbit_arc(self, m, seen):
        """Continuous arc toward the current waypoint, with the KEEP-CENTERED rule."""
        if not seen:
            self.drive.stop(reason="ORBIT: bottle not seen")       # all wheels, at once
            self.last_cmd = (0, 0, 0)
            self.orbit_lost_after_turn = False
            self.arc_phase = "lost"
            self.message = f"Bottle not seen - stopped, looking {ORBIT_LOST_LOOK_SEC:.0f} s"
            self._begin_look("orbit_lost")
            return

        off = m["offset"]
        if abs(off) > RECENTER_START_PX:                            # rule 4
            self.drive.stop(reason=f"ORBIT: offset {off:+d}px > {RECENTER_START_PX} - re-center")
            self.last_cmd = (0, 0, 0)
            self.log.write(f"ORBIT: PAUSED strafing, re-centering (off={off:+d}px)")
            self.recenter_tries = 0
            self._orbit_recenter_turn(off)
            return

        target = ARC_WAYPOINTS[self.wp_index]
        err = target - self.arc_angle                               # + = go right
        if abs(err) <= WAYPOINT_TOL_DEG:
            self.drive.stop(reason=f"waypoint {target:+d} reached")
            self.last_cmd = (0, 0, 0)
            self.log.write(f"ORBIT: waypoint {self.wp_index + 1}/{len(ARC_WAYPOINTS)} "
                           f"({target:+d} deg) reached at {self.arc_angle:+.1f} deg")
            self.wp_index += 1
            if self.wp_index >= len(ARC_WAYPOINTS):
                self._finish_stage(f"Arc done: all waypoints {ARC_WAYPOINTS} reached")
            else:
                self.message = f"Waypoint reached - next {ARC_WAYPOINTS[self.wp_index]:+d} deg"
            return

        # Speed: max 5 cm/s, linear slow-down inside the last ARC_SLOW_ZONE_DEG
        speed = ARC_MAX_SPEED * min(1.0, abs(err) / ARC_SLOW_ZONE_DEG)
        speed = clamp(speed, ARC_MIN_SPEED, ARC_MAX_SPEED)
        left = -math.copysign(speed, err)          # go right (+) -> strafe right (left < 0)
        # Feedforward: omega = v / R. Strafing right needs a LEFT turn to keep facing it.
        turn_ff = math.degrees(-left / ORBIT_RADIUS_CM)
        turn_corr = -K_TURN * off                  # bottle right -> turn right
        turn = clamp(turn_ff + turn_corr, -MAX_TURN, MAX_TURN)
        d = self.distance
        fwd = 0 if d is None else clamp(K_DIST_ORBIT * (d - TARGET_CM), -MAX_ORBIT_X, MAX_ORBIT_X)
        self.phase = f"arc -> {target:+d} deg"
        self.message = (f"Arc {self.arc_angle:+.0f} -> {target:+d} deg  "
                        f"({ 'right' if err > 0 else 'left'}, {speed:.1f} cm/s)"
                        + (f", {d:.0f} cm" if d is not None else ""))
        self._send(fwd, left, turn)

    def _orbit_recenter_turn(self, off):
        """Rule 4: turn in place by the exact angle of the offset (timed turn),
        then look again. Repeats until |offset| <= RECENTER_DONE_PX."""
        if self.recenter_tries >= RECENTER_MAX_TRIES:
            self.drive.stop(disable=True, reason="re-centering failed")
            self.look = None
            self._set("STOPPED", f"Re-centering failed after {RECENTER_MAX_TRIES} tries "
                                 f"({off:+.0f} px) - see log, press START")
            return
        self.recenter_tries += 1
        angle = math.degrees(math.atan(off / FOCAL_PX))
        turn = -math.copysign(max(abs(angle), MIN_TURN_STEP_DEG), angle)
        self.last_orbit_turn = turn
        self.orbit_lost_after_turn = True
        self.arc_phase = "recenter"
        self.message = (f"PAUSED - re-centering {self.recenter_tries}/{RECENTER_MAX_TRIES}: "
                        f"{off:+.0f} px -> turn {abs(turn):.0f} deg {'right' if turn < 0 else 'left'}")
        self._begin_look("orbit_recenter", ALIGN_LOOK_FRAMES)
        self.drive.turn_degrees(turn, ALIGN_TURN_SPEED,
                                f"orbit re-center {self.recenter_tries}: off={off:+.0f}px")

    def _orbit_look_result(self, hits):
        """A look window in ORBIT ended with hits: resume the arc or turn again."""
        off = statistics.median(h["offset"] for h in hits)
        if abs(off) <= RECENTER_DONE_PX:
            self.log.write(f"ORBIT: centered ({off:+.0f}px) - resume strafing")
            self.arc_phase = "arc"
            self.look = None
            self.recenter_tries = 0
            self.message = "Re-centered - resuming arc"
        else:
            self._orbit_recenter_turn(off)

    def _orbit_recovery_turn(self):
        """3 s without the bottle: ONE recovery turn, then look 2 s."""
        if self.orbit_lost_after_turn and self.last_orbit_turn:
            back = -self.last_orbit_turn * LOST_BACK_FRACTION     # undo half (side is wrong now)
            why = f"undo half of re-center turn {self.last_orbit_turn:+.0f}"
        else:                                      # lost while strafing, no turn in between:
            a = math.degrees(math.atan(abs(self.last_offset) / FOCAL_PX))   # last side is valid
            back = -math.copysign(max(a, ORBIT_RECOVER_MIN_DEG), self.last_offset or 1)
            why = f"toward last seen off={self.last_offset:+d}px"
        self.last_orbit_turn = None
        self.arc_phase = "back"
        self.message = f"Still not seen - one recovery turn ({abs(back):.0f} deg), then look 2 s"
        self.log.write(f"ORBIT: not seen for {ORBIT_LOST_LOOK_SEC:.0f}s -> recovery turn "
                       f"{back:+.0f} deg ({why})")
        self._begin_look("orbit_back")
        self.drive.turn_degrees(back, ALIGN_TURN_SPEED, f"orbit recovery: {why}")

    def status(self):
        with self.lock:
            return {
                "mission": self.state, "message": self.message,
                "distance": round(self.distance, 1) if self.distance is not None else None,
                "dist_method": self.dist_method, "stage": self.stage,
                "search_turns": self.search_turns,
                "phase": self.phase if self.state in ("SEARCH", "ALIGN", "ORBIT") else "",
                "align_tries": self.align_tries if self.state == "ALIGN" else 0,
                "log": self.log.path.name,
                "orbit": None if self.state != "ORBIT" and not (
                    self.state == "DONE" and self.stage == "orbit") else {
                    "waypoints": ARC_WAYPOINTS,
                    "index": self.wp_index,                       # = len -> all done
                    "target": (ARC_WAYPOINTS[self.wp_index]
                               if self.wp_index < len(ARC_WAYPOINTS) else None),
                    "angle": round(self.arc_angle, 1),
                    "phase": self.arc_phase,
                    "recentering": self.state == "ORBIT" and self.arc_phase == "recenter",
                    "lost": self.state == "ORBIT" and self.arc_phase in ("lost", "back"),
                    "source": self.heading_src,
                    "imu_bad": self.heading.imu_bad,
                    "imu_verified": self.heading.good_checks >= 1,
                },
            }


# --------------------------------------------------------------- dashboard ---
def make_orbit_handler(hub, recorder, mission):
    """bottle_detect's handler + the orbit page and START/STOP routes."""
    Base = bd.make_handler(hub, recorder)

    class OrbitHandler(Base):
        def do_GET(self):
            if self._route() == "/":
                self._send(200, ORBIT_DASHBOARD.read_bytes(), "text/html; charset=utf-8")
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

    return OrbitHandler


# ---------------------------------------------------------------- axis test ---
def axis_test(bob):
    """Short real moves BY MEANING, through the same send_xyz()/to_ugot() the
    mission uses - so a pass here means the mission's directions are right."""
    tests = [("FORWARD 6 cm/s", 6, 0, 0),
             ("STRAFE LEFT 6 cm/s", 0, 6, 0),
             ("TURN LEFT 20 deg/s", 0, 0, 20)]
    def yaw():
        try:
            data = bob.got.read_gyro_data()
            return float(data[2]) if data and any(data) else None
        except Exception:
            return None

    print("Axis test: each move lasts 1 second. Give Bob ~30 cm of free space.")
    try:
        for name, fwd, left, turn in tests:
            input(f"\nPress Enter -> Bob should {name}  (Ctrl+C to abort) ")
            y0 = yaw()
            ux, uy, uz = send_xyz(bob.got, fwd, left, turn)
            print(f"   sent mecanum_move_xyz(x={ux}, y={uy}, z={uz})")
            time.sleep(1.0)
            bob.got.mecanum_stop()
            time.sleep(0.3)
            y1 = yaw()
            print(f"   stopped. Did Bob {name}? If not, fix to_ugot() in bottle_orbit.py")
            if y0 is None or y1 is None:
                print("   IMU yaw: not available (ORBIT will use the turn commands)")
            else:
                dy = (y1 - y0 + 180.0) % 360.0 - 180.0
                print(f"   IMU yaw: {y0:.2f} -> {y1:.2f}  (change {dy:+.2f})")
                if turn:
                    sign = "+1" if dy > 0 else "-1"
                    print(f"   -> a LEFT turn changed yaw by {dy:+.1f}: set IMU_YAW_SIGN = {sign}"
                          f"  (expected ~+{turn:.0f} deg if yaw is in degrees)")
    finally:
        bob.got.mecanum_stop()
    print("\nAxis test done.")


# -------------------------------------------------------------------- main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default=bd.ROBOT_IP)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--dry-run", action="store_true", help="print speeds instead of moving")
    parser.add_argument("--stage", choices=["search", "approach", "orbit"], default="orbit",
                        help="stop after this stage (default: full mission)")
    parser.add_argument("--show", action="store_true", help="also show an OpenCV window")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--axis-test", action="store_true", help="check move directions, then exit")
    args = parser.parse_args()

    print("connecting to Bob at", args.ip)
    bob = bd.Bob(args.ip)
    bob.got.mecanum_stop()                         # start from a known, stopped state
    if args.axis_test:
        axis_test(bob)
        return

    print("loading", bd.YOLO_MODEL, f"(imgsz={IMGSZ})")
    model = YOLO(bd.YOLO_MODEL)

    log = RunLog(LOGS_DIR)
    log.write(f"RUN start  dry_run={args.dry_run} stage={args.stage} ip={args.ip} "
              f"imgsz={IMGSZ} conf={CONFIDENCE}")
    print("log file:", log.path)
    drive = Drive(bob, args.dry_run, log)
    signals = MissionSignals(bob)
    imu = ImuReader(bob, log)
    heading = Heading(drive, imu, log, use_imu=not args.dry_run)   # dry-run: Bob doesn't turn
    mission = Mission(drive, signals, args.stage, log, heading)

    # Any crash in ANY thread -> stop the wheels first
    default_hook = threading.excepthook
    def crash_hook(hook_args):
        drive.stop(disable=True, reason=f"CRASH in thread: {hook_args.exc_value!r}")
        default_hook(hook_args)
    threading.excepthook = crash_hook

    cam = bd.BobCamera(bob)
    hub = bd.Hub()
    recorder = bd.Recorder(bd.RECORDINGS_DIR, bd.RECORD_FPS)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_orbit_handler(hub, recorder, mission))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{args.port}"
    mode = "DRY-RUN (wheels never move)" if args.dry_run else "REAL WHEELS"
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
            det = find_bottle(model, frame)            # YOLO at IMGSZ / CONFIDENCE
            m = bd.measure(det, frame.shape[1])        # cx, offset, w, h, top_cut
            bd.annotate(frame, m)
            now = time.time()
            mission.step(m, frame_id, now)             # state machine -> wheel speeds

            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-3))
            t_prev = now

            ms = mission.status()
            cv2.putText(frame, f"{ms['mission']}  {fps:.1f} fps", (10, frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            recorder.write(frame)
            x, y, z = drive.last_sent
            status = {
                "state": m["state"], "cx": m.get("cx"), "offset": m.get("offset"),
                "w": m.get("w"), "h": m.get("h"),
                "conf": round(m["conf"], 2) if "conf" in m else None,
                "top_cut": m.get("top_cut", False),
                "frame_w": frame.shape[1], "frame_h": frame.shape[0],
                "tolerance": CENTER_TOL_PX, "fps": round(fps, 1),
                "cmd": {"x": x, "y": y, "z": z}, "dry_run": args.dry_run,
                "target": TARGET_CM, "dist_tol": DIST_TOL_CM,
                "imgsz": IMGSZ, "conf_threshold": CONFIDENCE,
            }
            status.update(ms)
            hub.publish(frame, status)

            if args.show:
                cv2.imshow("Bob orbit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\nCtrl+C - stopping")
        log.write("Ctrl+C")
    except Exception as e:
        log.write(f"CRASH in main loop: {e!r}")        # traceback still prints below
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
