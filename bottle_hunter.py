# bottle_hunter.py
# UGOT Engineer (Bob) - scan for bottles, center them in the camera, record video.
#
# Threads (same idea as remote.py from the CIM lab):
#   1. CameraThread    - keeps pulling the newest frame from the robot camera
#   2. DetectorThread  - runs YOLOv8s on the newest frame, finds and "locks" the bottle
#   3. MissionThread   - scan -> center -> auto-calibrate -> hold 3 s -> square of 4
#                        viewpoints around the bottle -> live tracking
#   4. Main thread     - shows the window, START / REC buttons, writes the video
#
# Run:
#   python bottle_hunter.py --test     (one-time hardware check)
#   python bottle_hunter.py            (full mission)
#
# Keys in the window: s = start mission, r = record on/off, c = calibrate distance, q = quit
# (or click the START / REC buttons with the mouse)

import argparse
import os
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from ugot import ugot
from ultralytics import YOLO

# ------------------------------------------------------------------ config ---
ROBOT_IP = "172.20.10.12"          # Bob

SCAN_STEP_DEG = 30                 # rotate 30 degrees between scan positions
SCAN_POSITIONS = 12                # 12 x 30 = full 360
TURN_SPEED = 45                    # deg/s
SETTLE_SEC = 0.8                   # wait after a turn before trusting the image

CAMERA_HFOV_DEG = 60               # approx. horizontal field of view of the camera
CENTER_TOLERANCE = 0.02            # bottle is "centered" if |error| < 2% of width
CENTER_GAIN = 0.7                  # turn only 70% of the estimated angle (no overshoot)
MIN_CENTER_STEP_DEG = 2
MAX_CENTER_STEP_DEG = 15
MAX_CENTER_ATTEMPTS = 30               # (includes the tries after moving closer)
CENTER_TURN_SPEED = 30             # too slow + too small = the wheels don't move at all
NO_PROGRESS_ERR = 0.01             # if the error changed less than this, the turn did nothing
NO_PROGRESS_BOOST_DEG = 1          # ...so add 1 more degree to the next step (and again...)
CENTER_SAMPLES = 4                 # median of the bottle position over 4 detections
CENTERED_CONFIRMATIONS = 2         # must be centered 2 times in a row
OSCILLATION_ACCEPT = 0.05          # jumping between -5% and +5% twice = accept as centered
APPROACH_AFTER_ATTEMPTS = 6        # not centered after 6 tries -> drive a bit closer
APPROACH_STEP_CM = 10              # ...this much
APPROACH_MAX_TIMES = 3             # at most 3 times (30 cm) per centering
APPROACH_MIN_DIST_CM = 60          # only if the bottle is farther than this

# Scanning: look at several frames at each position, not just one
SCAN_LOOK_FRAMES = 8               # check 8 fresh detections at every scan position
SCAN_MIN_HITS = 2                  # bottle must appear in at least 2 of them

# Distance estimation (pinhole camera):  distance = focal_px * real_height / box_height_px
BOTTLE_HEIGHT_CM = 22.0            # real bottle height
CALIB_DISTANCE_CM = 50.0           # manual calibration (optional): bottle at 50 cm, press C
FOCAL_FILE = "focal.txt"           # the calibration result is saved here
HEIGHT_SAMPLES = 7                 # median of the last 7 box heights (less jumpy)

# Auto calibration: drive forward a known distance and compare the two box heights
AUTO_CALIB_MOVE_CM = 20
AUTO_CALIB_SPEED = 10              # cm/s
MOVE_FORWARD = 0
MOVE_BACKWARD = 1
UNIT_CM = 1

# Live tracking after the first centering
TRACK_TRIGGER = 0.05               # re-center when the bottle drifts > 5% from the middle
LOST_TIMEOUT_SEC = 2.0             # no bottle for 2 s -> go back to scanning

# Square around the bottle (4 viewpoints, 90 deg apart, each at the same distance d)
#   turn 45 -> drive sqrt(2)*d -> turn 135 the other way -> bottle is in front again
HOLD_CENTERED_SEC = 3.0            # bottle must stay centered 3 s before each move
SQUARE_CORNERS = 4
SQUARE_FIRST_TURN = -1             # -1 = first turn LEFT 45 (then RIGHT 135), +1 = mirrored
SQUARE_MAX_DIST_CM = 120           # safety: don't do the square if the bottle is farther
SQUARE_MIN_DIST_CM = 25            # ...or closer than this
SQUARE_SPEED = 20                  # cm/s while driving the legs
RESULTS_FILE = "square_results.csv"  # every square adds one row (Q1 start ... Q1 end)
LOCAL_SEARCH_TURNS = [15, -30, -15, 60]   # if the bottle is not in view after a leg:
                                          # look at +15, -15, -30, +30 deg

YOLO_MODEL = "yolov8s.pt"          # "small" model: much better on a clear bottle than "nano"
                                   # (downloaded automatically on the first run, ~22 MB)
BOTTLE_CLASS_ID = 39               # "bottle" in the COCO dataset
MIN_CONFIDENCE = 0.30              # a NEW bottle needs at least this
LOW_CONFIDENCE = 0.12              # a bottle we already follow may drop to this
LOCK_GATE = 0.20                   # ...if it is within 20% of the width from where it was
EXPECT_GATE = 0.35                 # after a leg: weak detections allowed within 35% of center
LOCK_TIMEOUT_SEC = 3.0             # forget the lock after 3 s without seeing the bottle
MEMORY_SEC = 0.4                   # keep the last box for 0.4 s if YOLO blinks (only when still)

# UGOT chassis constants
TURN_LEFT = 2
TURN_RIGHT = 3
UNIT_DEGREES = 2

RECORD_DIR = "recordings"
RECORD_FPS = 15.0


# ------------------------------------------------------------ shared state ---
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None              # newest raw frame (BGR)
        self.frame_time = 0.0
        self.detection = None          # (x1, y1, x2, y2, conf) or None
        self.detection_time = 0.0      # time of the frame the detection came from
        self.status = "Ready - press S to start"
        self.status_color = (255, 255, 255)
        self.busy = False              # robot is moving
        self.recording = False
        self.mission_running = False
        self.stop = False              # quit the program
        self.abort = False             # stop the mission only (START button again)
        self.box_heights = deque(maxlen=HEIGHT_SAMPLES)   # recent bottle heights in pixels
        self.box_cut = False           # True if the box touches the top/bottom of the image
        self.focal_px = None           # from calibration
        # "lock" on the bottle we follow: where it was last seen (0..1 of width)
        self.lock_cx = None
        self.lock_time = 0.0
        self.lock_gate = LOCK_GATE
        self.last_good = None          # last real detection (for the short memory)
        self.last_good_time = 0.0
        self.square_log = []           # [(label, distance_cm), ...] of the current square


state = SharedState()


def clear_lock():
    """Forget where the bottle was (call before a big move)."""
    with state.lock:
        state.lock_cx = None
        state.last_good = None


def expect_bottle_at_center():
    """After a leg of the square we KNOW the bottle should be roughly in front,
    so accept also weak detections near the middle of the image."""
    with state.lock:
        state.lock_cx = 0.5
        state.lock_time = time.time()
        state.lock_gate = EXPECT_GATE
        state.last_good = None


def halted():
    return state.stop or state.abort


# -------------------------------------------------------- distance helpers ---
def load_focal():
    if os.path.exists(FOCAL_FILE):
        with open(FOCAL_FILE) as f:
            state.focal_px = float(f.read().strip())
        print(f"loaded focal length: {state.focal_px:.1f} px")
    else:
        print("no calibration yet - it will be done automatically on the first bottle")


def calibrate():
    with state.lock:
        heights = list(state.box_heights)
        cut = state.box_cut
    if len(heights) < 3:
        print("calibration failed: no bottle in view")
        return
    if cut:
        print("calibration failed: bottle is cut by the image edge - move it back")
        return
    h_px = float(np.median(heights))
    focal = h_px * CALIB_DISTANCE_CM / BOTTLE_HEIGHT_CM
    save_focal(focal)
    print(f"calibrated: box height {h_px:.1f} px -> focal length {focal:.1f} px (saved)")


def save_focal(focal):
    with open(FOCAL_FILE, "w") as f:
        f.write(f"{focal:.2f}")
    with state.lock:
        state.focal_px = focal


def fresh_height_px(min_samples=5, timeout=3.0):
    """Throw away old box heights and collect new ones. Returns median height or None."""
    with state.lock:
        state.box_heights.clear()
    start = time.time()
    while time.time() - start < timeout and not state.stop:
        with state.lock:
            heights = list(state.box_heights)
            cut = state.box_cut
        if len(heights) >= min_samples:
            return None if cut else float(np.median(heights))
        time.sleep(0.05)
    return None


def current_distance_cm():
    """Distance estimate in cm, or None if not possible right now."""
    with state.lock:
        heights = list(state.box_heights)
        focal = state.focal_px
        cut = state.box_cut
    if focal is None or len(heights) < 3 or cut:
        return None
    return focal * BOTTLE_HEIGHT_CM / float(np.median(heights))


# ----------------------------------------------------------- robot helpers ---
class Robot:
    """Small wrapper around the ugot library, so every robot call is in one place."""

    def __init__(self, ip):
        self.got = ugot.UGOT()
        self.got.initialize(ip)
        self.lock = threading.Lock()   # the camera and the mission share one connection

    def open_camera(self):
        with self.lock:
            self.got.open_camera()

    def read_frame(self):
        with self.lock:
            data = self.got.read_camera_data()
        if not data:
            return None
        buf = np.frombuffer(data, np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    def turn(self, degrees, speed=TURN_SPEED):
        """degrees > 0 = right, degrees < 0 = left. Blocks until the turn ends."""
        direction = TURN_RIGHT if degrees > 0 else TURN_LEFT
        amount = int(round(abs(degrees)))
        if amount == 0:
            return
        # no lock here: a long move must not freeze the camera thread
        self.got.mecanum_turn_speed_times(direction, speed, amount, UNIT_DEGREES)
        time.sleep(0.1)

    def move(self, cm, speed=AUTO_CALIB_SPEED):
        """cm > 0 = forward, cm < 0 = backward. Blocks until the move ends."""
        direction = MOVE_FORWARD if cm > 0 else MOVE_BACKWARD
        amount = int(round(abs(cm)))
        if amount == 0:
            return
        self.got.mecanum_move_speed_times(direction, speed, amount, UNIT_CM)
        time.sleep(0.1)

    def stop_wheels(self):
        with self.lock:
            self.got.mecanum_stop()

    def led(self, r, g, b):
        with self.lock:
            self.got.show_light_rgb_effect(r, g, b, 0)   # effect 0 = steady light

    def say(self, text):
        try:
            with self.lock:
                self.got.play_audio_tts(text, 0, False)
        except Exception as e:
            print("sound failed:", e)

    def green(self):
        self.led(0, 255, 0)

    def red(self):
        self.led(255, 0, 0)


# ----------------------------------------------------------------- threads ---
def camera_thread(robot):
    robot.open_camera()
    while not state.stop:
        frame = robot.read_frame()
        if frame is None:
            time.sleep(0.02)
            continue
        with state.lock:
            state.frame = frame
            state.frame_time = time.time()


def detector_thread():
    model = YOLO(YOLO_MODEL)
    last_used = 0.0
    while not state.stop:
        with state.lock:
            frame = state.frame
            t = state.frame_time
        if frame is None or t == last_used:
            time.sleep(0.01)
            continue
        last_used = t

        result = model(frame, classes=[BOTTLE_CLASS_ID], conf=LOW_CONFIDENCE, verbose=False)[0]
        w = frame.shape[1]
        with state.lock:
            lock_cx, lock_t, gate = state.lock_cx, state.lock_time, state.lock_gate
            busy = state.busy
            last_det, last_t = state.last_good, state.last_good_time
        locked = lock_cx is not None and t - lock_t < LOCK_TIMEOUT_SEC

        # High confidence = always a bottle. Low confidence = a bottle only if it is
        # where we expect it (near the bottle we are already following).
        best, best_score = None, None
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cx = (x1 + x2) / 2 / w
            near = locked and abs(cx - lock_cx) < gate
            if conf < MIN_CONFIDENCE and not near:
                continue
            # prefer the box closest to the lock, otherwise the biggest
            score = -abs(cx - lock_cx) if locked else (x2 - x1) * (y2 - y1)
            if best is None or score > best_score:
                best = (int(x1), int(y1), int(x2), int(y2), conf)
                best_score = score

        # short memory: if YOLO blinked for a moment while standing still,
        # keep the last box (never while moving - then it would be wrong)
        if best is None and not busy and last_det is not None and t - last_t < MEMORY_SEC:
            best = last_det

        with state.lock:
            state.detection = best
            state.detection_time = t
            if best is not None and best is not last_det:
                state.last_good, state.last_good_time = best, t
                state.lock_cx = (best[0] + best[2]) / 2 / w
                state.lock_time = t
                state.lock_gate = LOCK_GATE
            if best:
                fh = frame.shape[0]
                state.box_heights.append(best[3] - best[1])
                state.box_cut = best[1] <= 2 or best[3] >= fh - 2
            else:
                state.box_heights.clear()


def wait_for_fresh_detection(after_time, timeout=3.0):
    """Wait until we have a detection result from a frame taken after `after_time`."""
    start = time.time()
    while time.time() - start < timeout and not state.stop:
        with state.lock:
            if state.detection_time > after_time:
                frame_w = state.frame.shape[1] if state.frame is not None else None
                return state.detection, frame_w
        time.sleep(0.03)
    return None, None


def set_status(text, color):
    print(text)
    with state.lock:
        state.status = text
        state.status_color = color


def measure_error():
    """Average horizontal error of the bottle over a few fresh detections.
    Returns error in -0.5 ... +0.5 (negative = bottle is left of center), or None."""
    errors = []
    t = time.time()
    for _ in range(CENTER_SAMPLES):
        det, w = wait_for_fresh_detection(t)
        if det is None:
            continue
        x1, _, x2, _, _ = det
        errors.append(((x1 + x2) / 2 - w / 2) / w)
        with state.lock:
            t = state.detection_time
    if not errors:
        return None
    return float(np.median(errors))


def center_bottle(robot):
    """Turn little by little until the bottle is in the middle of the image.
    If a small turn didn't move the image (the wheels didn't overcome friction),
    the next step is made bigger, until the robot really moves."""
    confirmations = 0
    last_error = None
    boost = 0
    flips = 0                  # how many times the bottle jumped from one side to the other
    tries_here = 0             # failed attempts since the last move closer
    approaches = 0
    for attempt in range(1, MAX_CENTER_ATTEMPTS + 1):
        if halted():
            return False
        error = measure_error()
        if error is None:
            return False                          # lost the bottle

        if abs(error) < CENTER_TOLERANCE:
            print(f"  centering attempt {attempt}: error = {error:+.3f}  OK")
            confirmations += 1
            if confirmations >= CENTERED_CONFIRMATIONS:
                return True
            time.sleep(0.3)
            continue
        confirmations = 0

        # Jumping left-right around the middle: the smallest turn the wheels can
        # make is bigger than the tolerance -> this is as good as it gets.
        if last_error is not None and np.sign(error) != np.sign(last_error) \
                and abs(error) < OSCILLATION_ACCEPT and abs(last_error) < OSCILLATION_ACCEPT:
            flips += 1
            if flips >= 2:
                print(f"  centering attempt {attempt}: error = {error:+.3f}  "
                      f"OK (left-right jumping - best the wheels can do)")
                return True

        # Not getting centered after a few tries -> drive a little closer and try again
        tries_here += 1
        if tries_here > APPROACH_AFTER_ATTEMPTS and approaches < APPROACH_MAX_TIMES:
            dist = current_distance_cm()
            if dist is not None and dist > APPROACH_MIN_DIST_CM:
                approaches += 1
                tries_here = 0
                flips = 0
                last_error = None
                boost = 0
                print(f"  not centered after {APPROACH_AFTER_ATTEMPTS} tries at ~{dist:.0f} cm "
                      f"-> moving {APPROACH_STEP_CM} cm closer ({approaches}/{APPROACH_MAX_TIMES})")
                set_status(f"Moving {APPROACH_STEP_CM} cm closer to center better...", (0, 255, 255))
                robot.say("Moving closer")
                set_busy(True)
                robot.move(APPROACH_STEP_CM, speed=AUTO_CALIB_SPEED)
                set_busy(False)
                time.sleep(SETTLE_SEC)
                continue

        # did the last turn do anything?
        if last_error is not None and abs(error - last_error) < NO_PROGRESS_ERR \
                and np.sign(error) == np.sign(last_error):
            boost += NO_PROGRESS_BOOST_DEG
        elif last_error is not None:
            boost = max(0, boost - NO_PROGRESS_BOOST_DEG)
        last_error = error

        step = error * CAMERA_HFOV_DEG * CENTER_GAIN
        size = min(max(abs(step), MIN_CENTER_STEP_DEG) + boost, MAX_CENTER_STEP_DEG)
        print(f"  centering attempt {attempt}: error = {error:+.3f} -> turn {np.sign(step) * size:+.0f} deg"
              + (f" (boost +{boost})" if boost else ""))
        set_busy(True)
        robot.turn(np.sign(step) * size, speed=CENTER_TURN_SPEED)
        set_busy(False)
        time.sleep(SETTLE_SEC)
    return False


def look_for_bottle():
    """Check several fresh frames. Returns a detection if the bottle was seen
    in at least SCAN_MIN_HITS of them, else None. (One frame is not enough -
    YOLO misses a small bottle in some frames.)"""
    hits = []
    t = time.time()
    for _ in range(SCAN_LOOK_FRAMES):
        if halted():
            return None
        det, _ = wait_for_fresh_detection(t)
        with state.lock:
            t = state.detection_time
        if det is not None:
            hits.append(det)
            if len(hits) >= SCAN_MIN_HITS:
                return hits[-1]
    return None


def set_busy(value):
    with state.lock:
        state.busy = value


def auto_calibrate(robot):
    """Find the focal length by itself:
    h1 = box height now, drive forward D cm, h2 = box height there.
    Pinhole: h = f*H/d  ->  d1 = D*h2/(h2-h1),  f = h1*d1/H."""
    set_status("Auto-calibrating distance...", (0, 200, 255))
    h1 = fresh_height_px()
    if h1 is None:
        print("auto-calibration: bottle not clear at start")
        return False

    set_busy(True)
    robot.move(AUTO_CALIB_MOVE_CM)
    set_busy(False)
    time.sleep(SETTLE_SEC)
    h2 = fresh_height_px()

    set_busy(True)
    robot.move(-AUTO_CALIB_MOVE_CM)               # go back to where we were
    set_busy(False)
    time.sleep(SETTLE_SEC)

    if h2 is None or h2 <= h1 * 1.05:
        print(f"auto-calibration failed (h1={h1}, h2={h2}) - bottle too far, cut, or lost")
        return False

    d1 = AUTO_CALIB_MOVE_CM * h2 / (h2 - h1)
    focal = h1 * d1 / BOTTLE_HEIGHT_CM
    if not 20 <= d1 <= 500:
        print(f"auto-calibration gave a strange distance ({d1:.0f} cm) - ignored")
        return False
    save_focal(focal)
    print(f"auto-calibrated: h1={h1:.1f}px h2={h2:.1f}px -> d={d1:.0f}cm, focal={focal:.1f}px")
    return True


def fallback_focal_from_fov():
    """Rough focal length from the camera's field of view (used if auto-calibration fails)."""
    with state.lock:
        w = state.frame.shape[1] if state.frame is not None else 640
    return (w / 2) / np.tan(np.radians(CAMERA_HFOV_DEG / 2))


def announce_distance(robot, prefix):
    if fresh_height_px() is None:             # refresh the samples after the turn
        robot.say(prefix)
        set_status(f"{prefix} (distance unknown)", (0, 255, 0))
        return
    dist = current_distance_cm()
    robot.say(f"{prefix}, {dist:.0f} centimeters")
    set_status(f"{prefix} - distance ~{dist:.0f} cm", (0, 255, 0))


def scan_and_center(robot):
    """Scan in 30 deg steps until a bottle is found and centered. True/False."""
    for pos in range(SCAN_POSITIONS):
        if halted():
            return False
        angle = pos * SCAN_STEP_DEG
        set_status(f"Looking at {angle} deg...", (0, 200, 255))
        det = look_for_bottle()

        if det is not None:
            robot.green()
            robot.say("Bottle found")
            set_status(f"Bottle FOUND at {angle} deg - centering...", (0, 255, 0))
            if center_bottle(robot):
                return True
            robot.red()
            set_status("Lost the bottle - continuing scan", (0, 0, 255))
        else:
            robot.red()
            set_status(f"No bottle at {angle} deg", (0, 0, 255))

        set_busy(True)
        robot.turn(SCAN_STEP_DEG)
        set_busy(False)
        time.sleep(SETTLE_SEC)
    return False


def measure_distance_cm():
    """Fresh distance measurement (median of several boxes), or None."""
    h = fresh_height_px(min_samples=7)
    if h is None or state.focal_px is None:
        return None
    return state.focal_px * BOTTLE_HEIGHT_CM / h


def hold_centered(seconds):
    """Watch the bottle for `seconds`.
    True = stayed centered the whole time, False = it moved, None = lost."""
    start = time.time()
    lost_since = None
    while time.time() - start < seconds:
        if halted():
            return None
        error = measure_error()
        if error is None:
            lost_since = lost_since or time.time()
            if time.time() - lost_since > LOST_TIMEOUT_SEC:
                return None
            continue
        lost_since = None
        if abs(error) > TRACK_TRIGGER:
            return False
        left = seconds - (time.time() - start)
        set_status(f"Holding center... {max(0, left):.1f}s", (0, 255, 0))
    return True


def wait_until_stable(robot):
    """Re-center as needed until the bottle stays centered HOLD_CENTERED_SEC. True/False."""
    while not halted():
        result = hold_centered(HOLD_CENTERED_SEC)
        if result is True:
            return True
        if result is None:
            return False
        set_status("Bottle moved - re-centering", (0, 255, 255))
        if not center_bottle(robot):
            return False
    return False


def find_bottle_nearby(robot):
    """After a leg the bottle should be in front. If not, look a little left/right."""
    if look_for_bottle() is not None:
        return True
    offset = 0
    for t in LOCAL_SEARCH_TURNS:
        if halted():
            return False
        set_busy(True)
        robot.turn(t, speed=CENTER_TURN_SPEED)
        set_busy(False)
        offset += t
        time.sleep(SETTLE_SEC)
        set_status(f"Searching nearby ({offset:+d} deg)...", (0, 200, 255))
        if look_for_bottle() is not None:
            return True
    robot.turn(-offset, speed=CENTER_TURN_SPEED)    # back to where we were
    return False


def do_square(robot):
    """Visit 4 viewpoints around the bottle, 90 deg apart, at the same distance.

    Geometry (bottle at distance d straight ahead, at B = (d, 0)):
      turn 45 to the side, drive sqrt(2)*d  ->  robot is at (d, d),
      exactly d away from the bottle, which is now 135 deg to the other side.
      turn 135 the other way -> facing the bottle again, from a new angle.
    4 times = back at the start (a square around the bottle).
    """
    s = SQUARE_FIRST_TURN
    names = [f"Q{i + 1}" for i in range(SQUARE_CORNERS)] + ["Q1"]   # Q1 Q2 Q3 Q4 Q1
    with state.lock:
        state.square_log = []                      # shown on screen during the square

    for k, name in enumerate(names):
        if halted():
            return False
        last = k == len(names) - 1
        label = "Q1 end" if last else ("Q1 start" if k == 0 else name)

        # --- drive to this viewpoint (not needed for Q1 start - we are already there)
        if k > 0:
            leg = np.sqrt(2) * d
            print(f"leg {k}/{SQUARE_CORNERS}: d={d:.0f} cm -> turn {45*s:+d}, "
                  f"drive {leg:.0f} cm, turn {-135*s:+d}  -> {label}")
            set_status(f"Leg {k}/{SQUARE_CORNERS}: drive {leg:.0f} cm -> {label}", (0, 255, 255))
            clear_lock()
            set_busy(True)
            robot.turn(45 * s)
            time.sleep(0.3)
            robot.move(leg, speed=SQUARE_SPEED)
            time.sleep(0.3)
            robot.turn(-135 * s)
            set_busy(False)
            expect_bottle_at_center()              # geometry says: it's in front of us
            time.sleep(SETTLE_SEC)

            if not find_bottle_nearby(robot):
                robot.red()
                robot.say(f"Lost the bottle on the way to {name}")
                set_status(f"Lost the bottle on the way to {label}", (0, 0, 255))
                return False
            robot.green()
            if not center_bottle(robot):
                robot.red()
                set_status(f"Could not center at {label}", (0, 0, 255))
                return False

        # --- centered for 3 s, then measure and report this viewpoint
        if not wait_until_stable(robot):
            robot.red()
            return False
        d = measure_distance_cm()
        if d is None:
            robot.red()
            robot.say("Cannot measure distance")
            set_status(f"Square stopped - no distance at {label}", (0, 0, 255))
            return False

        record_viewpoint(label, d)
        robot.green()
        robot.say(f"{label.replace('Q', 'Q ')}, {d:.0f} centimeters")
        set_status(f"{label}: {d:.0f} cm", (0, 255, 0))

        if not last and not SQUARE_MIN_DIST_CM <= d <= SQUARE_MAX_DIST_CM:
            robot.red()
            robot.say(f"Distance {d:.0f} centimeters is not safe for the square")
            set_status(f"Square stopped - d={d:.0f} cm (allowed "
                       f"{SQUARE_MIN_DIST_CM}-{SQUARE_MAX_DIST_CM})", (0, 0, 255))
            return False

    report_square(robot)
    return True


def record_viewpoint(label, d):
    with state.lock:
        state.square_log.append((label, d))
    print(f"  >>> {label}: {d:.1f} cm")


def report_square(robot):
    """Print / save / say the results, including Q1 start vs Q1 end."""
    with state.lock:
        log = list(state.square_log)
    start, end = log[0][1], log[-1][1]
    diff = end - start
    print("\n========== square results ==========")
    for label, d in log:
        print(f"  {label:9s} {d:6.1f} cm")
    print(f"  Q1 end - Q1 start = {diff:+.1f} cm  ({100 * diff / start:+.1f}%)")
    print("====================================\n")

    new_file = not os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a") as f:
        if new_file:
            f.write("time,Q1_start_cm,Q2_cm,Q3_cm,Q4_cm,Q1_end_cm,Q1_diff_cm\n")
        f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ","
                + ",".join(f"{d:.1f}" for _, d in log) + f",{diff:.1f}\n")
    print(f"results added to {RESULTS_FILE}")

    robot.green()
    robot.say(f"Square complete. Q 1 start {start:.0f}, Q 1 end {end:.0f}, "
              f"difference {abs(diff):.0f} centimeters")
    set_status(f"Square done: Q1 {start:.0f} -> {end:.0f} cm ({diff:+.0f})", (0, 255, 0))


def track(robot):
    """Keep the bottle in the middle. Returns when the bottle is lost."""
    lost_since = None
    while not halted():
        error = measure_error()
        if error is None:
            lost_since = lost_since or time.time()
            if time.time() - lost_since > LOST_TIMEOUT_SEC:
                return
            continue
        lost_since = None

        if abs(error) > TRACK_TRIGGER:
            set_status(f"Bottle moved ({error:+.2f}) - re-centering", (0, 255, 255))
            if center_bottle(robot):
                robot.green()
                announce_distance(robot, "Centered")
                set_status("Tracking the bottle...", (0, 255, 0))
        time.sleep(0.05)


def mission_thread(robot):
    with state.lock:
        state.mission_running = True
        state.abort = False
    try:
        while not halted():
            if not scan_and_center(robot):
                if halted():
                    break
                robot.red()
                robot.say("No bottle found, scanning again")
                set_status("Full 360 - no bottle, scanning again", (0, 0, 255))
                continue                                # keep searching until STOP

            robot.green()
            if state.focal_px is None:
                if not auto_calibrate(robot):
                    focal = fallback_focal_from_fov()
                    with state.lock:
                        state.focal_px = focal          # rough, not saved to file
                    print(f"using rough focal from FOV: {focal:.1f}px")
                center_bottle(robot)                     # re-center after driving back
            robot.say("Bottle centered")              # the distance is reported as Q1 start

            # Q1 start (hold 3 s + measure) -> Q2 -> Q3 -> Q4 -> Q1 end
            do_square(robot)
            if halted():
                break

            set_status("Tracking the bottle...", (0, 255, 0))
            track(robot)
            if halted():
                break

            robot.red()
            robot.say("Lost the bottle, scanning again")
            set_status("Bottle lost - scanning again", (0, 0, 255))
    finally:
        robot.stop_wheels()
        if state.abort:
            set_status("Mission stopped - press START", (255, 255, 255))
        with state.lock:
            state.busy = False
            state.mission_running = False
            state.abort = False


# --------------------------------------------------------- window & buttons ---
BTN_START = (10, 60, 130, 100)     # x1, y1, x2, y2
BTN_REC = (140, 60, 260, 100)
click_queue = []


def on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    for name, (x1, y1, x2, y2) in (("start", BTN_START), ("rec", BTN_REC)):
        if x1 <= x <= x2 and y1 <= y <= y2:
            click_queue.append(name)


def draw_overlay(frame, det, status, color, busy, recording):
    h, w = frame.shape[:2]
    out = frame.copy()

    # center line + tolerance band
    cx = w // 2
    tol = int(CENTER_TOLERANCE * w)
    cv2.line(out, (cx, 0), (cx, h), (0, 0, 255), 2)                  # red = exact center
    cv2.line(out, (cx - tol, 0), (cx - tol, h), (255, 255, 0), 1)    # cyan = tolerance
    cv2.line(out, (cx + tol, 0), (cx + tol, h), (255, 255, 0), 1)

    if det is not None:
        x1, y1, x2, y2, conf = det
        bx = (x1 + x2) // 2
        err = (bx - cx) / w
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(out, (bx, (y1 + y2) // 2), 5, (0, 255, 0), -1)
        cv2.putText(out, f"err {err:+.3f}", (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(out, f"bottle {conf:.2f}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        dist = current_distance_cm()
        with state.lock:
            focal, cut = state.focal_px, state.box_cut
        if focal is None:
            dist_text, dist_color = "dist: not calibrated yet", (0, 200, 255)
        elif cut:
            dist_text, dist_color = "dist: bottle cut", (0, 0, 255)
        elif dist is None:
            dist_text, dist_color = "dist: ...", (0, 200, 255)
        else:
            dist_text, dist_color = f"dist ~{dist:.0f} cm", (0, 255, 0)
        cv2.putText(out, dist_text, (x1, y2 + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, dist_color, 2)

    cv2.putText(out, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(out, "Busy" if busy else "Ready", (w - 90, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255) if busy else (0, 255, 0), 2)

    # square results panel (bottom-left), so the video shows every Q measurement
    with state.lock:
        log = list(state.square_log)
    if log:
        lines = [f"{label:8s} {d:5.0f} cm" for label, d in log]
        if len(log) == SQUARE_CORNERS + 1:
            diff = log[-1][1] - log[0][1]
            lines.append(f"Q1 diff  {diff:+5.0f} cm")
        y0 = h - 12 - 22 * len(lines)
        cv2.rectangle(out, (6, y0 - 22), (190, h - 6), (0, 0, 0), -1)
        for i, line in enumerate(lines):
            col = (0, 255, 255) if line.startswith("Q1") else (255, 255, 255)
            cv2.putText(out, line, (12, y0 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)

    # buttons
    with state.lock:
        running = state.mission_running
    x1, y1, x2, y2 = BTN_START
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 140, 0) if running else (60, 60, 60), -1)
    cv2.putText(out, "HALT" if running else "START", (x1 + 18, y2 - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    x1, y1, x2, y2 = BTN_REC
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 200) if recording else (60, 60, 60), -1)
    cv2.putText(out, "STOP" if recording else "REC", (x1 + 25, y2 - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if recording and int(time.time() * 2) % 2 == 0:
        cv2.circle(out, (w - 25, h - 25), 10, (0, 0, 255), -1)
    return out


def start_mission(robot):
    """START starts the mission; pressing it again while running stops it."""
    with state.lock:
        if state.mission_running:
            state.abort = True
            print("stopping the mission...")
            return
    threading.Thread(target=mission_thread, args=(robot,), daemon=True).start()


def toggle_record(writer, frame_shape):
    with state.lock:
        state.recording = not state.recording
        rec = state.recording
    if rec:
        os.makedirs(RECORD_DIR, exist_ok=True)
        path = os.path.join(RECORD_DIR, datetime.now().strftime("bob_%Y%m%d_%H%M%S.mp4"))
        h, w = frame_shape[:2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), RECORD_FPS, (w, h))
        print("recording to", path)
        return writer
    if writer is not None:
        writer.release()
        print("recording saved")
    return None


def main_loop(robot):
    cv2.namedWindow("Bob camera")
    cv2.setMouseCallback("Bob camera", on_mouse)
    writer = None
    last_write = 0.0

    while True:
        with state.lock:
            frame = None if state.frame is None else state.frame.copy()
            det = state.detection
            status, color = state.status, state.status_color
            busy, recording = state.busy, state.recording

        if frame is None:
            blank = np.zeros((240, 480, 3), np.uint8)
            cv2.putText(blank, "Waiting for camera...", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Bob camera", blank)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
            continue

        shown = draw_overlay(frame, det, status, color, busy, recording)
        cv2.imshow("Bob camera", shown)

        if recording and writer is not None and time.time() - last_write >= 1.0 / RECORD_FPS:
            writer.write(shown)
            last_write = time.time()

        key = cv2.waitKey(1) & 0xFF
        actions = list(click_queue)
        click_queue.clear()
        if key == ord('s'):
            actions.append("start")
        if key == ord('r'):
            actions.append("rec")
        if key == ord('c'):
            calibrate()
        if key == ord('q'):
            break

        for a in actions:
            if a == "start":
                start_mission(robot)
            elif a == "rec":
                writer = toggle_record(writer, frame.shape)

    state.stop = True
    if writer is not None:
        writer.release()
        print("recording saved")
    robot.stop_wheels()
    cv2.destroyAllWindows()


# --------------------------------------------------------------- self test ---
def self_test(robot):
    print("1) LED red");   robot.red();   time.sleep(1)
    print("2) LED green"); robot.green(); time.sleep(1)
    print("3) Sound");     robot.say("Hello, I am Bob"); time.sleep(2)
    print("4) Turn right 30 degrees"); robot.turn(30); time.sleep(0.5)
    print("5) Camera")
    robot.open_camera()
    time.sleep(1)
    frame = robot.read_frame()
    if frame is None:
        print("   camera returned nothing")
    else:
        cv2.imwrite("test_frame.jpg", frame)
        print("   saved test_frame.jpg, size:", frame.shape)
    print("Self test done.")


# -------------------------------------------------------------------- main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default=ROBOT_IP)
    parser.add_argument("--test", action="store_true", help="hardware self test only")
    args = parser.parse_args()

    print("connecting to", args.ip)
    bob = Robot(args.ip)
    print("connected")

    if args.test:
        self_test(bob)
    else:
        load_focal()
        threading.Thread(target=camera_thread, args=(bob,), daemon=True).start()
        threading.Thread(target=detector_thread, daemon=True).start()
        main_loop(bob)
    print("bye bye")
