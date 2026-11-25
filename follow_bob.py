"""
follow_bob.py - Leader-Follower: Alex follows Bob and keeps TARGET_CM distance.
Rotation  -> keeps Bob centered (same as track_rotate.py)
Forward/back -> keeps the distance at TARGET_CM

Usage:
  1) Drive-direction test (once):   python follow_bob.py --test-drive
     Alex drives slowly for 1 second. It should drive FORWARD (toward where the camera looks).
     If it drives BACKWARD -> change DRIVE_SIGN below from 1 to -1.
  2) Following:                     python follow_bob.py
     Move Bob slowly. Alex should keep ~60 cm from him and face him.
     Press 'q' in the video window to stop (Alex stops immediately).
"""
import argparse
import math
import time
from collections import deque

import cv2
import numpy as np
from ugot import ugot

# ---------------- settings ----------------
ROBOT_IP = "172.20.10.10"   # Alex
MARKER_SIZE_CM = 10.0
FOCAL_PX = 428.0
MARKER_ID = 0

TURN_SIGN = -1       # already verified with track_rotate.py
DRIVE_SIGN = 1       # set to -1 if --test-drive drives BACKWARD

# rotation
KP_TURN = 1.0        # deg/s per deg of bearing
MAX_TURN = 30        # deg/s
DEADBAND_DEG = 3.0

# distance
TARGET_CM = 60.0     # distance to keep from Bob
KP_DRIVE = 0.8       # cm/s per cm of distance error
MAX_DRIVE = 15       # cm/s - slow for the first runs
DEADBAND_CM = 3.0
SAFETY_CM = 45.0     # closer than this -> back off at full MAX_DRIVE
MAX_BEARING_TO_DRIVE = 15.0  # if Bob is far off-center -> only turn, don't drive

LOST_TIMEOUT = 0.5   # seconds without marker -> stop
MEDIAN_N = 5         # median filter over last N distance readings (removes spikes)
# ------------------------------------------

DETECTOR = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
    cv2.aruco.DetectorParameters())


def read_frame(got):
    data = got.read_camera_data()
    if data is None:
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def find_marker(img):
    """Biggest (= closest) ID-0 marker. Later, with markers on all robots,
    this means 'follow the closest robot in view'."""
    corners_list, ids, _ = DETECTOR.detectMarkers(img)
    if ids is None:
        return None
    best = None
    for c, i in zip(corners_list, ids.flatten()):
        if i != MARKER_ID:
            continue
        pts = c.reshape(4, 2)
        tl, tr, br, bl = pts
        h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
        if best is None or h > best[1]:
            best = (pts[:, 0].mean(), h, pts)
    return best


def clamp(v, lim):
    return max(-lim, min(lim, v))


def move(got, x_cm_s, z_deg_s):
    """x > 0 = forward, z > 0 = turn right (after signs are set correctly)."""
    # UGOT: mecanum_move_xyz(sideways, forward, rotation) - 1st arg is sideways (right), 2nd is forward
    got.mecanum_move_xyz(0, int(round(DRIVE_SIGN * x_cm_s)), int(round(TURN_SIGN * z_deg_s)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-drive", action="store_true")
    args = ap.parse_args()

    got = ugot.UGOT()
    got.initialize(ROBOT_IP)

    if args.test_drive:
        print("Driving FORWARD slowly for 1 second...")
        move(got, 10, 0)
        time.sleep(1.0)
        got.mecanum_stop()
        print("Done. Did Alex drive FORWARD (camera side)? If not -> DRIVE_SIGN = -1")
        return

    got.open_camera()
    time.sleep(1.0)
    last_seen = 0.0
    dists = deque(maxlen=MEDIAN_N)
    print(f"Following Bob at {TARGET_CM:.0f} cm... press 'q' in the video window to stop.")

    try:
        while True:
            img = read_frame(got)
            if img is None:
                continue
            w = img.shape[1]
            found = find_marker(img)
            now = time.time()

            if found is not None:
                last_seen = now
                cx, h_px, pts = found
                dists.append(FOCAL_PX * MARKER_SIZE_CM / h_px)
                dist = float(np.median(dists))
                bearing = math.degrees(math.atan((cx - w / 2) / FOCAL_PX))

                # rotation
                z = 0.0 if abs(bearing) < DEADBAND_DEG else clamp(KP_TURN * bearing, MAX_TURN)

                # forward / back
                err = dist - TARGET_CM          # + = too far -> drive forward
                if dist < SAFETY_CM:
                    x = -MAX_DRIVE              # too close -> back off
                elif abs(err) < DEADBAND_CM or abs(bearing) > MAX_BEARING_TO_DRIVE:
                    x = 0.0
                else:
                    x = clamp(KP_DRIVE * err, MAX_DRIVE)

                if x == 0 and z == 0:
                    got.mecanum_stop()
                else:
                    move(got, x, z)

                text = f"dist {dist:5.1f} | bearing {bearing:+5.1f} | drive {x:+5.1f} | turn {z:+5.1f}"
                cv2.polylines(img, [pts.astype(int)], True, (0, 255, 0), 2)
                cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                print(text)
            else:
                if now - last_seen > LOST_TIMEOUT:
                    got.mecanum_stop()
                    dists.clear()
                cv2.putText(img, "no marker - stopped", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.line(img, (w // 2, 0), (w // 2, img.shape[0]), (255, 255, 0), 1)
            cv2.imshow("Alex - following Bob", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        got.mecanum_stop()
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()
