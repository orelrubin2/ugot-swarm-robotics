"""
track_rotate.py - Alex rotates IN PLACE to keep Bob's ArUco marker centered.
Alex does NOT drive forward/back in this script - rotation only.

Usage:
  1) Direction test (once):   python track_rotate.py --test-turn
     Alex turns slowly for 1 second. It should turn to the RIGHT (clockwise, seen from above).
     If it turns LEFT -> change TURN_SIGN below from 1 to -1.
  2) Tracking:                python track_rotate.py
     Move Bob slowly left/right in front of Alex. Alex should turn to follow him.
     Press 'q' in the video window to stop (Alex stops immediately).
"""
import argparse
import math
import time

import cv2
import numpy as np
from ugot import ugot

# ---------------- settings ----------------
ROBOT_IP = "172.20.10.10"   # Alex (the robot with the camera)
MARKER_SIZE_CM = 10.0
FOCAL_PX = 428.0
MARKER_ID = 0

TURN_SIGN = -1        # set to -1 if --test-turn turns LEFT
KP = 1.0             # deg/s of rotation per 1 deg of bearing error
MAX_TURN = 30        # max rotation speed [deg/s] - keep slow for the first runs
DEADBAND_DEG = 3.0   # if |bearing| is smaller than this -> don't turn
LOST_TIMEOUT = 0.5   # seconds without marker -> stop
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


def turn(got, z_deg_s):
    """Rotate in place. Positive z = turn RIGHT (after TURN_SIGN is set correctly)."""
    z = int(round(TURN_SIGN * z_deg_s))
    got.mecanum_move_xyz(0, 0, z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-turn", action="store_true")
    args = ap.parse_args()

    got = ugot.UGOT()
    got.initialize(ROBOT_IP)

    if args.test_turn:
        print("Turning RIGHT slowly for 1 second...")
        turn(got, 20)
        time.sleep(1.0)
        got.mecanum_stop()
        print("Done. Did Alex turn RIGHT (clockwise)? If not -> TURN_SIGN = -1")
        return

    got.open_camera()
    time.sleep(1.0)
    last_seen = 0.0
    print("Tracking Bob... press 'q' in the video window to stop.")

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
                dist_cm = FOCAL_PX * MARKER_SIZE_CM / h_px
                bearing = math.degrees(math.atan((cx - w / 2) / FOCAL_PX))  # + = Bob is to the RIGHT

                if abs(bearing) < DEADBAND_DEG:
                    z = 0
                    got.mecanum_stop()
                else:
                    z = max(-MAX_TURN, min(MAX_TURN, KP * bearing))
                    turn(got, z)

                text = f"dist {dist_cm:5.1f} | bearing {bearing:+5.1f} | turn {z:+5.1f}"
                cv2.polylines(img, [pts.astype(int)], True, (0, 255, 0), 2)
                cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                print(text)
            else:
                if now - last_seen > LOST_TIMEOUT:
                    got.mecanum_stop()
                cv2.putText(img, "no marker - stopped", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.line(img, (w // 2, 0), (w // 2, img.shape[0]), (255, 255, 0), 1)
            cv2.imshow("Alex - tracking Bob", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        got.mecanum_stop()
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()
