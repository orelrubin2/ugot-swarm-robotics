"""
aruco_check.py - Bob sees the ArUco marker on another robot and prints distance + bearing.

Usage:
  1) Calibration (once):  put the marker EXACTLY 50 cm in front of Bob's camera, then run
         python aruco_check.py --calibrate 50
     It prints a number F (focal length in pixels). Copy it into FOCAL_PX below.
  2) Normal run:
         python aruco_check.py
     Press 'q' in the video window to quit.

Bob does not move in this script - camera only.
"""
import argparse
import math
import time

import cv2
import numpy as np
from ugot import ugot

# ---------------- settings ----------------
BOB_IP = "172.20.10.10"      # <-- Bob's IP on the hotspot (same one you used in the bottle scripts)
MARKER_SIZE_CM = 10.0       # edge of the black square, as you measured after printing
FOCAL_PX = 442.7            # <-- replace with the value from --calibrate
MARKER_ID = 0               # all robots carry ID 0 (anonymous swarm)
# ------------------------------------------

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
DETECTOR = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters())


def read_frame(got):
    """Read one frame from the UGOT camera and decode it to an OpenCV image."""
    data = got.read_camera_data()
    if data is None:
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def marker_height_px(corners):
    """Average length of the two VERTICAL edges.
    Vertical edges don't shrink when the marker is turned sideways (yaw),
    so they give a more stable distance than the width."""
    tl, tr, br, bl = corners
    left = np.linalg.norm(bl - tl)
    right = np.linalg.norm(br - tr)
    return (left + right) / 2.0


def find_marker(img):
    """Returns (center_x, height_px, corners) of the biggest ID-0 marker, or None."""
    corners_list, ids, _ = DETECTOR.detectMarkers(img)
    if ids is None:
        return None
    best = None
    for c, i in zip(corners_list, ids.flatten()):
        if i != MARKER_ID:
            continue
        pts = c.reshape(4, 2)
        h = marker_height_px(pts)
        if best is None or h > best[1]:
            best = (pts[:, 0].mean(), h, pts)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", type=float, default=None,
                    help="known distance in cm from camera to marker (e.g. 50)")
    args = ap.parse_args()

    got = ugot.UGOT()
    got.initialize(BOB_IP)
    got.open_camera()
    time.sleep(1.0)

    calib_samples = []
    print("Running... press 'q' in the video window to quit.")

    while True:
        img = read_frame(got)
        if img is None:
            continue
        w = img.shape[1]
        found = find_marker(img)

        if found is not None:
            cx, h_px, pts = found
            cv2.polylines(img, [pts.astype(int)], True, (0, 255, 0), 2)

            if args.calibrate is not None:
                calib_samples.append(h_px * args.calibrate / MARKER_SIZE_CM)
                text = f"calibrating {len(calib_samples)}/30"
                if len(calib_samples) >= 30:
                    f = float(np.median(calib_samples))
                    print(f"\nFOCAL_PX = {f:.1f}   <-- copy this into the script\n")
                    break
            else:
                dist_cm = FOCAL_PX * MARKER_SIZE_CM / h_px
                bearing_deg = math.degrees(math.atan((cx - w / 2) / FOCAL_PX))  # + = marker is to the RIGHT
                text = f"dist {dist_cm:5.1f} cm | bearing {bearing_deg:+5.1f} deg"
                print(text)
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(img, "no marker", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.line(img, (w // 2, 0), (w // 2, img.shape[0]), (255, 255, 0), 1)  # image center line
        cv2.imshow("Bob - ArUco check", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
