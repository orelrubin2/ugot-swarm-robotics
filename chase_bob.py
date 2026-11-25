"""
chase_bob.py - You drive Bob with the keyboard arrows, Alex follows Bob automatically.
Both cameras are shown side by side: LEFT = Alex, RIGHT = Bob.

Needs follow_bob.py in the same folder (uses its settings and functions).

Keys (click on the video window first so it has focus):
  Up    -> Bob drives forward
  Down  -> Bob drives backward
  Left  -> Bob turns left
  Right -> Bob turns right
  SPACE -> Bob stops
  q     -> everything stops and the program exits

Bob keeps doing the last command until you press another arrow or SPACE.
"""
import math
import time
from collections import deque

import cv2
import numpy as np
from ugot import ugot

import follow_bob as fb   # Alex's settings + find_marker / move / clamp

# ---------------- settings ----------------
ALEX_IP = fb.ROBOT_IP
BOB_IP = "172.20.10.12"   # <-- Bob's IP (must be DIFFERENT from Alex's)

BOB_DRIVE = 8             # cm/s  - slower than Alex's MAX_DRIVE (15) so Alex can keep up
BOB_TURN = 20             # deg/s - slower than Alex's MAX_TURN (30)
VIEW_H = 360              # height of each camera view on screen
# ------------------------------------------

# Arrow key codes from cv2.waitKeyEx on Windows
KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 2490368, 2621440, 2424832, 2555904
KEY_SPACE, KEY_Q = 32, ord("q")


def read_frame(got):
    data = got.read_camera_data()
    if data is None:
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def fit(img, h):
    if img is None:
        return np.zeros((h, int(h * 4 / 3), 3), np.uint8)
    return cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))


def main():
    if BOB_IP == ALEX_IP or "XX" in BOB_IP:
        print("Set BOB_IP to Bob's real IP (it must be different from Alex's). Exiting.")
        return

    alex = ugot.UGOT(); alex.initialize(ALEX_IP); alex.open_camera()
    bob = ugot.UGOT();  bob.initialize(BOB_IP);   bob.open_camera()
    time.sleep(1.0)

    bob_cmd = (0, 0)          # (drive cm/s, turn deg/s)
    bob_label = "STOP"
    last_seen = 0.0
    dists = deque(maxlen=fb.MEDIAN_N)
    print("Click the video window, then drive Bob with the arrows. SPACE = stop Bob, q = quit.")

    try:
        while True:
            a_img = read_frame(alex)
            b_img = read_frame(bob)
            now = time.time()

            # ---------- Alex: follow Bob (same logic as follow_bob.py) ----------
            status = "no marker - stopped"
            if a_img is not None:
                w = a_img.shape[1]
                found = fb.find_marker(a_img)
                if found is not None:
                    last_seen = now
                    cx, h_px, pts = found
                    dists.append(fb.FOCAL_PX * fb.MARKER_SIZE_CM / h_px)
                    dist = float(np.median(dists))
                    bearing = math.degrees(math.atan((cx - w / 2) / fb.FOCAL_PX))

                    z = 0.0 if abs(bearing) < fb.DEADBAND_DEG else fb.clamp(fb.KP_TURN * bearing, fb.MAX_TURN)
                    err = dist - fb.TARGET_CM
                    if dist < fb.SAFETY_CM:
                        x = -fb.MAX_DRIVE
                    elif abs(err) < fb.DEADBAND_CM or abs(bearing) > fb.MAX_BEARING_TO_DRIVE:
                        x = 0.0
                    else:
                        x = fb.clamp(fb.KP_DRIVE * err, fb.MAX_DRIVE)

                    if x == 0 and z == 0:
                        alex.mecanum_stop()
                    else:
                        fb.move(alex, x, z)
                    status = f"dist {dist:5.1f} | bear {bearing:+5.1f} | drive {x:+5.1f} | turn {z:+5.1f}"
                    cv2.polylines(a_img, [pts.astype(int)], True, (0, 255, 0), 2)
                    print(f"ALEX {status} || BOB {bob_label}")
                elif now - last_seen > fb.LOST_TIMEOUT:
                    alex.mecanum_stop()
                    dists.clear()

            # ---------- keyboard -> Bob ----------
            key = cv2.waitKeyEx(1)
            new = None
            if key == KEY_UP:
                new, bob_label = (BOB_DRIVE, 0), "FORWARD"
            elif key == KEY_DOWN:
                new, bob_label = (-BOB_DRIVE, 0), "BACK"
            elif key == KEY_LEFT:
                new, bob_label = (0, -BOB_TURN), "TURN LEFT"
            elif key == KEY_RIGHT:
                new, bob_label = (0, BOB_TURN), "TURN RIGHT"
            elif key == KEY_SPACE:
                new, bob_label = (0, 0), "STOP"
            elif key == KEY_Q:
                break
            if new is not None and new != bob_cmd:
                bob_cmd = new
                if bob_cmd == (0, 0):
                    bob.mecanum_stop()
                else:
                    fb.move(bob, *bob_cmd)   # Bob is the same robot type -> same signs as Alex

            # ---------- display ----------
            left = fit(a_img, VIEW_H)
            right = fit(b_img, VIEW_H)
            color = (0, 255, 0) if status[0] == "d" else (0, 0, 255)
            cv2.putText(left, "ALEX (follower)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(left, status, (10, VIEW_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            cv2.putText(right, f"BOB (you drive): {bob_label}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow("Alex chases Bob  |  arrows = drive Bob, SPACE = stop, q = quit", np.hstack([left, right]))
    finally:
        for r in (alex, bob):
            try:
                r.mecanum_stop()
            except Exception:
                pass
        cv2.destroyAllWindows()
        print("Stopped both robots.")


if __name__ == "__main__":
    main()
