# bottle_detect.py
# Step 1 (+ dashboard): connect to Bob, detect a bottle with YOLOv8s, show
# everything on a local web dashboard, record video, and signal the state with
# Bob's LED + voice. Bob does NOT move in this step - no wheel commands at all.
#
# Run:   cd "C:\Users\user\Desktop\bottle pro"; py -3.8 bottle_detect.py
# Dashboard: http://127.0.0.1:5000  (video = MJPEG /stream, fallback = /frame.jpg)
# Opts:  --ip 172.20.10.12   --port 5000   --show (also open an OpenCV window)
#        --no-browser (don't open the dashboard automatically)
# Quit:  Ctrl+C in the terminal (or 'q' in the OpenCV window if --show)
#
# Only standard-library + the packages you already have (ugot, ultralytics,
# opencv-python, numpy). The web server is Python's built-in http.server.

import argparse
import json
import queue
import threading
import time
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np
from ugot import ugot
from ultralytics import YOLO

# ------------------------------------------------------------------ config ---
ROBOT_IP = "172.20.10.12"      # Bob (same as bottle_hunter.py)
SCRIPT_DIR = Path(__file__).resolve().parent
# Use the local weights file if it's next to this script (no internet needed)
YOLO_MODEL = str(SCRIPT_DIR / "yolov8s.pt") if (SCRIPT_DIR / "yolov8s.pt").exists() else "yolov8s.pt"
BOTTLE_CLASS_ID = 39           # "bottle" in the COCO dataset
CONFIDENCE = 0.4               # ignore detections weaker than this
TOP_CUT_PX = 5                 # box top within 5 px of the frame top -> "TOP CUT"
WARMUP_FRAMES = 10             # throw away first frames while auto-exposure settles

CENTER_TOLERANCE_PX = 40       # |center_x - frame center| <= 40 px -> CENTERED
STATE_STABLE_FRAMES = 3        # a new state must last 3 frames before LED/sound change
                               # (stops the LED flickering when a detection blinks)

RECORDINGS_DIR = SCRIPT_DIR / "recordings"
RECORD_FPS = 15                # fixed output frame rate of the MP4
DASHBOARD_FILE = SCRIPT_DIR / "dashboard.html"

# State name -> Bob LED color (r, g, b) and box color on the video (B, G, R)
STATE_LED = {
    "NO_BOTTLE": (255, 0, 0),           # red
    "BOTTLE_OFF_CENTER": (255, 255, 0), # yellow
    "CENTERED": (0, 255, 0),            # green
}
STATE_BOX_BGR = {
    "BOTTLE_OFF_CENTER": (0, 220, 255), # yellow
    "CENTERED": (0, 255, 0),            # green
}
CENTERED_SOUND = "Centered"     # short TTS word played when the bottle gets centered


# ----------------------------------------------------------- robot access ---
class Bob:
    """One ugot connection shared by the camera thread and the LED/sound thread.

    Same calls as bottle_hunter.py:
      initialize / open_camera / read_camera_data
      show_light_rgb_effect(r, g, b, 0)   -> steady LED color
      play_audio_tts(text, 0, False)      -> speak without waiting
    A lock makes sure two threads never talk to Bob at the same moment.
    """

    def __init__(self, ip):
        self.got = ugot.UGOT()
        self.got.initialize(ip)
        self.lock = threading.Lock()

    def open_camera(self):
        with self.lock:
            self.got.open_camera()

    def read_frame(self):
        # Bob sends a JPEG as raw bytes -> decode it into an OpenCV image
        with self.lock:
            data = self.got.read_camera_data()
        if not data:
            return None
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    def led(self, r, g, b):
        with self.lock:
            self.got.show_light_rgb_effect(r, g, b, 0)   # effect 0 = steady light

    def led_off(self):
        self.led(0, 0, 0)                                # black = LED off

    def say(self, text):
        with self.lock:
            self.got.play_audio_tts(text, 0, False)      # False = don't block

    def close_camera(self):
        # bottle_hunter.py never closes the camera; call it only if ugot has it
        fn = getattr(self.got, "close_camera", None)
        if fn is not None:
            with self.lock:
                fn()


class BobCamera:
    """Keeps the newest camera frame in a background thread, so slow YOLO
    inference never makes the video fall behind."""

    def __init__(self, bob):
        self.bob = bob
        self.bob.open_camera()
        self.frame = None              # newest decoded frame (BGR numpy array)
        self.frame_id = 0              # increases each time a new frame arrives
        self.lock = threading.Lock()   # protects frame / frame_id between threads
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        skipped = 0
        while self.running:
            try:
                frame = self.bob.read_frame()
            except Exception as e:     # network hiccup: wait a bit and retry
                print("camera read error:", e)
                time.sleep(0.1)
                continue
            if frame is None:
                time.sleep(0.01)
                continue
            if skipped < WARMUP_FRAMES:   # skip the first, often over-exposed frames
                skipped += 1
                continue
            with self.lock:
                self.frame = frame
                self.frame_id += 1

    def latest(self):
        # Return (frame_id, frame) of the newest frame, or (0, None) if none yet
        with self.lock:
            return self.frame_id, self.frame

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)


class Signals:
    """Changes Bob's LED (and plays the sound) in its own thread, so a slow
    robot call never freezes the video. It is only told about state CHANGES."""

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
            if state is None:              # shutdown signal
                return
            try:
                self.bob.led(*STATE_LED[state])
                if state == "CENTERED":
                    self.bob.say(CENTERED_SOUND)
            except Exception as e:
                print("LED/sound failed:", e)

    def stop(self):
        self.q.put(None)
        self.thread.join(timeout=2.0)


# --------------------------------------------------------------- detection ---
def find_bottle(model, frame):
    """Run YOLO on one frame; return the most confident bottle as
    (x1, y1, x2, y2, conf) in pixels, or None if no bottle is seen."""
    result = model(frame, classes=[BOTTLE_CLASS_ID], conf=CONFIDENCE, verbose=False)[0]
    if len(result.boxes) == 0:
        return None
    best = int(result.boxes.conf.argmax())                 # index of highest confidence
    x1, y1, x2, y2 = result.boxes.xyxy[best].tolist()      # box corners
    conf = float(result.boxes.conf[best])
    return int(x1), int(y1), int(x2), int(y2), conf


def measure(det, frame_w):
    """Turn a detection into the numbers we care about + the raw state."""
    if det is None:
        return {"state": "NO_BOTTLE"}
    x1, y1, x2, y2, conf = det
    cx = (x1 + x2) // 2                    # horizontal center of the bottle (px)
    offset = cx - frame_w // 2             # + = bottle right of center
    state = "CENTERED" if abs(offset) <= CENTER_TOLERANCE_PX else "BOTTLE_OFF_CENTER"
    return {
        "state": state, "box": (x1, y1, x2, y2),
        "cx": cx, "offset": offset,
        "w": x2 - x1, "h": y2 - y1,        # box width / height (px)
        "conf": conf,
        "top_cut": y1 <= TOP_CUT_PX,       # top of the box touches the top edge
    }


def annotate(frame, m):
    """Draw the center zone, the box and the numbers onto the frame."""
    fh, fw = frame.shape[:2]
    mid = fw // 2
    # Faint center line + the +/-40 px "centered" zone
    cv2.line(frame, (mid, 0), (mid, fh), (200, 200, 200), 1)
    for x in (mid - CENTER_TOLERANCE_PX, mid + CENTER_TOLERANCE_PX):
        cv2.line(frame, (x, 0), (x, fh), (120, 120, 120), 1)

    if m["state"] == "NO_BOTTLE":
        cv2.putText(frame, "no bottle", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return

    x1, y1, x2, y2 = m["box"]
    # Orange if the top is cut, otherwise green (centered) / yellow (off center)
    color = (0, 165, 255) if m["top_cut"] else STATE_BOX_BGR[m["state"]]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.line(frame, (m["cx"], y1), (m["cx"], y2), color, 1)
    cv2.putText(frame, f"bottle {m['conf']:.2f}  cx={m['cx']} w={m['w']} h={m['h']}",
                (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    if m["top_cut"]:
        cv2.putText(frame, "TOP CUT", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


def report(m):
    """Terminal output (same numbers as before)."""
    if m["state"] == "NO_BOTTLE":
        return
    msg = (f"center_x={m['cx']:4d} px  width={m['w']:4d} px  height={m['h']:4d} px  "
           f"conf={m['conf']:.2f}")
    if m["top_cut"]:
        msg += "  WARNING: TOP CUT"
    print(msg)


# --------------------------------------------------------------- recording ---
class Recorder:
    """Writes annotated frames to recordings/bottle_YYYY-MM-DD_HH-MM-SS.mp4.

    Detection runs at an uneven rate, so frames are written on a fixed
    RECORD_FPS clock (repeating the last frame when needed). That way the
    video plays back at real speed."""

    def __init__(self, folder, fps):
        self.folder = Path(folder)
        self.fps = fps
        self.lock = threading.Lock()
        self.writer = None
        self.active = False
        self.path = None
        self.start_time = 0.0
        self.count = 0
        self.last_saved = None     # file name of the last finished recording

    def start(self):
        with self.lock:
            if self.active:
                return
            self.folder.mkdir(exist_ok=True)
            name = datetime.now().strftime("bottle_%Y-%m-%d_%H-%M-%S.mp4")
            self.path = self.folder / name
            self.writer = None          # opened on the first frame (needs its size)
            self.count = 0
            self.start_time = time.time()
            self.active = True
            print("recording started:", self.path)

    def stop(self):
        with self.lock:
            if not self.active:
                return
            self.active = False
            if self.writer is not None:
                self.writer.release()
                self.writer = None
                self.last_saved = self.path.name
                print("recording saved:", self.path)
            else:
                print("recording stopped (no frames were written)")

    def write(self, frame):
        with self.lock:
            if not self.active:
                return
            if self.writer is None:
                h, w = frame.shape[:2]
                self.writer = cv2.VideoWriter(str(self.path),
                                              cv2.VideoWriter_fourcc(*"mp4v"),
                                              self.fps, (w, h))
                if not self.writer.isOpened():
                    print("ERROR: could not open video file", self.path)
                    self.writer = None
                    self.active = False
                    return
            # How many frames the video should hold by now at RECORD_FPS
            target = int((time.time() - self.start_time) * self.fps) + 1
            for _ in range(min(target - self.count, self.fps * 2)):   # cap catch-up
                self.writer.write(frame)
                self.count += 1
            self.count = max(self.count, target)

    def status(self):
        with self.lock:
            return {
                "recording": self.active,
                "elapsed": round(time.time() - self.start_time, 1) if self.active else 0,
                "file": self.path.name if self.active and self.path else None,
                "last_saved": self.last_saved,
            }


# --------------------------------------------------------------- dashboard ---
class Hub:
    """Shared data between the detection loop and the web server threads."""

    def __init__(self):
        self.cond = threading.Condition()
        self.jpeg = None           # newest annotated frame as JPEG bytes
        self.jpeg_id = 0
        self.status = {"state": None}
        self.stopping = False

    def publish(self, frame, status):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with self.cond:
            if ok:
                self.jpeg = buf.tobytes()
                self.jpeg_id += 1
            self.status = status
            self.cond.notify_all()      # wake up every waiting video stream

    def shutdown(self):
        with self.cond:
            self.stopping = True
            self.cond.notify_all()


def make_handler(hub, recorder):
    """Build the HTTP request handler (routes) for the dashboard."""
    frame_clients = set()          # browsers that already polled /frame.jpg (print once each)
    frame_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):   # keep the terminal quiet
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _route(self):
            # Path WITHOUT the "?t=123" cache-buster (the old bug: "/stream?t=..."
            # never matched "/stream", so the browser got a 404 instead of video)
            return urlsplit(self.path).path

        def do_GET(self):
            route = self._route()
            if route == "/":
                # Read the page each time, so editing dashboard.html needs no restart
                self._send(200, DASHBOARD_FILE.read_bytes(), "text/html; charset=utf-8")
            elif route == "/status":
                with hub.cond:
                    data = dict(hub.status)
                data.update(recorder.status())
                self._send(200, json.dumps(data).encode(), "application/json")
            elif route == "/stream":
                self._stream()
            elif route == "/frame.jpg":
                self._frame()
            else:
                print(f"[web] 404 for {self.path}")
                self._send(404, b"not found", "text/plain")

        def _frame(self):
            # Fallback: one single JPEG of the newest annotated frame
            client = self.client_address[0]
            with frame_lock:
                first = client not in frame_clients
                frame_clients.add(client)
            if first:
                print(f"[web] {client} started polling /frame.jpg (fallback mode)")
            with hub.cond:
                jpeg = hub.jpeg
            if jpeg is None:
                self._send(503, b"no frame yet", "text/plain")
            else:
                self._send(200, jpeg, "image/jpeg")

        def do_POST(self):
            route = self._route()
            if route == "/record/start":
                recorder.start()
            elif route == "/record/stop":
                recorder.stop()
            else:
                self._send(404, b"not found", "text/plain")
                return
            self._send(200, json.dumps(recorder.status()).encode(), "application/json")

        def _stream(self):
            # MJPEG: one endless HTTP response, each part is a new JPEG frame.
            # The browser shows it in a plain <img> tag.
            client = self.client_address[0]
            print(f"[web] {client} connected to /stream (MJPEG)")
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last = 0
            sent = 0
            try:
                while True:
                    with hub.cond:
                        hub.cond.wait_for(lambda: hub.jpeg_id != last or hub.stopping,
                                          timeout=1.0)
                        if hub.stopping:
                            return
                        if hub.jpeg_id == last:
                            continue
                        jpeg, last = hub.jpeg, hub.jpeg_id
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                     + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                                     + jpeg + b"\r\n")
                    self.wfile.flush()  # push each frame out right away
                    sent += 1
                    if sent == 1:
                        print(f"[web] first MJPEG frame sent to {client}")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass                    # browser tab closed / switched to fallback
            print(f"[web] {client} left /stream after {sent} frames")

    return Handler


# -------------------------------------------------------------------- main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default=ROBOT_IP)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--show", action="store_true", help="also show an OpenCV window")
    parser.add_argument("--no-browser", action="store_true", help="don't open the dashboard")
    args = parser.parse_args()

    print("loading", YOLO_MODEL)
    model = YOLO(YOLO_MODEL)

    print("connecting to Bob at", args.ip)
    bob = Bob(args.ip)
    cam = BobCamera(bob)
    signals = Signals(bob)
    print("connected")
    # NOTE: no movement commands are sent anywhere in this file -> Bob stays still.

    hub = Hub()
    recorder = Recorder(RECORDINGS_DIR, RECORD_FPS)

    # Web server in a background thread. 0.0.0.0 = listen on every network
    # interface (so other devices on your Wi-Fi could open it too).
    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(hub, recorder))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{args.port}"
    print("dashboard:", url, " - press Ctrl+C here to quit")
    if not args.no_browser:
        webbrowser.open(url)

    current_state = None           # state Bob's LED is showing
    candidate, candidate_n = None, 0
    last_id = 0
    fps, t_prev = 0.0, time.time()

    try:
        while True:
            frame_id, frame = cam.latest()
            if frame is None or frame_id == last_id:
                # No new frame yet: keep the OpenCV window alive, otherwise just wait
                if args.show:
                    if cv2.waitKey(5) & 0xFF == ord("q"):
                        break
                else:
                    time.sleep(0.005)
                continue
            last_id = frame_id

            frame = frame.copy()                       # draw on a copy
            det = find_bottle(model, frame)            # YOLO bottle detection
            m = measure(det, frame.shape[1])           # numbers + raw state
            annotate(frame, m)                         # box, center zone, TOP CUT
            report(m)                                  # terminal print

            # --- state change with debounce: LED/sound only on a real change ---
            if m["state"] == candidate:
                candidate_n += 1
            else:
                candidate, candidate_n = m["state"], 1
            if candidate_n >= STATE_STABLE_FRAMES and candidate != current_state:
                current_state = candidate
                print(">>> state:", current_state)
                signals.set_state(current_state)

            # --- detection speed (smoothed) ---
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-3))
            t_prev = now

            recorder.write(frame)                      # no-op unless recording
            hub.publish(frame, {
                "state": current_state or m["state"],
                "cx": m.get("cx"), "offset": m.get("offset"),
                "w": m.get("w"), "h": m.get("h"),
                "conf": round(m["conf"], 2) if "conf" in m else None,
                "top_cut": m.get("top_cut", False),
                "frame_w": frame.shape[1], "frame_h": frame.shape[0],
                "tolerance": CENTER_TOLERANCE_PX,
                "fps": round(fps, 1),
            })

            if args.show:
                cv2.imshow("Bob camera - bottle detect", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\nCtrl+C - shutting down")
    finally:
        # Clean shutdown, each step guarded so one failure doesn't skip the rest
        recorder.stop()                                # finish the MP4 file
        signals.stop()                                 # stop LED/sound thread
        for step in (bob.led_off, cam.stop, bob.close_camera):
            try:
                step()
            except Exception as e:
                print("shutdown step failed:", e)
        hub.shutdown()                                 # end video streams
        server.shutdown()                              # stop web server
        if args.show:
            cv2.destroyAllWindows()
        print("bye")


if __name__ == "__main__":
    main()
