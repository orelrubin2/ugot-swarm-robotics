# geometry.py
# Pure math for the equidistant experiment - no robot, no camera, no cv2/ugot.
# Every physical constant is passed in as an argument; the values live in the
# config section of equidistant.py (so there is only one place to change them).
#
# Frame (all in cm / degrees):
#   origin = Bob's turning center at the start, x = "heading 0" axis (forward),
#   y = to the LEFT, angles + = LEFT (counter-clockwise).
#   beta (bottle angle in the image) is + when the bottle is RIGHT of center.
#
# Test:  py -3.11 -m pytest test_geometry.py -v

import math
from typing import List, NamedTuple, Optional, Tuple


class BottleMeasurement(NamedTuple):
    z_cm: float        # pinhole depth along the camera axis (to the bottle FRONT)
    beta_deg: float    # angle in the image, + = bottle right of center
    d_cm: float        # range from the camera to the bottle CENTER
    method: str        # "height" or "width"


class BottlePosition(NamedTuple):
    x: float           # cm along heading 0
    y: float           # cm, + = left
    phi_deg: float     # direction from heading 0, + = left


class EquidistantResult(NamedTuple):
    ok: bool
    s: Optional[float]     # target point (s, 0) on the axis
    r: Optional[float]     # predicted distance to both bottles from (s, 0)
    reason: str            # why it failed ("" if ok)


class FinalResult(NamedTuple):
    ratio: float           # d_max / d_min (1.0 = perfect)
    error_pct: float       # (ratio - 1) * 100
    r_error_cm: float      # mean(d1, d2) - predicted r


# ------------------------------------------------------------- one bottle ---
def measure_bottle(box_h, box_w, top_cut, offset_px, *,
                   focal_px, bottle_h_cm, bottle_d_cm, bottle_r_cm):
    """Box size + offset in the image -> depth, image angle, range to the center.

    Z    = focal * H / box_h          (if the top is cut: focal * D / box_w)
    beta = atan(offset / focal)       (+ = right of center)
    d    = Z / cos(beta) + R          (front surface -> bottle center)
    """
    if top_cut:
        size, real, method = box_w, bottle_d_cm, "width"
    else:
        size, real, method = box_h, bottle_h_cm, "height"
    if size <= 0:
        raise ValueError(f"box {method} must be > 0, got {size}")
    z = focal_px * real / size
    beta = math.atan(offset_px / focal_px)
    d = z / math.cos(beta) + bottle_r_cm
    return BottleMeasurement(z, math.degrees(beta), d, method)


def bottle_position(d_cm, beta_deg, theta_deg, cam_fwd_offset_cm=0.0):
    """Range + angles -> bottle center in the start frame.

    phi = theta - beta
    x = c*cos(theta) + d*cos(phi),  y = c*sin(theta) + d*sin(phi)
    (c = camera distance ahead of the turning center)
    """
    phi = math.radians(theta_deg - beta_deg)
    th = math.radians(theta_deg)
    x = cam_fwd_offset_cm * math.cos(th) + d_cm * math.cos(phi)
    y = cam_fwd_offset_cm * math.sin(th) + d_cm * math.sin(phi)
    return BottlePosition(x, y, math.degrees(phi))


# ------------------------------------------------------- the target point ---
def equidistant_point(p1, p2, *, min_dx_cm=10.0, max_travel_cm=150.0):
    """Point (s, 0) on the x axis that is equally far from both bottles.

    (x1-s)^2 + y1^2 = (x2-s)^2 + y2^2
      ->  s = (x1^2 + y1^2 - x2^2 - y2^2) / (2*(x1 - x2))
    Fails (ok=False) if |x1 - x2| < min_dx_cm (s jumps around / no solution)
    or |s| > max_travel_cm (too far to drive).
    """
    (x1, y1), (x2, y2) = p1[:2], p2[:2]
    dx = x1 - x2
    if abs(dx) < min_dx_cm:
        return EquidistantResult(False, None, None,
                                 f"|x1 - x2| = {abs(dx):.1f} cm < {min_dx_cm:.0f} cm: "
                                 f"no stable solution")
    s = (x1 ** 2 + y1 ** 2 - x2 ** 2 - y2 ** 2) / (2 * dx)
    r = math.hypot(x1 - s, y1)
    if abs(s) > max_travel_cm:
        return EquidistantResult(False, s, r,
                                 f"|s| = {abs(s):.0f} cm > {max_travel_cm:.0f} cm: too far")
    return EquidistantResult(True, s, r, "")


def distance_to_path(point, s):
    """Shortest distance from a bottle center to the driven segment (0,0)-(s,0)."""
    x, y = point[:2]
    lo, hi = min(0.0, s), max(0.0, s)
    nearest_x = min(max(x, lo), hi)
    return math.hypot(x - nearest_x, y)


def path_blocked(points, s, *, clearance_cm=20.0):
    """Reason string if any bottle is within clearance_cm of the path, else None."""
    for i, p in enumerate(points, start=1):
        dist = distance_to_path(p, s)
        if dist < clearance_cm:
            return (f"bottle B{i} is {dist:.0f} cm from the driving path "
                    f"(< {clearance_cm:.0f} cm) - would hit it")
    return None


# ------------------------------------------------------------- the result ---
def final_result(d1, d2, r):
    """Compare the two distances measured at (s, 0) and the prediction r."""
    if min(d1, d2) <= 0:
        raise ValueError("distances must be > 0")
    ratio = max(d1, d2) / min(d1, d2)
    return FinalResult(ratio, (ratio - 1.0) * 100.0, (d1 + d2) / 2.0 - r)


# ----------------------------------------------- consistency after driving ---
def to_start_frame(point, s):
    """Position measured at (s, 0) facing heading 0 -> position in the start frame."""
    return (point[0] + s, point[1])


def match_bottles(first: List[Tuple[float, float]], second: List[Tuple[float, float]]):
    """Pair two bottles from measurement 1 with two from measurement 2 (same frame)
    by the smaller total distance. Returns [(i_first, i_second, drift_cm), ...]."""
    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])
    straight = dist(first[0], second[0]) + dist(first[1], second[1])
    crossed = dist(first[0], second[1]) + dist(first[1], second[0])
    pairs = [(0, 0), (1, 1)] if straight <= crossed else [(0, 1), (1, 0)]
    return [(i, j, dist(first[i], second[j])) for i, j in pairs]
