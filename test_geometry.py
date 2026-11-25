# test_geometry.py - hand-checked cases for geometry.py
# Run:  py -3.11 -m pytest test_geometry.py -v

import math

import pytest

import geometry as g

# Same values as the experiment config
CAM = dict(focal_px=480.0, bottle_h_cm=32.0, bottle_d_cm=8.0, bottle_r_cm=4.0)


# ------------------------------------------------------------ measure_bottle ---
def test_straight_ahead_from_height():
    # Z = 480*32/160 = 96, beta = 0, d = 96 + 4 = 100
    m = g.measure_bottle(160, 50, False, 0, **CAM)
    assert m.method == "height"
    assert m.z_cm == pytest.approx(96.0)
    assert m.beta_deg == pytest.approx(0.0)
    assert m.d_cm == pytest.approx(100.0)


def test_offset_45_degrees():
    # offset = focal -> beta = 45 deg; d = 96/cos45 + 4 = 135.765 + 4
    m = g.measure_bottle(160, 50, False, 480, **CAM)
    assert m.beta_deg == pytest.approx(45.0)
    assert m.d_cm == pytest.approx(96 * math.sqrt(2) + 4)


def test_top_cut_uses_width():
    # Z = 480*8/48 = 80, d = 84 (height is ignored)
    m = g.measure_bottle(999, 48, True, 0, **CAM)
    assert m.method == "width"
    assert m.d_cm == pytest.approx(84.0)


def test_zero_box_is_rejected():
    with pytest.raises(ValueError):
        g.measure_bottle(0, 50, False, 0, **CAM)


# ----------------------------------------------------------- bottle_position ---
def test_position_heading_zero():
    p = g.bottle_position(100, 0, 0)
    assert (p.x, p.y) == (pytest.approx(100), pytest.approx(0))


def test_position_bottle_right_is_negative_y():
    # beta +45 (right), theta 0 -> phi -45 -> y < 0
    p = g.bottle_position(100, 45, 0)
    assert p.phi_deg == pytest.approx(-45)
    assert p.x == pytest.approx(70.711, abs=1e-3)
    assert p.y == pytest.approx(-70.711, abs=1e-3)


def test_position_turned_left_90_with_camera_offset():
    # theta 90 (facing +y), camera 10 cm ahead -> bottle at (0, 10 + 100)
    p = g.bottle_position(100, 0, 90, cam_fwd_offset_cm=10)
    assert p.x == pytest.approx(0, abs=1e-9)
    assert p.y == pytest.approx(110)


# --------------------------------------------------------- equidistant_point ---
def test_equidistant_hand_case():
    # s = (100^2+50^2 - 60^2-30^2) / (2*40) = 8000/80 = 100,  r = 50
    res = g.equidistant_point((100, 50), (60, -30))
    assert res.ok
    assert res.s == pytest.approx(100)
    assert res.r == pytest.approx(50)
    # check: from (100,0) bottle 2 is sqrt(40^2+30^2) = 50 away too
    assert math.hypot(60 - 100, -30) == pytest.approx(res.r)


def test_symmetric_case_s_is_zero():
    # mirror images around the y axis -> Bob is already at the right point
    res = g.equidistant_point((50, 40), (-50, 40))
    assert res.ok
    assert res.s == pytest.approx(0)
    assert res.r == pytest.approx(math.hypot(50, 40))


def test_same_x_has_no_stable_solution():
    res = g.equidistant_point((80, 30), (80, -30))
    assert not res.ok
    assert res.s is None
    assert "no stable solution" in res.reason


def test_too_far_is_rejected():
    # s = (30^2+200^2 - 10^2-50^2) / (2*20) = 38300/40 = 957.5
    res = g.equidistant_point((30, 200), (10, -50))
    assert not res.ok
    assert res.s == pytest.approx(957.5)
    assert "too far" in res.reason


@pytest.mark.parametrize("p1,p2", [
    ((120, 40), (40, -60)),
    ((-30, 80), (90, 20)),
    ((70, -15), (20, 90)),
])
def test_both_distances_equal_r(p1, p2):
    res = g.equidistant_point(p1, p2, max_travel_cm=1e9)
    assert res.ok
    assert math.hypot(p1[0] - res.s, p1[1]) == pytest.approx(res.r)
    assert math.hypot(p2[0] - res.s, p2[1]) == pytest.approx(res.r)


# -------------------------------------------------------------- path_blocked ---
@pytest.mark.parametrize("bottle,s,blocked", [
    ((50, 10), 100, True),     # on the path
    ((50, 30), 100, False),    # 30 cm to the side
    ((110, 0), 100, True),     # 10 cm past the end point
    ((-30, 5), 100, False),    # 30 cm behind the start
    ((-50, 10), -80, True),    # driving backward into it
])
def test_path_blocked(bottle, s, blocked):
    assert (g.path_blocked([bottle], s) is not None) == blocked


# -------------------------------------------------------------- final_result ---
def test_final_result():
    res = g.final_result(52, 50, 50)
    assert res.ratio == pytest.approx(1.04)
    assert res.error_pct == pytest.approx(4.0)
    assert res.r_error_cm == pytest.approx(1.0)
    assert g.final_result(50, 52, 50).ratio == pytest.approx(1.04)   # order does not matter


# ------------------------------------------------------ consistency checking ---
def test_to_start_frame_and_matching():
    first = [(100, 50), (60, -30)]
    # measured at s = 100: bottles appear at (0, 50) and (-40, -30), order swapped
    second = [g.to_start_frame(p, 100) for p in [(-40, -30), (0, 50)]]
    pairs = g.match_bottles(first, second)
    assert [(i, j) for i, j, _ in pairs] == [(0, 1), (1, 0)]
    assert all(drift == pytest.approx(0) for _, _, drift in pairs)


# ------------------------------------------------- full round trip (synthetic) ---
def fake_camera(bottle_xy, theta_deg, cam_off):
    """Inverse model: where would Bob see a bottle at bottle_xy? -> (box_h, offset_px)."""
    th = math.radians(theta_deg)
    cx, cy = cam_off * math.cos(th), cam_off * math.sin(th)
    vx, vy = bottle_xy[0] - cx, bottle_xy[1] - cy
    phi = math.degrees(math.atan2(vy, vx))
    beta = math.radians(theta_deg - phi)
    z = (math.hypot(vx, vy) - CAM["bottle_r_cm"]) * math.cos(beta)
    return CAM["focal_px"] * CAM["bottle_h_cm"] / z, CAM["focal_px"] * math.tan(beta)


@pytest.mark.parametrize("theta,cam_off", [(0, 0), (20, 0), (-15, 8), (35, 12)])
def test_round_trip(theta, cam_off):
    truth = [(110, 45), (70, -25)]
    found = []
    for b in truth:
        box_h, off = fake_camera(b, theta, cam_off)
        m = g.measure_bottle(box_h, 50, False, off, **CAM)
        p = g.bottle_position(m.d_cm, m.beta_deg, theta, cam_off)
        found.append((p.x, p.y))
    for (x, y), (tx, ty) in zip(found, truth):
        assert x == pytest.approx(tx) and y == pytest.approx(ty)
    res = g.equidistant_point(*found)
    assert res.ok
    assert math.hypot(truth[0][0] - res.s, truth[0][1]) == pytest.approx(res.r)
