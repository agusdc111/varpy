# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Literal
import numpy as np
import cv2

Axis = Literal["x", "y"]

# Dimensiones en metros
PENAL_W, PENAL_D = 40.32, 16.5
GOAL_W, GOAL_D   = 18.32, 5.5

@dataclass
class Calibration:
    template: str  # "penal","goal","generic","generic_scaled"
    H: np.ndarray
    Hinv: np.ndarray
    width_m: float
    depth_m: float
    img_quad: np.ndarray  # (4,2)
    world_quad: np.ndarray  # (4,2)
    axis_mode: Axis  # eje seleccionado (auto-detectado o forzado)

def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    v1 = v1.astype(np.float32); v2 = v2.astype(np.float32)
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0: return 0.0
    cosang = float(np.clip(np.dot(v1, v2) / (n1*n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))

def project_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts_h = cv2.convertPointsToHomogeneous(pts).reshape(-1,3).T
    out = H @ pts_h
    out = (out[:2] / out[2:3]).T
    return out

def _decide_axis(img_points: np.ndarray, Hinv: np.ndarray, W: float, D: float) -> Axis:
    # Generamos una recta y=const y la comparamos contra el borde 3->2 (línea de gol)
    y0 = D * 0.3
    world = np.array([[ -50.0, y0],[ W+50.0, y0 ]], dtype=np.float32)
    test_line = project_points(Hinv, world)
    goal_vec_img = img_points[2] - img_points[3]  # cerca-der -> cerca-izq (arista de gol)
    cand_vec = test_line[1] - test_line[0]
    ang = _angle_between(cand_vec, goal_vec_img)
    return "y" if abs(ang) < 45.0 or abs(ang-180.0) < 45.0 else "x"

def _world_quad(template: str, depth_override: Optional[float]=None):
    if template == "penal":
        W,D = PENAL_W, PENAL_D
    elif template == "goal":
        W,D = GOAL_W, GOAL_D
    elif template == "generic_scaled":
        W,D = 1.0, float(depth_override if depth_override else 1.0)
    else:
        W,D = 1.0, 1.0
    world = np.array([[0.0, D],
                      [W,   D],
                      [W,   0.0],
                      [0.0, 0.0]], dtype=np.float32)
    return world, W, D

def compute_homography(img_points: np.ndarray, template: str,
                       depth_override: Optional[float]=None,
                       axis_override: Optional[Axis]=None) -> Calibration:
    assert img_points.shape == (4,2)
    world_quad, W, D = _world_quad(template, depth_override)
    H, status = cv2.findHomography(img_points, world_quad, method=0)
    if H is None: raise ValueError("No se pudo estimar la homografía.")
    Hinv = np.linalg.inv(H)
    axis_mode = axis_override if axis_override in ("x","y") else _decide_axis(img_points, Hinv, W, D)
    return Calibration(template, H, Hinv, W, D, img_points.copy(), world_quad, axis_mode)

def line_image(Hinv: np.ndarray, W: float, D: float, ref: float, axis: Axis) -> np.ndarray:
    margin = 50.0
    if axis == "y":
        world = np.array([[-margin, ref],[W+margin, ref]], dtype=np.float32)
    else:
        world = np.array([[ref, -margin],[ref, D+margin]], dtype=np.float32)
    return project_points(Hinv, world)

def offside_verdict(cal: Calibration,
                    attacker_img: Tuple[float,float],
                    defender2_img: Tuple[float,float],
                    ball_img: Optional[Tuple[float,float]] = None,
                    tolerance_m: float = 0.05,
                    axis_override: Optional[Axis]=None) -> Dict:
    axis = axis_override if axis_override in ("x","y") else cal.axis_mode
    a_w = project_points(cal.H, np.array([attacker_img], dtype=np.float32))[0]
    d2_w = project_points(cal.H, np.array([defender2_img], dtype=np.float32))[0]
    b_w = None if ball_img is None else project_points(cal.H, np.array([ball_img], dtype=np.float32))[0]

    if axis == "y":
        A = float(a_w[1]); D2 = float(d2_w[1]); B = None if b_w is None else float(b_w[1])
    else:
        A = float(a_w[0]); D2 = float(d2_w[0]); B = None if b_w is None else float(b_w[0])

    ref = D2 if B is None else min(D2, B)
    delta = ref - A
    tol = float(tolerance_m)

    if abs(A - ref) <= tol:
        verdict = "EMPATE (misma línea)"
    elif A < ref - tol:
        verdict = "OFFSIDE"
    else:
        verdict = "HABILITADO"

    return {
        "axis": axis,
        "attacker_world": (float(a_w[0]), float(a_w[1])),
        "defender2_world": (float(d2_w[0]), float(d2_w[1])),
        "ball_world": None if b_w is None else (float(b_w[0]), float(b_w[1])),
        "reference_value": float(ref),
        "attacker_value": float(A),
        "delta": float(delta),
        "tolerance": tol,
        "verdict": verdict,
    }

def build_offside_lines(cal: Calibration, verdict: Dict, axis_override: Optional[Axis]=None):
    axis = axis_override if axis_override in ("x","y") else verdict["axis"]
    ref = verdict["reference_value"]
    av  = verdict["attacker_value"]
    Lref = line_image(cal.Hinv, cal.width_m, cal.depth_m, ref, axis)
    Latt = line_image(cal.Hinv, cal.width_m, cal.depth_m, av, axis)
    return {"defender_line": Lref, "attacker_line": Latt}
