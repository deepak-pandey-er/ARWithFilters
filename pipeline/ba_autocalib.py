"""
Bundle-adjustment based markerless autocalibration.

This module:
- Builds initial pairwise poses and triangulated 3D points (using an initial K guess).
- Constructs observations across frames.
- Runs a joint optimization (camera poses, 3D points, and intrinsics) with a robust loss.

Notes/limitations:
- This is a Python-based BA using scipy.optimize.least_squares. For production quality and
  performance, a sparse solver (Ceres / g2o) would be preferable. This implementation uses
  a dense parameter vector that can get large for many frames/points.
- It estimates fx, fy, cx, cy (optionally constraining fx==fy).
- It does not currently estimate distortion coefficients; that can be added but increases
  sensitivity and may require better priors or regularization.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
import cv2
from scipy.optimize import least_squares

from pipeline.autocalib import (
    default_initial_K,
    detect_and_match,
    recover_pairwise_pose,
    triangulate_points,
    build_observations,
    pack_observations,
)

# -----------------------
# Helper projection code
# -----------------------
def project_point(X: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Project 3D point(s) X (shape (...,3)) into image using Rodrigues rvec, tvec and K.
    Returns 2D point (u, v).
    """
    R, _ = cv2.Rodrigues(rvec)
    X_cam = (R @ X.reshape(3, 1)).ravel() + tvec.ravel()
    if X_cam[2] <= 1e-8:
        # behind the camera -> large residuals will handle this
        return np.array([1e6, 1e6], dtype=float)
    x = X_cam[0] / X_cam[2]
    y = X_cam[1] / X_cam[2]
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    return np.array([u, v], dtype=float)


# -----------------------
# Parameter pack/unpack
# -----------------------
def pack_params(cam_rvecs: np.ndarray, cam_tvecs: np.ndarray, points3d: np.ndarray, K: np.ndarray, fxfy_equal: bool):
    """
    Pack into 1D parameter vector:
      [cam0_r(3), cam0_t(3), cam1_r(3), cam1_t(3), ..., points3d (3*M), intrinsics (3 or 4)]
    """
    cams = np.hstack([cam_rvecs.ravel(), cam_tvecs.ravel()])
    pts = points3d.ravel()
    if fxfy_equal:
        f = 0.5 * (K[0, 0] + K[1, 1])
        cx = K[0, 2]
        cy = K[1, 2]
        intr = np.array([f, cx, cy], dtype=float)
    else:
        intr = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=float)
    return np.hstack([cams, pts, intr])


def unpack_params(x: np.ndarray, n_cams: int, n_points: int, fxfy_equal: bool):
    cam_params_len = n_cams * 6
    cams = x[:cam_params_len]
    pts = x[cam_params_len:cam_params_len + n_points * 3]
    intr = x[cam_params_len + n_points * 3:]
    cam_rvecs = cams.reshape(n_cams, 6)[:, :3]
    cam_tvecs = cams.reshape(n_cams, 6)[:, 3:6]
    points3d = pts.reshape(n_points, 3)
    if fxfy_equal:
        f, cx, cy = intr
        K = np.array([[f, 0, cx],
                      [0, f, cy],
                      [0, 0, 1]], dtype=float)
    else:
        fx, fy, cx, cy = intr
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=float)
    return cam_rvecs, cam_tvecs, points3d, K


# -----------------------
# Residual function
# -----------------------
def reprojection_residuals_ba(x: np.ndarray,
                              n_cams: int,
                              n_points: int,
                              observations_per_cam: List[Dict[int, np.ndarray]],
                              obs_list: List[Tuple[int, int, np.ndarray]],
                              fxfy_equal: bool):
    """
    x: packed parameters
    observations_per_cam: list of dicts mapping point_idx -> observed uv for each camera
    obs_list: list of tuples (cam_idx, pidx, uv) to iterate in a deterministic order
    """
    cam_rvecs, cam_tvecs, points3d, K = unpack_params(x, n_cams, n_points, fxfy_equal)
    residuals = []
    # iterate obs_list (flat list) to avoid nested loops overhead in Python
    for cam_idx, pidx, uv in obs_list:
        rvec = cam_rvecs[cam_idx]
        tvec = cam_tvecs[cam_idx]
        X = points3d[pidx]
        proj = project_point(X, rvec, tvec, K)
        residuals.append(proj[0] - uv[0])
        residuals.append(proj[1] - uv[1])
    return np.array(residuals)


# -----------------------
# High level BA entry
# -----------------------
def run_bundle_adjustment(poses_init: List[Tuple[np.ndarray, np.ndarray]],
                          points3d_init: np.ndarray,
                          per_camera_obs: List[Dict[int, np.ndarray]],
                          init_K: np.ndarray,
                          fxfy_equal: bool = True,
                          verbose: int = 2):
    """
    poses_init: list of (R, t) for each camera (R 3x3, t 3,)
    points3d_init: (M,3) initial points
    per_camera_obs: list length n_cams mapping point_idx->2D observation
    """
    n_cams = len(poses_init)
    n_points = points3d_init.shape[0]
    # convert R to rvec
    cam_rvecs = []
    cam_tvecs = []
    for R, t in poses_init:
        rvec, _ = cv2.Rodrigues(R)
        cam_rvecs.append(rvec.ravel())
        cam_tvecs.append(t.ravel())
    cam_rvecs = np.array(cam_rvecs, dtype=float)
    cam_tvecs = np.array(cam_tvecs, dtype=float)
    points3d = points3d_init.copy()

    # Build a flat list of observations for faster iteration
    obs_list = []
    for cam_idx, obs_dict in enumerate(per_camera_obs):
        for pidx, uv in obs_dict.items():
            obs_list.append((cam_idx, int(pidx), np.array(uv, dtype=float)))

    x0 = pack_params(cam_rvecs, cam_tvecs, points3d, init_K, fxfy_equal)

    # choose loss: 'huber' or 'soft_l1'. Huber requires specifying parameter via 'loss' arg.
    loss = 'huber'  # robust
    # initial scale for huber: use median reprojection error heuristic
    res = least_squares(
        reprojection_residuals_ba,
        x0,
        args=(n_cams, n_points, per_camera_obs, obs_list, fxfy_equal),
        method='lm' if loss is None else 'trf',
        loss=loss,
        verbose=verbose,
        max_nfev=200,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
    )

    cam_rvecs_opt, cam_tvecs_opt, points3d_opt, K_opt = unpack_params(res.x, n_cams, n_points, fxfy_equal)
    # Convert rvec back to R matrices
    poses_opt = []
    for rvec, tvec in zip(cam_rvecs_opt, cam_tvecs_opt):
        R_opt, _ = cv2.Rodrigues(rvec)
        poses_opt.append((R_opt, tvec))
    return poses_opt, points3d_opt, K_opt, res


# -----------------------
# Convenience entrypoint: runs initial pose/triangulation flow and then BA
# -----------------------
def run_markerless_from_frames(frames: List[np.ndarray],
                               initial_K: Optional[np.ndarray] = None,
                               fxfy_equal: bool = True,
                               min_frames: int = 3,
                               verbose: int = 2):
    """
    frames: list of BGR or gray frames (at least 3).
    Returns optimized K, poses, points and the optimizer result.
    """
    if len(frames) < min_frames:
        raise ValueError("Need at least 3 frames with parallax for markerless autocalibration.")
    # convert to gray
    frames_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f for f in frames]
    img_shape = frames_gray[0].shape
    if initial_K is None:
        initial_K = default_initial_K(img_shape)

    # Build initial poses and triangulated points using previous helper
    poses_init, points3d_init, observations = build_observations(frames_gray, initial_K)
    if points3d_init.shape[0] == 0:
        raise RuntimeError("No triangulated points found. Try collecting frames with more parallax and texture.")

    per_camera_obs = pack_observations(points3d_init, observations, len(poses_init))
    poses_opt, points3d_opt, K_opt, res = run_bundle_adjustment(
        poses_init, points3d_init, per_camera_obs, initial_K, fxfy_equal=fxfy_equal, verbose=verbose
    )
    return K_opt, poses_opt, points3d_opt, res
