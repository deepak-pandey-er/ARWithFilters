"""
Markerless intrinsic autocalibration (approximate).

Workflow:
- Given N frames (grayscale BGR images), detect ORB features and match frame0 -> frame_i.
- For each pair, estimate essential matrix with an initial K, recover relative pose (R,t).
- Triangulate matched points between frame0 and frame_i to get 3D points (up-to-scale).
- Build a set of observations (image points and corresponding 3D points).
- Optimize camera intrinsics (fx, fy, cx, cy) by minimizing reprojection error while keeping poses fixed.

Limitations:
- Poses are computed using an initial K guess. If initial K is far off, poses will be wrong.
- This implementation optimizes intrinsics only. Full bundle adjustment would jointly optimize
  poses and 3D points (more accurate but more complex).
"""

import json
import numpy as np
import cv2
from scipy.optimize import least_squares
from typing import List, Tuple, Optional


def default_initial_K(img_shape: Tuple[int, int]) -> np.ndarray:
    h, w = img_shape
    # Fallback initial focal length: approx half of image width (can be tuned)
    f = 0.5 * w
    cx = w / 2.0
    cy = h / 2.0
    return np.array([[f, 0, cx],
                     [0, f, cy],
                     [0, 0, 1]], dtype=float)


def detect_and_match(frame1_gray: np.ndarray, frame2_gray: np.ndarray, nfeatures=2000):
    orb = cv2.ORB_create(nfeatures)
    k1, d1 = orb.detectAndCompute(frame1_gray, None)
    k2, d2 = orb.detectAndCompute(frame2_gray, None)
    if d1 is None or d2 is None:
        return [], [], []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(d1, d2, k=2)
    good = []
    pts1 = []
    pts2 = []
    for m in matches:
        if len(m) == 2:
            m1, m2 = m
            if m1.distance < 0.75 * m2.distance:
                good.append(m1)
                pts1.append(k1[m1.queryIdx].pt)
                pts2.append(k2[m1.trainIdx].pt)
    pts1 = np.array(pts1, dtype=float)
    pts2 = np.array(pts2, dtype=float)
    return good, pts1, pts2


def recover_pairwise_pose(pts1, pts2, K):
    # Requires at least 5 points for findEssentialMat
    if len(pts1) < 8:
        return None, None, None
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        return None, None, None
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    return R, t, mask_pose


def triangulate_points(pts1, pts2, K, R, t):
    # Convert to homogeneous projection matrices
    P0 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P1 = K @ np.hstack((R, t.reshape(3, 1)))
    pts1_h = pts1.T
    pts2_h = pts2.T
    pts4d = cv2.triangulatePoints(P0, P1, pts1_h, pts2_h)  # shape (4, N)
    pts3d = (pts4d[:3, :] / pts4d[3, :]).T  # shape (N, 3)
    return pts3d


def build_observations(frames_gray: List[np.ndarray], init_K: np.ndarray):
    """
    Build camera poses relative to frame0 and triangulate points.
    Returns:
      poses: list of (R, t) where first pose is (I, 0)
      points3d: Nx3 array of triangulated points (from multiple pairs)
      observations: list of lists where observations[i] are the 2D points in frame i that correspond to points3d (or None)
    """
    n = len(frames_gray)
    poses = [(np.eye(3), np.zeros(3))]  # frame0 pose
    img0 = frames_gray[0]
    all_pts3d = []
    all_obs = []  # list of tuples: (frame_index, img_point, point3d_index)
    # We will triangulate between frame0 and frame_i for i=1..n-1
    for i in range(1, n):
        _, pts0, ptsi = detect_and_match(img0, frames_gray[i])
        if len(pts0) < 8:
            continue
        R, t, mask_pose = recover_pairwise_pose(pts0, ptsi, init_K)
        if R is None:
            continue
        poses.append((R, t.ravel()))
        # Take only inliers
        mask_flat = mask_pose.ravel().astype(bool)
        pts0_in = pts0[mask_flat]
        ptsi_in = ptsi[mask_flat]
        if len(pts0_in) < 8:
            continue
        pts3d = triangulate_points(pts0_in, ptsi_in, init_K, R, t)
        base_idx = len(all_pts3d)
        all_pts3d.append(pts3d)
        # store observations for frame0 and frame i
        for j, p3 in enumerate(pts3d):
            # frame0 observation
            all_obs.append((0, tuple(pts0_in[j]), base_idx + j))
            # frame i observation
            all_obs.append((i, tuple(ptsi_in[j]), base_idx + j))
    if not all_pts3d:
        return poses, np.empty((0, 3)), []
    points3d = np.vstack(all_pts3d)
    return poses, points3d, all_obs


def pack_observations(points3d, observations, n_cameras):
    """
    Convert list of observations of form (frame_index, img_point_xy, point3d_index)
    into per-camera lists for easier residual computation.
    Returns per_camera_obs: list length n_cameras, each a dict mapping point_index->2Dpoint
    """
    per_camera = [dict() for _ in range(n_cameras)]
    for frame_idx, img_pt, pidx in observations:
        per_camera[frame_idx][pidx] = np.array(img_pt, dtype=float)
    return per_camera


def reprojection_residuals(params, points3d, per_camera_obs, poses, img_shape, fxfy_equal=True):
    """
    params: [fx, fy, cx, cy] or [f, cx, cy] if fxfy_equal
    points3d: (M,3)
    per_camera_obs: list of dicts: camera -> {pidx: (x,y)}
    poses: list of (R,t)
    """
    if fxfy_equal:
        f = params[0]
        cx = params[1]
        cy = params[2]
        K = np.array([[f, 0, cx],
                      [0, f, cy],
                      [0, 0, 1]], dtype=float)
    else:
        f_x = params[0]
        f_y = params[1]
        cx = params[2]
        cy = params[3]
        K = np.array([[f_x, 0, cx],
                      [0, f_y, cy],
                      [0, 0, 1]], dtype=float)

    residuals = []
    for cam_idx, obs_dict in enumerate(per_camera_obs):
        R, t = poses[cam_idx]
        rvec, _ = cv2.Rodrigues(R)
        for pidx, uv in obs_dict.items():
            X = points3d[pidx].reshape(1, 3)
            proj, _ = cv2.projectPoints(X, rvec, t.reshape(3, 1), K, distCoeffs=None)
            u_proj = proj.ravel()
            residuals.append(u_proj[0] - uv[0])
            residuals.append(u_proj[1] - uv[1])
    return np.array(residuals)


def optimize_intrinsics(points3d, per_camera_obs, poses, img_shape, init_K, fxfy_equal=True):
    h, w = img_shape
    if fxfy_equal:
        init_f = (init_K[0, 0] + init_K[1, 1]) / 2.0
        init_cx = init_K[0, 2]
        init_cy = init_K[1, 2]
        x0 = np.array([init_f, init_cx, init_cy], dtype=float)
    else:
        x0 = np.array([init_K[0, 0], init_K[1, 1], init_K[0, 2], init_K[1, 2]], dtype=float)

    def fun(x):
        return reprojection_residuals(x, points3d, per_camera_obs, poses, img_shape, fxfy_equal)

    res = least_squares(fun, x0, method='lm', verbose=2, max_nfev=200)
    if fxfy_equal:
        f_opt, cx_opt, cy_opt = res.x
        K_opt = np.array([[f_opt, 0, cx_opt],
                          [0, f_opt, cy_opt],
                          [0, 0, 1]], dtype=float)
    else:
        fx_opt, fy_opt, cx_opt, cy_opt = res.x
        K_opt = np.array([[fx_opt, 0, cx_opt],
                          [0, fy_opt, cy_opt],
                          [0, 0, 1]], dtype=float)
    return K_opt, res


def calibrate_from_frames(frames: List[np.ndarray], initial_K: Optional[np.ndarray] = None, fxfy_equal=True):
    """
    Main entry point.
    frames: list of color or grayscale images (BGR or gray)
    initial_K: optional initial camera matrix (3x3)
    returns: optimized K (3x3), result object (scipy result)
    """
    if len(frames) < 3:
        raise ValueError("Need at least 3 frames with parallax for markerless autocalibration.")
    frames_gray = []
    for f in frames:
        if f.ndim == 3:
            frames_gray.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        else:
            frames_gray.append(f)

    img_shape = frames_gray[0].shape
    if initial_K is None:
        initial_K = default_initial_K(img_shape)

    poses, points3d, observations = build_observations(frames_gray, initial_K)
    if points3d.shape[0] == 0:
        raise RuntimeError("No triangulated points found. Try collecting frames with more parallax and texture.")

    per_camera_obs = pack_observations(points3d, observations, len(poses))
    K_opt, res = optimize_intrinsics(points3d, per_camera_obs, poses, img_shape, initial_K, fxfy_equal=fxfy_equal)
    return K_opt, res


def save_intrinsics(path: str, K: np.ndarray):
    out = {
        "camera_matrix": {
            "rows": 3,
            "cols": 3,
            "data": K.reshape(-1).tolist()
        }
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
