"""
Robust Ground alignment and solvePnP fallback logic.

Replaces earlier implementation that called solvePnP with only 2 points (which causes an OpenCV assertion).
This version:
 - Handles cases with >=4, ==3, ==2 image points appropriately.
 - For >=4: runs solvePnP (ITERATIVE).
 - For 3: runs solvePnP with an IMU-based initial rvec as an extrinsic guess.
 - For 2: estimates depth from pixel separation using pinhole approximation and IMU rotation for orientation.
 - Falls back to IMU-only pose if none of the above succeeds.

World coordinate convention:
 - X: across the pitch (left-right)
 - Y: up
 - Z: forward away from the camera (standard right-handed camera -> world inversion used for cam_to_world)
"""

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

# wicket-to-wicket distance in meters (22 yards ≈ 20.1168 m)
WICKET_DIST_M = 20.1168

class GroundAligner:
    def __init__(self, intrinsics):
        fx = intrinsics["fx"]; fy = intrinsics["fy"]; cx = intrinsics["cx"]; cy = intrinsics["cy"]
        self.K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
        self.fx = fx; self.fy = fy; self.cx = cx; self.cy = cy

    def align_frame(self, rel_pose, pitch_points, imu_quat, timestamp):
        """
        rel_pose: 4x4 relative camera pose from VO (prev -> cur). (Not required here)
        pitch_points: list of image (x,y) tuples (wicket / pitch endpoints / centerline points)
        imu_quat: quaternion [x,y,z,w] giving orientation estimate (camera in world)
        Returns: cam_to_world 4x4 transform (camera pose in world coordinates)
        """

        def build_cam_to_world_from_world_to_cam(Rwc, tvec):
            world_to_cam = np.eye(4, dtype=np.float64)
            world_to_cam[:3,:3] = Rwc
            world_to_cam[:3,3] = tvec.reshape(3)
            return np.linalg.inv(world_to_cam)

        # Convert IMU quaternion to rotation matrix (if valid)
        try:
            r_imu = R.from_quat([imu_quat[0], imu_quat[1], imu_quat[2], imu_quat[3]])
            R_imu_mat = r_imu.as_matrix()
        except Exception:
            R_imu_mat = np.eye(3)

        n_pts = len(pitch_points)

        # Case: >=4 points -> normal solvePnP
        if n_pts >= 4:
            img_pts = np.array(pitch_points, dtype=np.float64)
            # Map these image points to linearly spaced object points across the known wicket segment.
            xs = np.linspace(-WICKET_DIST_M/2, WICKET_DIST_M/2, len(img_pts))
            object_pts = np.vstack([xs, np.zeros_like(xs), np.zeros_like(xs)]).T.astype(np.float64)
            try:
                success, rvec, tvec = cv2.solvePnP(object_pts, img_pts, self.K, None, flags=cv2.SOLVEPNP_ITERATIVE)
                if success:
                    Rmat, _ = cv2.Rodrigues(rvec)
                    return build_cam_to_world_from_world_to_cam(Rmat, tvec.reshape(3))
            except Exception:
                # Fall through to other strategies
                pass

        # Case: exactly 3 points -> try using IMU as initial rotation guess
        if n_pts == 3:
            img_pts = np.array(pitch_points, dtype=np.float64)
            xs = np.linspace(-WICKET_DIST_M/2, WICKET_DIST_M/2, 3)
            object_pts = np.vstack([xs, np.zeros_like(xs), np.zeros_like(xs)]).T.astype(np.float64)
            try:
                rvec_init, _ = cv2.Rodrigues(R_imu_mat)
                success, rvec, tvec = cv2.solvePnP(object_pts, img_pts, self.K, None,
                                                  rvec=rvec_init.reshape(3,1), useExtrinsicGuess=True,
                                                  flags=cv2.SOLVEPNP_ITERATIVE)
                if success:
                    Rmat, _ = cv2.Rodrigues(rvec)
                    return build_cam_to_world_from_world_to_cam(Rmat, tvec.reshape(3))
            except Exception:
                pass

        # Case: exactly 2 points -> estimate depth from pixel separation and use IMU rotation
        if n_pts == 2:
            p1 = np.array(pitch_points[0], dtype=np.float64)
            p2 = np.array(pitch_points[1], dtype=np.float64)
            dx = np.linalg.norm(p2 - p1)
            if dx > 1.0:
                # Z_est ≈ fx * W / dx_px
                Z_est = (self.fx * WICKET_DIST_M) / dx
                mid = (p1 + p2) * 0.5
                u_mid, v_mid = mid
                x_cam = (u_mid - self.cx) * Z_est / self.fx
                y_cam = (v_mid - self.cy) * Z_est / self.fy
                tvec = np.array([x_cam, y_cam, Z_est], dtype=np.float64)
                Rwc = R_imu_mat
                try:
                    return build_cam_to_world_from_world_to_cam(Rwc, tvec)
                except Exception:
                    pass

        # Last resort: IMU-only orientation + heuristic height
        cam_to_world = np.eye(4, dtype=np.float64)
        cam_to_world[:3,:3] = R_imu_mat
        # Heuristic: set camera 5 m above ground (Y up). Adjust as needed.
        cam_to_world[:3,3] = np.array([0.0, 5.0, 0.0], dtype=np.float64)
        return cam_to_world