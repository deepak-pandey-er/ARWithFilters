"""
A simple ORB-based monocular VO that provides relative poses between frames.
Outputs pose as 4x4 numpy transform (camera-to-camera).
"""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

class MonoVO:
    def __init__(self, intrinsics):
        self.prev_kp = None
        self.prev_des = None
        self.prev_img = None
        self.orb = cv2.ORB_create(3000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        fx = intrinsics["fx"]; fy = intrinsics["fy"]; cx = intrinsics["cx"]; cy = intrinsics["cy"]
        self.K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)

    def process_frame(self, img, timestamp=None):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        pose = np.eye(4, dtype=np.float64)
        if self.prev_img is None:
            self.prev_img = gray
            self.prev_kp = kp
            self.prev_des = des
            return pose
        if des is None or self.prev_des is None:
            self.prev_img = gray; self.prev_kp = kp; self.prev_des = des
            return pose
        matches = self.matcher.match(des, self.prev_des)
        matches = sorted(matches, key=lambda x: x.distance)[:200]
        pts1 = np.float32([kp[m.queryIdx].pt for m in matches])
        pts2 = np.float32([self.prev_kp[m.trainIdx].pt for m in matches])
        if len(pts1) < 6:
            self.prev_img = gray; self.prev_kp = kp; self.prev_des = des
            return pose
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, cv2.RANSAC, 0.999, 1.0)
        if E is None:
            self.prev_img = gray; self.prev_kp = kp; self.prev_des = des
            return pose
        _, Rmat, tvec, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K)
        # build transform from prev -> current
        T = np.eye(4)
        T[:3,:3] = Rmat
        T[:3,3] = tvec.squeeze()
        # in monocular case scale unknown. Keep raw relative transform; higher-level module will scale.
        self.prev_img = gray; self.prev_kp = kp; self.prev_des = des
        return T