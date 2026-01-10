"""
Undistorter: uses camera intrinsics JSON to undistort frames.
Expected intrinsics JSON format:
{
  "fx": ...,
  "fy": ...,
  "cx": ...,
  "cy": ...,
  "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0
}
"""

import numpy as np
import cv2

class Undistorter:
    def __init__(self, intrinsics):
        self.K = np.array([[intrinsics["fx"], 0, intrinsics["cx"]],
                           [0, intrinsics["fy"], intrinsics["cy"]],
                           [0, 0, 1]], dtype=np.float64)
        dist = [intrinsics.get("k1",0), intrinsics.get("k2",0),
                intrinsics.get("p1",0), intrinsics.get("p2",0),
                intrinsics.get("k3",0)]
        self.dist = np.array(dist, dtype=np.float64)
        self.map1 = None
        self.map2 = None
        self._last_shape = None

    def undistort(self, frame):
        h, w = frame.shape[:2]
        if self._last_shape != (h,w):
            self.map1, self.map2 = cv2.initUndistortRectifyMap(self.K, self.dist, None, self.K, (w,h), cv2.CV_32FC1)
            self._last_shape = (h,w)
        return cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)