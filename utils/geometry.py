"""
Utility helpers for transforms and quaternions.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

def make_transform_from_rot_trans(rot: np.ndarray, trans: np.ndarray):
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = rot
    T[:3,3] = trans
    return T

def transform_inverse(T: np.ndarray):
    Rm = T[:3,:3]; t = T[:3,3]
    Tinv = np.eye(4, dtype=np.float64)
    Tinv[:3,:3] = Rm.T
    Tinv[:3,3] = -Rm.T @ t
    return Tinv

def quat_from_matrix(Rm: np.ndarray):
    r = R.from_matrix(Rm)
    return r.as_quat()  # x,y,z,w