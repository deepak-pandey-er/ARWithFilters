"""
Simple Kalman smoother for translation + velocity + orientation smoothing.
State: [x,y,z, vx,vy,vz] plus orientation as quaternion smoothed separately via slerp.
A very simple implementation for broadcast smoothing.
"""

import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

class PoseKalman:
    def __init__(self, dt=1/25.0):
        self.dt = dt
        # init state
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.P = np.eye(6) * 0.1
        self.Q = np.eye(6) * 0.01
        self.R = np.eye(3) * 0.05
        self.last_q = Quaternion(axis=[0,0,1], angle=0)

    def update(self, cam_to_world, timestamp=None):
        # cam_to_world: 4x4 transform
        pos = cam_to_world[:3,3]
        # predict
        self.pos = self.pos + self.vel * self.dt
        # update position with measurement pos
        z = pos
        H = np.hstack([np.eye(3), np.zeros((3,3))])
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        y = z - H @ np.hstack([self.pos, self.vel])
        x_upd = K @ y
        state = np.hstack([self.pos, self.vel]) + x_upd
        self.pos = state[0:3]
        self.vel = state[3:6]
        # update P
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P + self.Q
        # orientation smoothing via slerp
        Rm = cam_to_world[:3,:3]
        q = Quaternion(matrix=Rm)
        # slerp factor small to smooth
        alpha = 0.15
        self.last_q = Quaternion.slerp(self.last_q, q, amount=alpha)
        # build smoothed transform
        T = np.eye(4)
        T[:3,:3] = self.last_q.rotation_matrix
        T[:3,3] = self.pos
        return T