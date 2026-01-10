"""
IMU loader + complementary orientation filter.
CSV columns: time(s), ax,ay,az, gx,gy,gz
Outputs orientation as quaternion (x,y,z,w) approximating gravity alignment.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import csv

class IMUFuser:
    def __init__(self, imu_csv):
        self.imu = []
        if imu_csv:
            with open(imu_csv, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    t = float(row.get("time", row.get("timestamp", 0.0)))
                    ax = float(row.get("ax", row.get("accel_x", 0.0)))
                    ay = float(row.get("ay", row.get("accel_y", 0.0)))
                    az = float(row.get("az", row.get("accel_z", 0.0)))
                    gx = float(row.get("gx", row.get("gyro_x", 0.0)))
                    gy = float(row.get("gy", row.get("gyro_y", 0.0)))
                    gz = float(row.get("gz", row.get("gyro_z", 0.0)))
                    self.imu.append((t, np.array([ax,ay,az]), np.array([gx,gy,gz])))
        self.imu.sort(key=lambda x: x[0])
        # state orientation quaternion (as scipy Rotation)
        self.cur_rot = R.from_quat([0,0,0,1])
        self.last_t = None

    def get_orientation_at(self, t):
        if not self.imu:
            # return identity quaternion
            return np.array([0,0,0,1])
        # find nearest IMU sample
        idx = min(range(len(self.imu)), key=lambda i: abs(self.imu[i][0]-t))
        ts, acc, gyro = self.imu[idx]
        # simple complementary: use accelerometer for pitch/roll (gravity), use gyro for yaw integration
        # compute accel-based orientation (assume acc vector points to -Z in camera)
        ax,ay,az = acc
        acc_norm = np.linalg.norm(acc)
        if acc_norm < 1e-6:
            return self.cur_rot.as_quat()
        g = acc / acc_norm
        # compute roll & pitch from accelerometer
        pitch = np.arcsin(-g[0])  # approximate
        roll = np.arctan2(g[1], g[2])
        acc_rot = R.from_euler("yx", [pitch, roll], degrees=False)  # note axes mapping
        # gyro: integrate small rotation around z for yaw
        dt = 0.01
        if self.last_t is not None:
            dt = ts - self.last_t
        self.last_t = ts
        gz = gyro[2]
        yaw_delta = gz * dt
        yaw_rot = R.from_euler("z", yaw_delta)
        # combine: mostly gyro yaw + acc roll/pitch (complementary)
        fused = R.from_euler("z", [self.cur_rot.as_euler("zyx")[0]]) * acc_rot
        self.cur_rot = fused * yaw_rot
        return self.cur_rot.as_quat()