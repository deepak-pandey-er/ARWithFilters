"""
Project a simple ground-grid / pitch model into image and save poses to CSV.
Pose format saved: timestamp, tx, ty, tz, qx, qy, qz, qw
"""

import csv
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

class AROutput:
    def __init__(self, intrinsics, csv_path):
        self.K = np.array([[intrinsics["fx"],0,intrinsics["cx"]],[0,intrinsics["fy"],intrinsics["cy"]],[0,0,1]], dtype=np.float64)
        self.csv_path = csv_path
        self.f = open(csv_path, "w", newline="")
        self.writer = csv.writer(self.f)
        self.writer.writerow(["timestamp","tx","ty","tz","qx","qy","qz","qw"])
        # sample 3D pitch geometry for overlay: centerline long line
        self.grid_points = self._make_pitch_points()

    def _make_pitch_points(self):
        # generate points along centerline in world coords (x across, y up, z forward)
        half = 10.0
        pts = []
        for z in np.linspace(-10,10,21):
            pts.append([0.0, 0.0, z])
        return np.array(pts, dtype=np.float32)

    def render_overlay(self, img, cam_to_world, mask, pitch_lines):
        # draw segmentation mask overlay
        overlay = img.copy()
        color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(overlay, 0.85, color_mask, 0.15, 0)

        # project pitch grid
        # world_to_cam is inverse of cam_to_world
        world_to_cam = np.linalg.inv(cam_to_world)
        Rwc = world_to_cam[:3,:3]; twc = world_to_cam[:3,3]
        rvec, _ = cv2.Rodrigues(Rwc)
        pts2d, _ = cv2.projectPoints(self.grid_points, rvec, twc, self.K, None)
        pts2d = pts2d.reshape(-1,2).astype(int)
        for p in pts2d:
            cv2.circle(overlay, tuple(p), 3, (0,0,255), -1)

        # draw pitch lines
        if pitch_lines:
            for ((x1,y1),(x2,y2)) in pitch_lines:
                cv2.line(overlay, (x1,y1), (x2,y2), (255,0,0), 2)

        # annotate
        h,w = overlay.shape[:2]
        cv2.putText(overlay, "AR overlay", (10,h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        # write pose to csv
        tx,ty,tz = cam_to_world[:3,3]
        Rmat = cam_to_world[:3,:3]
        rot = R.from_matrix(Rmat)
        qx,qy,qz,qw = rot.as_quat()  # x,y,z,w
        self.writer.writerow([0.0, tx,ty,tz, qx,qy,qz,qw])
        return overlay

    def close(self):
        self.f.close()