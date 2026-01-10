"""
PyQt5 UI and pipeline orchestrator.
Simple UI: load video, intrinsics JSON, imu CSV (optional), start processing.
"""

import sys
import os
import json
import time
import importlib.util
from pathlib import Path

from PyQt5 import QtWidgets, QtGui, QtCore
import cv2
import numpy as np

from pipeline.undistort import Undistorter
from pipeline.segmentation import FieldSegmenter
from pipeline.lines import PitchExtractor
from pipeline.visual_odometry import MonoVO
from pipeline.imu import IMUFuser
from pipeline.ground_lock import GroundAligner
from pipeline.kalman import PoseKalman
from pipeline.ar_output import AROutput

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cricket AR Tracker")
        self.setGeometry(50, 50, 1200, 800)

        layout = QtWidgets.QVBoxLayout(self)

        topbar = QtWidgets.QHBoxLayout()
        self.loadVideoBtn = QtWidgets.QPushButton("Load Video")
        self.loadIntrBtn = QtWidgets.QPushButton("Load Intrinsics (JSON)")
        self.loadIMUBtn = QtWidgets.QPushButton("Load IMU CSV (optional)")
        self.runBtn = QtWidgets.QPushButton("Run")
        topbar.addWidget(self.loadVideoBtn)
        topbar.addWidget(self.loadIntrBtn)
        topbar.addWidget(self.loadIMUBtn)
        topbar.addWidget(self.runBtn)
        layout.addLayout(topbar)

        self.videoLabel = QtWidgets.QLabel()
        self.videoLabel.setFixedSize(960, 540)
        layout.addWidget(self.videoLabel)

        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

        # state
        self.videoPath = None
        self.intrinsics = None
        self.imuPath = None

        # connect
        self.loadVideoBtn.clicked.connect(self.load_video)
        self.loadIntrBtn.clicked.connect(self.load_intrinsics)
        self.loadIMUBtn.clicked.connect(self.load_imu)
        self.runBtn.clicked.connect(self.run_pipeline)

    def log_msg(self, s):
        ts = time.strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {s}")
        print(s)

    def load_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open video", ".", "Videos (*.mp4 *.avi *.mov)")
        if path:
            self.videoPath = path
            self.log_msg(f"Loaded video: {path}")

    def load_intrinsics(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open intrinsics JSON", ".", "JSON (*.json)")
        if path:
            with open(path, "r") as f:
                self.intrinsics = json.load(f)
            self.log_msg(f"Loaded intrinsics: {path}")

    def load_imu(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open IMU CSV", ".", "CSV (*.csv)")
        if path:
            self.imuPath = path
            self.log_msg(f"Loaded IMU CSV: {path}")

    def run_pipeline(self):
        if not self.videoPath:
            self.log_msg("No video selected.")
            return
        if not self.intrinsics:
            self.log_msg("No intrinsics JSON selected.")
            return

        self.log_msg("Starting pipeline...")
        cap = cv2.VideoCapture(self.videoPath)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_idx = 0

        undistorter = Undistorter(self.intrinsics)
        use_torch_guess = torch_installed_on_path()
        segmenter = FieldSegmenter(device="cpu", use_torch=use_torch_guess)
        # segmenter = FieldSegmenter(device="cuda" if torch_is_available() else "cpu")

        pitcher = PitchExtractor()
        vo = MonoVO(self.intrinsics)
        imu = IMUFuser(self.imuPath) if self.imuPath else IMUFuser(None)
        aligner = GroundAligner(self.intrinsics)
        smoother = PoseKalman()
        ar = AROutput(self.intrinsics, OUTPUT_DIR / "poses.csv")

        self.log_msg("Pipeline modules created. Processing frames...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = frame_idx / fps
            und = undistorter.undistort(frame)
            mask = segmenter.segment(und)
            pitch_lines, pitch_points = pitcher.extract_centerline(und, mask)
            rel_pose = vo.process_frame(und, timestamp)
            imu_orient = imu.get_orientation_at(timestamp)
            world_pose = aligner.align_frame(rel_pose, pitch_points, imu_orient, timestamp)
            smooth_pose = smoother.update(world_pose, timestamp)
            ar_img = ar.render_overlay(und, smooth_pose, mask, pitch_lines)

            # display
            draw = cv2.cvtColor(ar_img, cv2.COLOR_BGR2RGB)
            h, w, ch = draw.shape
            qimg = QtGui.QImage(draw.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(qimg).scaled(self.videoLabel.size(), QtCore.Qt.KeepAspectRatio)
            self.videoLabel.setPixmap(pix)
            QtWidgets.QApplication.processEvents()

            frame_idx += 1

        ar.close()
        cap.release()
        self.log_msg("Processing finished. Outputs saved to output/")

def torch_is_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def torch_installed_on_path():
    # Check existence of torch package without importing it (avoids triggering DLL load)
    return importlib.util.find_spec("torch") is not None

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())