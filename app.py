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

# New import for BA autocalib
from pipeline.ba_autocalib import run_markerless_from_frames, default_initial_K  # noqa: F401

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
        self.calibBtn = QtWidgets.QPushButton("Run Markerless Calibration")  # NEW
        self.runBtn = QtWidgets.QPushButton("Run")
        topbar.addWidget(self.loadVideoBtn)
        topbar.addWidget(self.loadIntrBtn)
        topbar.addWidget(self.loadIMUBtn)
        topbar.addWidget(self.calibBtn)  # NEW
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
        self.calibBtn.clicked.connect(self.run_markerless_calib)  # NEW handler
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

    def run_markerless_calib(self):
        """
        Entrypoint for the UI button. Samples frames and runs markerless BA-based calibration.
        Sampling strategy:
            - If a video is loaded: sample `n_frames` evenly spaced frames from the video.
            - Else: open webcam and let user press SPACE to capture frames interactively.
        """
        # ask for number of frames
        n_frames, ok = QtWidgets.QInputDialog.getInt(self, "Frames", "Number of frames to capture (>=3):", 8, 3, 50, 1)
        if not ok:
            return

        frames = []
        if self.videoPath:
            # sample evenly
            cap = cv2.VideoCapture(self.videoPath)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total < n_frames:
                self.log_msg(f"Video has only {total} frames; reducing requested frames.")
                n_frames = max(3, total)
            indices = np.linspace(0, max(0, total - 1), n_frames, dtype=int)
            self.log_msg(f"Sampling {n_frames} frames from video at indices: {indices.tolist()}")
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame.copy())
            cap.release()
        else:
            # open webcam capture
            cam_id, okc = QtWidgets.QInputDialog.getInt(self, "Camera", "Camera index:", 0, 0, 10, 1)
            if not okc:
                return
            cap = cv2.VideoCapture(cam_id)
            if not cap.isOpened():
                QtWidgets.QMessageBox.critical(self, "Camera Error", f"Cannot open camera {cam_id}")
                return
            self.log_msg("Press SPACE to capture a frame, ESC to cancel.")
            while len(frames) < n_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                disp = frame.copy()
                cv2.putText(disp, f"Frame {len(frames)}/{n_frames}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.imshow("Capture (press SPACE)", disp)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    self.log_msg("ESC pressed. Stopping capture.")
                    break
                if key == 32:
                    frames.append(frame.copy())
                    self.log_msg(f"Captured frame {len(frames)}/{n_frames}")
            cap.release()
            cv2.destroyAllWindows()

        if len(frames) < 3:
            QtWidgets.QMessageBox.warning(self, "Not enough frames", "Captured fewer than 3 frames. Aborting calibration.")
            return

        # run calibration (this may take time)
        self.log_msg("Running markerless bundle-adjustment autocalibration (this may take a while)...")
        try:
            # initial K guess from frame size
            h, w = frames[0].shape[:2]
            K0 = default_initial_K((h, w))
            K_opt, poses_opt, pts_opt, res = run_markerless_from_frames(frames, initial_K=K0, fxfy_equal=True, verbose=2)
        except Exception as e:
            self.log_msg(f"Calibration failed: {e}")
            QtWidgets.QMessageBox.critical(self, "Calibration Failed",
                                           "Markerless calibration failed. Try capturing frames with more parallax and texture, or use a checkerboard/marker fallback.")
            return

        # Save intrinsics to JSON and update internal state
        out = {
            "camera_matrix": {
                "rows": 3,
                "cols": 3,
                "data": K_opt.reshape(-1).tolist()
            }
        }
        out_path = "intrinsics_calibrated.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        self.intrinsics = out  # update in-memory intrinsics used by pipeline
        self.log_msg(f"Calibration successful. Saved intrinsics to {out_path}")

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
