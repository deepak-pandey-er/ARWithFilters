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

SETTINGS_PATH = Path("app_settings.json")
DEFAULT_SETTINGS = {
    "udp_ip": "127.0.0.1",
    "udp_port": 5555
}


def normalize_intrinsics(intrinsics):
    """Normalize supported intrinsics JSON formats to a canonical dict.

    Supported input formats:
      - {"fx", "fy", "cx", "cy", ...}
      - {"camera_matrix": {"rows": 3, "cols": 3, "data": [..]}}
      - {"camera_matrix": [[..],[..],[..]]}
      - {"camera_matrix": [9 values]}
    """
    if intrinsics is None:
        return None

    intr = dict(intrinsics)

    def _read_camera_matrix(cm):
        if isinstance(cm, dict):
            data = cm.get("data")
            if data is None:
                data = cm.get("matrix")
            if data is None and cm.get("rows") == 3 and cm.get("cols") == 3:
                # Support OpenCV-style nested camera_matrix dict
                data = cm.get("data")
        else:
            data = cm

        if isinstance(data, (list, tuple)) and len(data) == 9:
            return np.array(data, dtype=float).reshape((3, 3))
        if isinstance(data, (list, tuple)) and len(data) == 3 and all(isinstance(row, (list, tuple)) and len(row) == 3 for row in data):
            return np.array(data, dtype=float)
        return None

    if "camera_matrix" in intr:
        K = _read_camera_matrix(intr["camera_matrix"])
        if K is not None:
            intr["fx"] = float(K[0, 0])
            intr["fy"] = float(K[1, 1])
            intr["cx"] = float(K[0, 2])
            intr["cy"] = float(K[1, 2])

    if {"fx", "fy", "cx", "cy"}.issubset(intr):
        return intr

    raise ValueError("Unsupported intrinsics JSON format: expected fx/fy/cx/cy or camera_matrix data.")


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

        self.udpIpEdit = QtWidgets.QLineEdit()
        self.udpIpEdit.setFixedWidth(140)
        self.udpPortEdit = QtWidgets.QLineEdit()
        self.udpPortEdit.setFixedWidth(80)
        self.udpPortEdit.setValidator(QtGui.QIntValidator(1, 65535, self))

        udp_layout = QtWidgets.QHBoxLayout()
        udp_layout.addWidget(QtWidgets.QLabel("UDP IP:"))
        udp_layout.addWidget(self.udpIpEdit)
        udp_layout.addWidget(QtWidgets.QLabel("Port:"))
        udp_layout.addWidget(self.udpPortEdit)
        self.saveUdpBtn = QtWidgets.QPushButton("Save UDP Settings")
        udp_layout.addWidget(self.saveUdpBtn)
        layout.addLayout(udp_layout)

        self.saveUdpBtn.clicked.connect(self.save_settings)
        self.load_settings()

    def log_msg(self, s):
        ts = time.strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {s}")
        print(s)

    def load_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open video", ".", "Videos (*.mp4 *.avi *.mov)")
        if path:
            self.videoPath = path
            self.log_msg(f"Loaded video: {path}")

    def load_settings(self):
        self.udp_ip = DEFAULT_SETTINGS["udp_ip"]
        self.udp_port = DEFAULT_SETTINGS["udp_port"]
        if SETTINGS_PATH.exists():
            try:
                with open(SETTINGS_PATH, "r") as f:
                    data = json.load(f)
                self.udp_ip = data.get("udp_ip", self.udp_ip)
                self.udp_port = int(data.get("udp_port", self.udp_port))
            except Exception as e:
                self.log_msg(f"Failed to load settings: {e}")
        self.udpIpEdit.setText(str(self.udp_ip))
        self.udpPortEdit.setText(str(self.udp_port))
        self.log_msg(f"Loaded UDP settings: {self.udp_ip}:{self.udp_port}")

    def save_settings(self):
        ip = self.udpIpEdit.text().strip() or DEFAULT_SETTINGS["udp_ip"]
        port_text = self.udpPortEdit.text().strip()
        try:
            port = int(port_text)
            if port < 1 or port > 65535:
                raise ValueError("Port out of range")
        except Exception:
            QtWidgets.QMessageBox.warning(self, "Invalid Port", "Port must be an integer between 1 and 65535.")
            return
        self.udp_ip = ip
        self.udp_port = port
        settings = {"udp_ip": self.udp_ip, "udp_port": self.udp_port}
        try:
            with open(SETTINGS_PATH, "w") as f:
                json.dump(settings, f, indent=2)
            self.log_msg(f"Saved UDP settings: {self.udp_ip}:{self.udp_port}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Failed", f"Unable to save settings: {e}")

    def load_intrinsics(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open intrinsics JSON", ".", "JSON (*.json)")
        if path:
            with open(path, "r") as f:
                raw = json.load(f)
            try:
                self.intrinsics = normalize_intrinsics(raw)
            except ValueError as e:
                QtWidgets.QMessageBox.critical(self, "Invalid Intrinsics", str(e))
                return
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
        K_list = K_opt.reshape(-1).tolist()
        out = {
            "camera_matrix": {
                "rows": 3,
                "cols": 3,
                "data": K_list
            },
            "fx": float(K_opt[0, 0]),
            "fy": float(K_opt[1, 1]),
            "cx": float(K_opt[0, 2]),
            "cy": float(K_opt[1, 2]),
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "k3": 0.0
        }
        out_path = "intrinsics_calibrated.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        self.intrinsics = normalize_intrinsics(out)  # update in-memory intrinsics used by pipeline
        self.log_msg(f"Calibration successful. Saved intrinsics to {out_path}")

    def run_pipeline(self):
        if not self.videoPath:
            self.log_msg("No video selected.")
            return
        if not self.intrinsics:
            self.log_msg("No intrinsics JSON selected.")
            return

        self.log_msg("Starting pipeline...")
        try:
            intr = normalize_intrinsics(self.intrinsics)
        except ValueError as e:
            self.log_msg(f"Invalid intrinsics: {e}")
            return

        self.log_msg(f"Using UDP settings: {self.udp_ip}:{self.udp_port}")

        cap = cv2.VideoCapture(self.videoPath)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_idx = 0

        undistorter = Undistorter(intr)
        use_torch_guess = torch_installed_on_path()
        segmenter = FieldSegmenter(device="cpu", use_torch=use_torch_guess)
        # segmenter = FieldSegmenter(device="cuda" if torch_is_available() else "cpu")

        pitcher = PitchExtractor()
        vo = MonoVO(intr)
        imu = IMUFuser(self.imuPath) if self.imuPath else IMUFuser(None)
        aligner = GroundAligner(intr)
        smoother = PoseKalman()
        ar = AROutput(intr, OUTPUT_DIR / "poses.csv")

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
