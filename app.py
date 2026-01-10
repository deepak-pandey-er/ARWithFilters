"""
PyQt5 UI and pipeline orchestrator with STYPE UDP broadcast support and async calibration.
"""
import sys
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

# BA autocalib
from pipeline.ba_autocalib import run_markerless_from_frames, default_initial_K
# STYPE UDP broadcaster
from pipeline.udp_stype import send_camera_stype, BackgroundSender

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


class CalibrationWorker(QtCore.QObject):
    """
    Worker object to run calibration inside a QThread.
    Signals:
      progress(int) - optional progress percentage (not fine-grained here)
      finished(dict) - emits result dict { 'K': K_opt, 'poses': poses_opt, 'pts': pts_opt, 'res': res }
      error(str) - emits error message on failure
    """
    progress = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)

    def __init__(self, frames, initial_K=None, fxfy_equal=True, verbose=0):
        super().__init__()
        self.frames = frames
        self.initial_K = initial_K
        self.fxfy_equal = fxfy_equal
        self.verbose = verbose
        self._cancel_requested = False

    @QtCore.pyqtSlot()
    def run(self):
        try:
            # Indicate started
            self.progress.emit(0)

            if self.initial_K is None:
                h, w = self.frames[0].shape[:2]
                self.initial_K = default_initial_K((h, w))

            # Quick notify before heavy work
            self.progress.emit(10)

            # Run the heavy BA call (this is the main time-consuming operation)
            # Note: run_markerless_from_frames is a blocking call; progress here is coarse.
            K_opt, poses_opt, pts_opt, res = run_markerless_from_frames(
                self.frames,
                initial_K=self.initial_K,
                fxfy_equal=self.fxfy_equal,
                verbose=self.verbose
            )

            # Done
            self.progress.emit(100)
            self.finished.emit({'K': K_opt, 'poses': poses_opt, 'pts': pts_opt, 'res': res})
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cricket AR Tracker")
        self.setGeometry(50, 50, 1200, 900)

        layout = QtWidgets.QVBoxLayout(self)

        topbar = QtWidgets.QHBoxLayout()
        self.loadVideoBtn = QtWidgets.QPushButton("Load Video")
        self.loadIntrBtn = QtWidgets.QPushButton("Load Intrinsics (JSON)")
        self.loadIMUBtn = QtWidgets.QPushButton("Load IMU CSV (optional)")
        self.calibBtn = QtWidgets.QPushButton("Run Markerless Calibration")
        self.runBtn = QtWidgets.QPushButton("Run")
        topbar.addWidget(self.loadVideoBtn)
        topbar.addWidget(self.loadIntrBtn)
        topbar.addWidget(self.loadIMUBtn)
        topbar.addWidget(self.calibBtn)
        topbar.addWidget(self.runBtn)

        # UDP broadcast controls
        self.ipEdit = QtWidgets.QLineEdit("127.0.0.1")
        self.portEdit = QtWidgets.QLineEdit("5005")
        self.broadcastCheck = QtWidgets.QCheckBox("Enable UDP Broadcast (STYPE)")
        self.continuousCheck = QtWidgets.QCheckBox("Continuous")
        self.rateSpin = QtWidgets.QSpinBox()
        self.rateSpin.setRange(1, 100)
        self.rateSpin.setValue(10)
        topbar.addWidget(QtWidgets.QLabel("IP:"))
        topbar.addWidget(self.ipEdit)
        topbar.addWidget(QtWidgets.QLabel("Port:"))
        topbar.addWidget(self.portEdit)
        topbar.addWidget(self.broadcastCheck)
        topbar.addWidget(self.continuousCheck)
        topbar.addWidget(QtWidgets.QLabel("Hz:"))
        topbar.addWidget(self.rateSpin)

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
        self.packet_no = 0
        self._bg_sender = None

        # thread handles
        self._calib_thread = None
        self._calib_worker = None
        self._progress_dialog = None

        # connect
        self.loadVideoBtn.clicked.connect(self.load_video)
        self.loadIntrBtn.clicked.connect(self.load_intrinsics)
        self.loadIMUBtn.clicked.connect(self.load_imu)
        self.calibBtn.clicked.connect(self.run_markerless_calib)
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

    def _show_calib_results(self, K_opt, pose0):
        # Display intrinsics matrix and first camera pose (R, t) in a dialog
        R0, t0 = pose0
        rvec0, _ = cv2.Rodrigues(R0)
        text = []
        text.append("Optimized camera matrix (K):")
        text.append(np.array2string(K_opt, precision=3, separator=', '))
        text.append("")
        text.append("First camera pose (R):")
        text.append(np.array2string(R0, precision=3, separator=', '))
        text.append("")
        text.append("First camera translation (t) in meters:")
        text.append(np.array2string(t0, precision=4, separator=', '))
        text.append("")
        text.append("Rodrigues rvec:")
        text.append(np.array2string(rvec0.ravel(), precision=4, separator=', '))
        dlg = QtWidgets.QMessageBox(self)
        dlg.setWindowTitle("Calibration Results")
        dlg.setText('\n'.join(text))
        dlg.exec_()

    def _stop_bg_sender(self):
        if self._bg_sender is not None:
            try:
                self._bg_sender.stop()
            except Exception:
                pass
            self._bg_sender = None

    def _on_calib_finished(self, result):
        # Clean up thread + progress dialog
        try:
            if self._progress_dialog is not None:
                self._progress_dialog.close()
                self._progress_dialog = None
        except Exception:
            pass

        # result is dict {'K','poses','pts','res'}
        K_opt = result['K']
        poses_opt = result['poses']
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
        self.intrinsics = out
        self.log_msg(f"Calibration successful. Saved intrinsics to {out_path}")

        # show results
        if len(poses_opt) > 0:
            pose0 = poses_opt[0]
            self._show_calib_results(K_opt, pose0)

        # send STYPE packet if enabled (reuse previous logic)
        ip = self.ipEdit.text().strip()
        try:
            port = int(self.portEdit.text().strip())
        except Exception:
            port = 5005
        do_bcast = self.broadcastCheck.isChecked()
        continuous = self.continuousCheck.isChecked()
        rate = float(self.rateSpin.value())

        # stop any previous continuous sender
        self._stop_bg_sender()

        if do_bcast and len(poses_opt) > 0:
            R0, t0 = poses_opt[0]
            rvec0, _ = cv2.Rodrigues(R0)
            image_size = (self.videoLabel.pixmap().height(), self.videoLabel.pixmap().width()) if self.videoLabel.pixmap() is not None else (720, 1280)
            fx = K_opt[0, 0]
            fy = K_opt[1, 1]
            cx = K_opt[0, 2]
            cy = K_opt[1, 2]
            K_tuple = (fx, fy, cx, cy)
            packet = send_camera_stype(ip=ip, port=port, packet_no=self.packet_no,
                                       position=(float(t0[0]), float(t0[1]), float(t0[2])),
                                       rvec=rvec0.ravel(),
                                       image_size=image_size,
                                       K=K_tuple,
                                       broadcast=self.broadcastCheck.isChecked())
            self.log_msg(f"Sent STYPE packet to {ip}:{port} (packet_no={self.packet_no})")
            self.packet_no = (self.packet_no + 1) & 0xFF

            if continuous:
                bg = BackgroundSender(ip=ip, port=port, packet=packet, rate_hz=rate, broadcast=self.broadcastCheck.isChecked())
                bg.start()
                self._bg_sender = bg
                self.log_msg(f"Started continuous STYPE broadcast at {rate} Hz")

        # thread cleanup
        if self._calib_thread is not None:
            self._calib_thread.quit()
            self._calib_thread.wait()
            self._calib_thread = None
            self._calib_worker = None

    def _on_calib_error(self, msg):
        try:
            if self._progress_dialog is not None:
                self._progress_dialog.close()
                self._progress_dialog = None
        except Exception:
            pass
        self.log_msg(f"Calibration failed: {msg}")
        QtWidgets.QMessageBox.critical(self, "Calibration Failed",
                                       "Markerless calibration failed. Try capturing frames with more parallax and texture, or use a checkerboard/marker fallback.")
        if self._calib_thread is not None:
            self._calib_thread.quit()
            self._calib_thread.wait()
            self._calib_thread = None
            self._calib_worker = None

    def run_markerless_calib(self):
        n_frames, ok = QtWidgets.QInputDialog.getInt(self, "Frames", "Number of frames to capture (>=3):", 8, 3, 50, 1)
        if not ok:
            return

        frames = []
        if self.videoPath:
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

        # Prepare initial K
        h, w = frames[0].shape[:2]
        K0 = default_initial_K((h, w))

        # Stop any previous background sender
        self._stop_bg_sender()

        # Setup progress dialog (indeterminate)
        self._progress_dialog = QtWidgets.QProgressDialog("Running calibration...", "Cancel", 0, 0, self)
        self._progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
        self._progress_dialog.setWindowTitle("Markerless Calibration")
        self._progress_dialog.setMinimumDuration(0)
        self._progress_dialog.setAutoClose(False)
        self._progress_dialog.canceled.connect(self._on_progress_canceled)
        self._progress_dialog.show()

        # Create worker + thread
        self._calib_thread = QtCore.QThread(self)
        self._calib_worker = CalibrationWorker(frames, initial_K=K0, fxfy_equal=True, verbose=0)
        self._calib_worker.moveToThread(self._calib_thread)
        self._calib_thread.started.connect(self._calib_worker.run)
        self._calib_worker.progress.connect(self._on_worker_progress)
        self._calib_worker.finished.connect(self._on_calib_finished)
        self._calib_worker.error.connect(self._on_calib_error)
        # Ensure cleanup when thread finishes
        self._calib_thread.finished.connect(self._calib_worker.deleteLater)
        self._calib_thread.start()
        self.log_msg("Calibration started in background thread...")

    def _on_worker_progress(self, val):
        # If progress dialog is indeterminate (0,0) we don't setValue; keep it spinning.
        try:
            if self._progress_dialog is not None and self._progress_dialog.maximum() > 0:
                self._progress_dialog.setValue(int(val))
        except Exception:
            pass

    def _on_progress_canceled(self):
        # We cannot safely interrupt a running scipy least_squares call in this worker implementation.
        # Inform the user and allow the thread to finish; disable the UI cancel button.
        QtWidgets.QMessageBox.information(self, "Cancel requested",
                                          "Cancellation requested — the background calibration will finish its current iteration and then stop. This may still take some time.")
        # disable cancel button to avoid multiple clicks
        try:
            if self._progress_dialog:
                self._progress_dialog.setCancelButtonText("Cancelling...")
                self._progress_dialog.setEnabled(False)
        except Exception:
            pass

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
    return importlib.util.find_spec("torch") is not None

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
