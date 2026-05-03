"""Tkinter UI and pipeline orchestrator.
Simple UI: load video, intrinsics JSON, imu CSV (optional), start processing.
"""

import sys
import json
import time
import importlib.util
import base64
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
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

from pipeline.ba_autocalib import run_markerless_from_frames, default_initial_K
from pipeline.udp_stype import send_camera_stype, BackgroundSender

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

SETTINGS_PATH = Path("app_settings.json")
DEFAULT_SETTINGS = {
    "udp_ip": "127.0.0.1",
    "udp_port": 5555
}

BG_COLOR = "#121212"
FG_COLOR = "#F5F5F5"
SIDEBAR_BG = "#1B1D23"
PANEL_BG = "#181A20"
ENTRY_BG = "#252731"
BUTTON_BG = "#2F313B"
BUTTON_ACTIVE = "#3C3F4B"
LOG_BG = "#101214"
VIDEO_BORDER = "#2C3040"
HIGHLIGHT = "#5E9BF6"

BUTTON_FONT = ("Segoe UI", 9)
TITLE_FONT = ("Segoe UI", 13, "bold")
SECTION_FONT = ("Segoe UI", 11, "bold")


def normalize_intrinsics(intrinsics):
    if intrinsics is None:
        return None

    intr = dict(intrinsics)

    def _read_camera_matrix(cm):
        if isinstance(cm, dict):
            data = cm.get("data")
            if data is None:
                data = cm.get("matrix")
            if data is None and cm.get("rows") == 3 and cm.get("cols") == 3:
                data = cm.get("data")
        else:
            data = cm

        if isinstance(data, (list, tuple)) and len(data) == 9:
            return np.array(data, dtype=float).reshape((3, 3))
        if isinstance(data, (list, tuple)) and len(data) == 3 and all(
            isinstance(row, (list, tuple)) and len(row) == 3 for row in data
        ):
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


class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Cricket AR Tracker")
        self.root.configure(bg=BG_COLOR)
        self.root.geometry("1280x860")
        self.root.minsize(1100, 700)

        self.videoPath = None
        self.intrinsics = None
        self.imuPath = None
        self.udp_ip = DEFAULT_SETTINGS["udp_ip"]
        self.udp_port = DEFAULT_SETTINGS["udp_port"]
        self.packet_no = 0
        self._bg_sender = None
        self.photo = None
        self.log_overlay_visible = False
        self.sidebar_visible = True

        self._build_ui()
        self.load_settings()

    def _build_ui(self):
        title_frame = tk.Frame(self.root, bg=PANEL_BG, height=42)
        title_frame.pack(fill="x")
        tk.Label(
            title_frame,
            text="AR Tracker",
            bg=PANEL_BG,
            fg=FG_COLOR,
            font=TITLE_FONT,
            padx=16,
            pady=10,
        ).pack(side="left")

        self.toggle_panel_btn = tk.Button(
            title_frame,
            text="Hide Panel",
            command=self.toggle_sidebar,
            bg=BUTTON_BG,
            fg=FG_COLOR,
            activebackground=BUTTON_ACTIVE,
            activeforeground=FG_COLOR,
            relief="flat",
            padx=10,
            pady=8,
            font=BUTTON_FONT,
        )
        self.toggle_panel_btn.pack(side="right", padx=10, pady=4)

        content_frame = tk.Frame(self.root, bg=BG_COLOR)
        content_frame.pack(fill="both", expand=True, padx=10, pady=(10, 10))

        self.video_container = tk.Frame(content_frame, bg=VIDEO_BORDER, bd=1, relief="flat")
        self.video_container.pack(side="left", fill="both", expand=True)

        self.video_label = tk.Label(self.video_container, bg="#000000")
        self.video_label.pack(fill="both", expand=True)

        sidebar = tk.Frame(self.video_container, bg=SIDEBAR_BG, width=300)
        sidebar.place(relx=1.0, rely=0.02, anchor="ne", width=320, relheight=0.92)
        self.sidebar = sidebar
        sidebar.lift(aboveThis=self.video_label)

        self.log_overlay_frame = tk.Frame(self.video_container, bg=LOG_BG, bd=1, relief="solid")
        self.log_overlay_text = tk.Text(
            self.log_overlay_frame,
            bg=LOG_BG,
            fg=FG_COLOR,
            insertbackground=FG_COLOR,
            height=10,
            wrap="word",
            relief="flat",
        )
        self.log_overlay_text.pack(fill="both", expand=True)
        self.log_overlay_text.config(state="disabled")

        button_frame = tk.Frame(sidebar, bg=SIDEBAR_BG, width=280)
        button_frame.pack(fill="x", pady=(10, 6), padx=10)

        self.load_video_btn = self._make_button(button_frame, "Load Video", self.load_video)
        self.load_intr_btn = self._make_button(button_frame, "Load Intrinsics", self.load_intrinsics)
        self.load_imu_btn = self._make_button(button_frame, "Load IMU CSV", self.load_imu)
        self.calib_btn = self._make_button(button_frame, "Run Markerless Calibration", self.run_markerless_calib)
        self.run_btn = self._make_button(button_frame, "Run", self.run_pipeline)

        for btn in (
            self.load_video_btn,
            self.load_intr_btn,
            self.load_imu_btn,
            self.calib_btn,
            self.run_btn,
        ):
            btn.pack(fill="x", pady=6)

        ui_section = tk.Frame(sidebar, bg=SIDEBAR_BG)
        ui_section.pack(fill="x", pady=(20, 8), padx=12)
        tk.Label(ui_section, text="Video / UDP Settings", bg=SIDEBAR_BG, fg=FG_COLOR, font=SECTION_FONT).pack(anchor="w")

        self.toggle_log_btn = self._make_button(ui_section, "Show Logs", self.toggle_log_overlay)
        self.toggle_log_btn.pack(fill="x", pady=6)

        entry_frame = tk.Frame(ui_section, bg=SIDEBAR_BG)
        entry_frame.pack(fill="x", pady=4)
        tk.Label(entry_frame, text="UDP IP", bg=SIDEBAR_BG, fg=FG_COLOR).pack(anchor="w")
        self.udp_ip_var = tk.StringVar(value=self.udp_ip)
        self.udp_ip_entry = self._make_entry(entry_frame, self.udp_ip_var, width=24)
        self.udp_ip_entry.pack(fill="x", pady=(4, 10))

        tk.Label(entry_frame, text="Port", bg=SIDEBAR_BG, fg=FG_COLOR).pack(anchor="w")
        self.udp_port_var = tk.StringVar(value=str(self.udp_port))
        self.udp_port_entry = self._make_entry(entry_frame, self.udp_port_var, width=24)
        self.udp_port_entry.pack(fill="x", pady=(4, 10))

        self.save_udp_btn = self._make_button(ui_section, "Save UDP Settings", self.save_settings)
        self.save_udp_btn.pack(fill="x", pady=6)

        self.broadcast_var = tk.BooleanVar(value=False)
        self.continuous_var = tk.BooleanVar(value=False)
        self.rate_var = tk.IntVar(value=10)

        check_frame = tk.Frame(ui_section, bg=SIDEBAR_BG)
        check_frame.pack(fill="x", pady=(8, 8))
        tk.Checkbutton(
            check_frame,
            text="Enable UDP Broadcast",
            variable=self.broadcast_var,
            bg=SIDEBAR_BG,
            fg=FG_COLOR,
            selectcolor=SIDEBAR_BG,
            activebackground=SIDEBAR_BG,
            activeforeground=FG_COLOR,
            bd=0,
        ).pack(anchor="w")
        tk.Checkbutton(
            check_frame,
            text="Continuous",
            variable=self.continuous_var,
            bg=SIDEBAR_BG,
            fg=FG_COLOR,
            selectcolor=SIDEBAR_BG,
            activebackground=SIDEBAR_BG,
            activeforeground=FG_COLOR,
            bd=0,
        ).pack(anchor="w", pady=(4, 0))

        rate_frame = tk.Frame(ui_section, bg=SIDEBAR_BG)
        rate_frame.pack(fill="x", pady=(10, 0))
        tk.Label(rate_frame, text="Broadcast Rate", bg=SIDEBAR_BG, fg=FG_COLOR).pack(anchor="w")
        self.rate_spin = tk.Spinbox(
            rate_frame,
            from_=1,
            to=100,
            textvariable=self.rate_var,
            width=8,
            bg=ENTRY_BG,
            fg=FG_COLOR,
            insertbackground=FG_COLOR,
            relief="flat",
            justify="center",
        )
        self.rate_spin.pack(pady=(4, 0))

        self.status_label = tk.Label(sidebar, text="Ready", bg=SIDEBAR_BG, fg=FG_COLOR, anchor="w")
        self.status_label.pack(fill="x", side="bottom", pady=14, padx=12)

    def _make_button(self, parent, text, command):
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=BUTTON_BG,
            fg=FG_COLOR,
            activebackground=BUTTON_ACTIVE,
            activeforeground=FG_COLOR,
            relief="flat",
            padx=10,
            pady=8,
            font=BUTTON_FONT,
        )

    def _make_entry(self, parent, textvariable, width=20):
        return tk.Entry(
            parent,
            textvariable=textvariable,
            bg=ENTRY_BG,
            fg=FG_COLOR,
            insertbackground=FG_COLOR,
            width=width,
            relief="flat",
        )


    def log_msg(self, message):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {message}\n"
        self.status_label.config(text=message)

        self.log_overlay_text.config(state="normal")
        self.log_overlay_text.insert("end", line)
        self.log_overlay_text.see("end")
        self.log_overlay_text.config(state="disabled")
        print(message)

    def toggle_log_overlay(self):
        self.log_overlay_visible = not self.log_overlay_visible
        if self.log_overlay_visible:
            self.log_overlay_frame.place(relx=0.02, rely=0.02, relwidth=0.45, relheight=0.32)
            self.toggle_log_btn.config(text="Hide Logs")
        else:
            self.log_overlay_frame.place_forget()
            self.toggle_log_btn.config(text="Show Logs")

    def toggle_sidebar(self):
        self.sidebar_visible = not self.sidebar_visible
        if self.sidebar_visible:
            self.sidebar.place(relx=1.0, rely=0.02, anchor="ne", width=320, relheight=0.92)
            self.sidebar.lift()
            self.toggle_panel_btn.config(text="Hide Panel")
        else:
            self.sidebar.place_forget()
            self.toggle_panel_btn.config(text="Show Panel")

    def load_video(self):
        path = filedialog.askopenfilename(
            title="Open video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*")],
        )
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
        self.udp_ip_var.set(self.udp_ip)
        self.udp_port_var.set(str(self.udp_port))
        self.log_msg(f"Loaded UDP settings: {self.udp_ip}:{self.udp_port}")

    def save_settings(self):
        ip = self.udp_ip_var.get().strip() or DEFAULT_SETTINGS["udp_ip"]
        port_text = self.udp_port_var.get().strip()
        try:
            port = int(port_text)
            if port < 1 or port > 65535:
                raise ValueError("Port out of range")
        except Exception:
            messagebox.showwarning("Invalid Port", "Port must be an integer between 1 and 65535.")
            return
        self.udp_ip = ip
        self.udp_port = port
        settings = {"udp_ip": self.udp_ip, "udp_port": self.udp_port}
        try:
            with open(SETTINGS_PATH, "w") as f:
                json.dump(settings, f, indent=2)
            self.log_msg(f"Saved UDP settings: {self.udp_ip}:{self.udp_port}")
        except Exception as e:
            messagebox.showerror("Save Failed", f"Unable to save settings: {e}")

    def load_intrinsics(self):
        path = filedialog.askopenfilename(
            title="Open intrinsics JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*")],
        )
        if path:
            with open(path, "r") as f:
                raw = json.load(f)
            try:
                self.intrinsics = normalize_intrinsics(raw)
            except ValueError as e:
                messagebox.showerror("Invalid Intrinsics", str(e))
                return
            self.log_msg(f"Loaded intrinsics: {path}")

    def load_imu(self):
        path = filedialog.askopenfilename(
            title="Open IMU CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*")],
        )
        if path:
            self.imuPath = path
            self.log_msg(f"Loaded IMU CSV: {path}")

    def _stop_bg_sender(self):
        if self._bg_sender is not None:
            try:
                self._bg_sender.stop()
            except Exception:
                pass
            self._bg_sender = None

    def run_markerless_calib(self):
        n_frames = simpledialog.askinteger(
            "Frames", "Number of frames to capture (>=3):", initialvalue=8, minvalue=3, maxvalue=50
        )
        if n_frames is None:
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
            cam_id = simpledialog.askinteger("Camera", "Camera index:", initialvalue=0, minvalue=0, maxvalue=10)
            if cam_id is None:
                return
            cap = cv2.VideoCapture(cam_id)
            if not cap.isOpened():
                messagebox.showerror("Camera Error", f"Cannot open camera {cam_id}")
                return
            self.log_msg("Press SPACE to capture a frame, ESC to cancel.")
            while len(frames) < n_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                disp = frame.copy()
                cv2.putText(
                    disp,
                    f"Frame {len(frames)}/{n_frames}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )
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
            messagebox.showwarning("Not enough frames", "Captured fewer than 3 frames. Aborting calibration.")
            return

        self.log_msg("Running markerless bundle-adjustment autocalibration (this may take a while)...")
        try:
            h, w = frames[0].shape[:2]
            K0 = default_initial_K((h, w))
            K_opt, poses_opt, pts_opt, res = run_markerless_from_frames(
                frames, initial_K=K0, fxfy_equal=True, verbose=2
            )
        except Exception as e:
            self.log_msg(f"Calibration failed: {e}")
            messagebox.showerror(
                "Calibration Failed",
                "Markerless calibration failed. Try capturing frames with more parallax and texture, or use a checkerboard/marker fallback.",
            )
            return

        K_list = K_opt.reshape(-1).tolist()
        out = {
            "camera_matrix": {"rows": 3, "cols": 3, "data": K_list},
            "fx": float(K_opt[0, 0]),
            "fy": float(K_opt[1, 1]),
            "cx": float(K_opt[0, 2]),
            "cy": float(K_opt[1, 2]),
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "k3": 0.0,
        }
        out_path = "intrinsics_calibrated.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        self.intrinsics = normalize_intrinsics(out)
        self.log_msg(f"Calibration successful. Saved intrinsics to {out_path}")

        if self.broadcast_var.get() and len(poses_opt) > 0:
            self._stop_bg_sender()
            R0, t0 = poses_opt[0]
            rvec0, _ = cv2.Rodrigues(R0)
            image_size = (w, h)
            fx = K_opt[0, 0]
            fy = K_opt[1, 1]
            cx = K_opt[0, 2]
            cy = K_opt[1, 2]
            K_tuple = (fx, fy, cx, cy)
            try:
                packet = send_camera_stype(
                    ip=self.udp_ip,
                    port=self.udp_port,
                    packet_no=self.packet_no,
                    position=(float(t0[0]), float(t0[1]), float(t0[2])),
                    rvec=rvec0.ravel(),
                    image_size=image_size,
                    K=K_tuple,
                    broadcast=self.broadcast_var.get(),
                )
                self.log_msg(f"Sent STYPE packet to {self.udp_ip}:{self.udp_port} (packet_no={self.packet_no})")
                self.packet_no = (self.packet_no + 1) & 0xFF
                if self.continuous_var.get():
                    self._bg_sender = BackgroundSender(
                        ip=self.udp_ip,
                        port=self.udp_port,
                        packet=packet,
                        rate_hz=float(self.rate_var.get()),
                        broadcast=self.broadcast_var.get(),
                    )
                    self._bg_sender.start()
                    self.log_msg(f"Started continuous STYPE broadcast at {self.rate_var.get()} Hz")
            except Exception as e:
                self.log_msg(f"Failed to send STYPE packet: {e}")

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

        cap = cv2.VideoCapture(self.videoPath)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_idx = 0

        undistorter = Undistorter(intr)
        segmentation_device = "cuda" if torch_is_available() else "cpu"
        self.log_msg(f"Segmentation using device: {segmentation_device}")
        segmenter = FieldSegmenter(device=segmentation_device, use_torch=torch_installed_on_path())
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

            self._show_frame(ar_img)
            self.root.update()
            frame_idx += 1

        ar.close()
        cap.release()
        self.log_msg("Processing finished. Outputs saved to output/")

    def _show_frame(self, frame):
        if frame is None:
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            _, buffer = cv2.imencode(".png", rgb_frame)
            data = base64.b64encode(buffer).decode("ascii")
            self.photo = tk.PhotoImage(data=data, format="png")
        except Exception:
            _, buffer = cv2.imencode(".ppm", rgb_frame)
            self.photo = tk.PhotoImage(data=buffer.tobytes())

        self.video_label.configure(image=self.photo)
        self.video_label.image = self.photo


def torch_is_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def torch_installed_on_path():
    return importlib.util.find_spec("torch") is not None


if __name__ == "__main__":
    root = tk.Tk()
    root.state("zoomed")
    MainWindow(root)
    root.mainloop()
