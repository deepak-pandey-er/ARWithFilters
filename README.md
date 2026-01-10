# Markerless Drone Camera Tracking — Cricket AR TV (Open-source, Python)

Overview
--------
This project provides a full pipeline to convert drone video + IMU telemetry into a stable, ground-locked camera pose stream ready for AR overlays and Unreal import. It was implemented with broadcast workflows in mind: camera intrinsics, field geometry anchoring (wicket-to-wicket), IMU-aware orientation stabilization, and a smoothing stage for broadcast-quality output.

Features
- Markerless camera tracking
- Drone-friendly (accepts typical drone video + IMU CSV)
- IMU-aware orientation stabilization
- Ground-locked: uses cricket pitch geometry (wicket-to-wicket distance) to recover scale & world alignment
- Zoom-robust: works across focal changes if intrinsics/zoom metadata are provided
- Unreal-ready pose output (CSV: timestamp, translation, quaternion)

Requirements
------------
- Python 3.9+
- OpenCV (cv2)
- torch, torchvision
- numpy, scipy
- PyQt5
- pyquaternion (for quaternion utilities)
- scikit-image (optional, nicer morphological ops)

Install
-------
Create venv and install:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Model downloads
---------------
The code uses torchvision's DeeplabV3 pretrained segmentation model. No extra manual download is needed: torchvision will download the model on first run (requires internet).

If you have a more accurate segmentation model for cricket fields, replace pipeline/segmentation.py to load your model and the corresponding pre/post-processing.

Usage
-----
- Prepare a video file (drone-recorded MP4).
- Prepare a CSV IMU log (timestamp(s), ax,ay,az, gx,gy,gz). Timestamps must be in seconds. If no IMU provided, orientation will be computed visually only.
- Prepare camera intrinsics (fx, fy, cx, cy, k1,k2,p1,p2,k3) in a JSON or via the UI.
- Run:

```bash
python app.py
```

In the UI:
- Load video
- Load IMU CSV (optional)
- Load camera intrinsics (JSON) or enter manually
- Press "Run" to process. The preview will show segmentation, detected pitch centerline, and AR overlay. The output pose CSV will be saved to `output/poses.csv`.

Files
-----
- app.py — PyQt5 UI and pipeline orchestrator
- pipeline/undistort.py — apply intrinsics undistortion
- pipeline/segmentation.py — field segmentation (Deeplab)
- pipeline/lines.py — pitch centerline / wicket detection (Hough + morphological)
- pipeline/visual_odometry.py — ORB-based monocular visual odometry (relative poses)
- pipeline/imu.py — IMU parsing + complementary orientation filter
- pipeline/ground_lock.py — RANSAC ground plane + solvePnP alignment using known wickets
- pipeline/kalman.py — pose smoothing
- pipeline/ar_output.py — project 3D overlays and save poses
- utils/geometry.py — quaternion & transform helpers

Limitations & next steps
------------------------
- Segmentation quality depends on model; fine-tuning on cricket fields is recommended.
- Monocular VO scale is estimated from pitch geometry; occlusion or poor detection of pitch lines will reduce accuracy.
- For production, use high-rate IMU telemetry tightly synchronized with frames (using timestamps).
- Replace Deeplab with a lightweight real-time segmentation model if running on embedded or resource-limited hardware.
- Consider integrating an existing VIO engine (OpenVINS, ORB-SLAM3 with IMU) for stronger performance.

License
-------
MIT
