"""
Utilities to build and send STYPE binary packets over UDP according to the STYPE package format provided.

Packet layout (byte offsets):
0:  Header (1 byte) - const 0x0F
1:  Commands (1 byte) - unsigned char
2:  Timecode (3 bytes) - unsigned int24 little-endian
5:  Packet_no (1 byte) - unsigned char
6:  X (4 bytes) float little-endian - +Right (meters)
10: Y (4) float - +Up (meters)
14: Z (4) float - -Look (meters)
18: Pan (4) float degrees - +Pan to the right
22: Tilt (4) float degrees - +Tilt up
26: Roll (4) float degrees - +Roll clockwise
30: FovX (4) float degrees
34: Aspect_Ratio (4) float
38: Focus (4) float - 0-close, 1-far
42: Zoom (4) float - 0-wide, 1-tele
46: k1 (4) float little-endian - radial distortion k1 (mm^-2)
50: k2 (4) float - radial distortion k2 (mm^-4)
54: Center_X (4) float - horizontal center shift (mm)
58: Center_Y (4) float - vertical center shift (mm)
62: PA_width (4) float - projection area width (mm)
66: Checksum (1 byte) - unsigned sum of preceding bytes modulo 256

Total packet size: 67 bytes

This module provides helpers to build/compute the packet from computed intrinsics/extrinsics and send it via UDP,
plus a simple background sender for continuous broadcast.
"""

import struct
import socket
from typing import Tuple
import math

PACKET_SIZE = 67
HEADER_CONST = 0x0F

def _pack_uint24_le(value: int) -> bytes:
    """Pack a 24-bit unsigned integer little-endian into 3 bytes."""
    value = int(value) & 0xFFFFFF
    return bytes([value & 0xFF, (value >> 8) & 0xFF, (value >> 16) & 0xFF])

def build_stype_packet(packet_no: int,
                       position: Tuple[float, float, float],
                       pan_deg: float,
                       tilt_deg: float,
                       roll_deg: float,
                       fovx_deg: float,
                       aspect_ratio: float,
                       focus: float = 1.0,
                       zoom: float = 0.0,
                       k1: float = 0.0,
                       k2: float = 0.0,
                       center_x_mm: float = 0.0,
                       center_y_mm: float = 0.0,
                       pa_width_mm: float = 0.0,
                       commands: int = 0,
                       timecode: int = 0) -> bytes:
    """
    Build a 67-byte STYPE packet.
    Arguments are mapped to the STYPE fields described above.
    """
    buf = bytearray()
    buf.append(HEADER_CONST & 0xFF)
    buf.append(commands & 0xFF)
    buf.extend(_pack_uint24_le(timecode))
    buf.append(packet_no & 0xFF)

    x, y, z = position
    floats = (x, y, z, pan_deg, tilt_deg, roll_deg, fovx_deg, aspect_ratio,
              focus, zoom, k1, k2, center_x_mm, center_y_mm, pa_width_mm)
    for val in floats:
        buf.extend(struct.pack('<f', float(val)))

    # checksum: sum of all preceding bytes modulo 256
    checksum = sum(buf) & 0xFF
    buf.append(checksum)

    # Safety: ensure exact length
    if len(buf) != PACKET_SIZE:
        if len(buf) < PACKET_SIZE:
            buf.extend(b'\x00' * (PACKET_SIZE - len(buf)))
        else:
            buf = buf[:PACKET_SIZE]
    return bytes(buf)

def send_stype_packet(ip: str, port: int, packet: bytes, broadcast: bool = False) -> None:
    """Send the given packet bytes to ip:port over UDP. If broadcast=True, enable socket broadcast."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        if broadcast:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.sendto(packet, (ip, port))
    finally:
        sock.close()

class BackgroundSender:
    """Simple background sender that repeatedly sends the same STYPE packet at a given rate."""
    def __init__(self, ip: str, port: int, packet: bytes, rate_hz: float = 10.0, broadcast: bool = False):
        import threading
        self.ip = ip
        self.port = port
        self.packet = packet
        self.rate = max(0.1, float(rate_hz))
        self.broadcast = broadcast
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._stop_event.clear()
        if not self._thread.is_alive():
            import threading as _th
            self._thread = _th.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _run(self):
        import time
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            if self.broadcast:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            while not self._stop_event.is_set():
                try:
                    sock.sendto(self.packet, (self.ip, self.port))
                except Exception:
                    pass
                time.sleep(1.0 / self.rate)
        finally:
            sock.close()

# Convenience helper that builds a packet from intrinsics/pose and optionally sends it.
def send_camera_stype(ip: str,
                      port: int,
                      packet_no: int,
                      position: Tuple[float, float, float],
                      rvec: Tuple[float, float, float],
                      image_size: Tuple[int, int],
                      K: Tuple[float, float, float, float],
                      broadcast: bool = False) -> bytes:
    """
    Build a STYPE packet from camera parameters and send it. Returns the raw packet.
    - position: (x,y,z) in meters
    - rvec: Rodrigues rotation vector (3,)
    - image_size: (height, width)
    - K: (fx, fy, cx, cy)
    The helper computes pan/tilt/roll (deg) from rotation and fovx/aspect from K and image size.
    """
    # Convert rvec to Euler angles in degrees
    import cv2 as _cv2
    R, _ = _cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        rx = math.degrees(math.atan2(R[2, 1], R[2, 2]))
        ry = math.degrees(math.atan2(-R[2, 0], sy))
        rz = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    else:
        rx = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
        ry = math.degrees(math.atan2(-R[2, 0], sy))
        rz = 0.0

    roll_deg = rx
    tilt_deg = ry
    pan_deg = rz

    h, w = image_size
    fx, fy, cx, cy = K
    try:
        fovx_rad = 2.0 * math.atan((w / 2.0) / fx)
        fovx_deg = math.degrees(fovx_rad)
    except Exception:
        fovx_deg = 60.0
    aspect = float(w) / float(h) if h != 0 else 1.0

    packet = build_stype_packet(packet_no=packet_no,
                                position=position,
                                pan_deg=pan_deg,
                                tilt_deg=tilt_deg,
                                roll_deg=roll_deg,
                                fovx_deg=fovx_deg,
                                aspect_ratio=aspect,
                                focus=1.0,
                                zoom=0.0,
                                k1=0.0,
                                k2=0.0,
                                center_x_mm=0.0,
                                center_y_mm=0.0,
                                pa_width_mm=0.0,
                                commands=0,
                                timecode=0)
    send_stype_packet(ip, port, packet, broadcast=broadcast)
    return packet
