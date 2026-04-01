"""
WebRTC data stream subscribers with thread-safe latest-value buffers.

Each subscriber runs on the WebRTC data channel callback thread and writes
into a LatestValue buffer. The 20 Hz recording loop reads from these buffers
via snapshot().
"""

import logging
import threading
import time

import numpy as np

try:
    import av
    av.logging.set_level(av.logging.PANIC)
except ImportError:
    pass

logging.getLogger("libav").setLevel(logging.CRITICAL)

import cv2

from .constants import (
    TOPIC_SPORTMODE, TOPIC_LOWSTATE, TOPIC_ROBOT_POSE,
    NUM_JOINTS,
)


class LatestValue:
    """Thread-safe container that holds the most recent value and tracks freshness."""

    def __init__(self, default: np.ndarray):
        self._value = default.copy()
        self._default = default.copy()
        self._lock = threading.Lock()
        self._updated = False
        self._last_update: float = 0.0

    def set(self, value: np.ndarray):
        with self._lock:
            self._value = value
            self._updated = True
            self._last_update = time.monotonic()

    def get(self) -> tuple[np.ndarray, bool]:
        """Return (value, is_fresh). Clears the fresh flag after reading."""
        with self._lock:
            fresh = self._updated
            self._updated = False
            return self._value.copy(), fresh

    def peek(self) -> np.ndarray:
        """Return value without clearing freshness."""
        with self._lock:
            return self._value.copy()

    @property
    def age(self) -> float:
        """Seconds since last update, or inf if never updated."""
        with self._lock:
            if self._last_update == 0.0:
                return float("inf")
            return time.monotonic() - self._last_update


class LatestFrame:
    """Thread-safe container for the latest camera frame (numpy HWC uint8)."""

    def __init__(self):
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._count = 0

    def set(self, frame: np.ndarray):
        with self._lock:
            self._frame = frame
            self._count += 1

    def get(self) -> np.ndarray | None:
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    @property
    def count(self) -> int:
        with self._lock:
            return self._count


class RobotStreams:
    """
    Manages WebRTC subscriptions and latest-value buffers for all robot data streams.

    Call attach() after the WebRTC connection is established to start receiving data.
    Call snapshot() at 20 Hz to get a dict of the latest values for all streams.
    """

    def __init__(self, enable_camera: bool = True, enable_lidar: bool = True):
        self.enable_camera = enable_camera
        self.enable_lidar = enable_lidar

        # Sport mode state: [pos(3), vel(3), yaw_speed(1), imu_quat(4), foot_force(4), battery(1)]
        self.sport_state = LatestValue(np.zeros(16, dtype=np.float32))

        # Motor joint data from lowstate
        self.joint_positions = LatestValue(np.zeros(NUM_JOINTS, dtype=np.float32))
        self.joint_velocities = LatestValue(np.zeros(NUM_JOINTS, dtype=np.float32))
        self.joint_torques = LatestValue(np.zeros(NUM_JOINTS, dtype=np.float32))
        self.motor_temperatures = LatestValue(np.zeros(NUM_JOINTS, dtype=np.float32))
        self.power = LatestValue(np.zeros(4, dtype=np.float32))
        self._lowstate_fresh = False
        self._lowstate_lock = threading.Lock()

        # LiDAR robot pose: [x, y, z, roll, pitch, yaw]
        self.lidar_pose = LatestValue(np.zeros(6, dtype=np.float32))

        # Camera frame
        self.camera = LatestFrame()

        self._msg_counts: dict[str, int] = {}

    def attach(self, conn):
        """Subscribe to all data topics on the given WebRTC connection."""
        dc = conn.datachannel

        dc.pub_sub.subscribe(TOPIC_SPORTMODE, self._on_sportmode)
        dc.pub_sub.subscribe(TOPIC_LOWSTATE, self._on_lowstate)

        if self.enable_lidar:
            dc.pub_sub.subscribe(TOPIC_ROBOT_POSE, self._on_robot_pose)

        if self.enable_camera:
            try:
                conn.video.switchVideoChannel(True)
            except Exception:
                pass
            conn.video.add_track_callback(self._on_video_track)

        # Enable lidar data channel traffic
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            asyncio.run_coroutine_threadsafe(
                dc.disableTrafficSaving(True), loop
            )
        except Exception:
            pass

    def _on_sportmode(self, msg: dict):
        """Parse rt/lf/sportmodestate JSON payload."""
        self._msg_counts["sportmode"] = self._msg_counts.get("sportmode", 0) + 1
        try:
            d = msg.get("data", msg)
            if isinstance(d, str):
                import json
                d = json.loads(d)

            state = np.zeros(16, dtype=np.float32)

            if "position" in d:
                pos = d["position"]
                if isinstance(pos, dict):
                    state[0] = pos.get("x", 0.0)
                    state[1] = pos.get("y", 0.0)
                    state[2] = pos.get("z", 0.0)
                elif isinstance(pos, list) and len(pos) >= 3:
                    state[0:3] = pos[:3]

            if "velocity" in d:
                vel = d["velocity"]
                if isinstance(vel, dict):
                    state[3] = vel.get("x", 0.0)
                    state[4] = vel.get("y", 0.0)
                    state[5] = vel.get("z", 0.0)
                elif isinstance(vel, list) and len(vel) >= 3:
                    state[3:6] = vel[:3]

            state[6] = float(d.get("yaw_speed", d.get("yawSpeed", 0.0)))

            imu = d.get("imu_state", d.get("imuState", {}))
            if isinstance(imu, dict):
                quat = imu.get("quaternion", [1, 0, 0, 0])
                if isinstance(quat, list) and len(quat) >= 4:
                    state[7:11] = quat[:4]

            ff = d.get("foot_force", d.get("footForce", []))
            if isinstance(ff, list) and len(ff) >= 4:
                state[11:15] = [float(f) for f in ff[:4]]

            state[15] = float(d.get("battery_level", d.get("batteryLevel", 0.0)))

            self.sport_state.set(state)
        except Exception:
            pass

    def _on_lowstate(self, msg: dict):
        """Parse rt/lf/lowstate JSON payload for motor states and power."""
        self._msg_counts["lowstate"] = self._msg_counts.get("lowstate", 0) + 1
        try:
            d = msg.get("data", msg)
            if isinstance(d, str):
                import json
                d = json.loads(d)

            motor_states = d.get("motor_state", d.get("motorState", []))
            if isinstance(motor_states, list) and len(motor_states) >= NUM_JOINTS:
                q = np.zeros(NUM_JOINTS, dtype=np.float32)
                dq = np.zeros(NUM_JOINTS, dtype=np.float32)
                tau = np.zeros(NUM_JOINTS, dtype=np.float32)
                temp = np.zeros(NUM_JOINTS, dtype=np.float32)

                for i in range(NUM_JOINTS):
                    m = motor_states[i]
                    if isinstance(m, dict):
                        q[i] = m.get("q", 0.0)
                        dq[i] = m.get("dq", 0.0)
                        tau[i] = m.get("tau_est", m.get("tauEst", 0.0))
                        temp[i] = m.get("temperature", 0.0)

                self.joint_positions.set(q)
                self.joint_velocities.set(dq)
                self.joint_torques.set(tau)
                self.motor_temperatures.set(temp)

            pwr = np.zeros(4, dtype=np.float32)
            pwr[0] = float(d.get("power_v", d.get("powerV", 0.0)))
            pwr[1] = float(d.get("power_a", d.get("powerA", 0.0)))

            bms = d.get("bms_state", d.get("bmsState", {}))
            if isinstance(bms, dict):
                pwr[2] = float(bms.get("soc", bms.get("SOC", 0.0)))
                pwr[3] = float(bms.get("current", 0.0))

            self.power.set(pwr)

            with self._lowstate_lock:
                self._lowstate_fresh = True

        except Exception:
            pass

    def _on_robot_pose(self, msg: dict):
        """Parse rt/utlidar/robot_pose JSON payload."""
        self._msg_counts["robot_pose"] = self._msg_counts.get("robot_pose", 0) + 1
        try:
            d = msg.get("data", msg)
            if isinstance(d, str):
                import json
                d = json.loads(d)

            pose_data = np.zeros(6, dtype=np.float32)

            # robot_pose uses: data.pose.position and data.pose.orientation
            pose_wrapper = d.get("pose", d)
            pos = pose_wrapper.get("position", {})
            if isinstance(pos, dict):
                pose_data[0] = pos.get("x", 0.0)
                pose_data[1] = pos.get("y", 0.0)
                pose_data[2] = pos.get("z", 0.0)

            ori = pose_wrapper.get("orientation", {})
            if isinstance(ori, dict) and any(k in ori for k in ("w", "x", "y", "z")):
                import math
                w = ori.get("w", 1.0)
                x = ori.get("x", 0.0)
                y = ori.get("y", 0.0)
                z = ori.get("z", 0.0)
                sinr = 2.0 * (w * x + y * z)
                cosr = 1.0 - 2.0 * (x * x + y * y)
                pose_data[3] = math.atan2(sinr, cosr)
                sinp = 2.0 * (w * y - z * x)
                pose_data[4] = math.asin(max(-1.0, min(1.0, sinp)))
                siny = 2.0 * (w * z + x * y)
                cosy = 1.0 - 2.0 * (y * y + z * z)
                pose_data[5] = math.atan2(siny, cosy)

            self.lidar_pose.set(pose_data)
        except Exception:
            pass

    async def _on_video_track(self, track):
        """Receive video frames from the WebRTC video track."""
        while True:
            try:
                frame = await track.recv()
                # Use native yuv420p -> OpenCV RGB to avoid swscaler stderr spam
                yuv = frame.to_ndarray()
                img = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)
                self.camera.set(img)
            except Exception:
                break

    def snapshot(self, include_camera: bool = True, include_lidar: bool = True) -> dict:
        """
        Return a dict of all latest values suitable for dataset.add_frame().
        All numpy arrays are freshly copied. Camera frame is in HWC uint8 format.
        """
        sport, _ = self.sport_state.get()
        jpos, _ = self.joint_positions.get()
        jvel, _ = self.joint_velocities.get()
        jtau, _ = self.joint_torques.get()
        mtemp, _ = self.motor_temperatures.get()
        pwr, _ = self.power.get()

        with self._lowstate_lock:
            lowstate_fresh = self._lowstate_fresh
            self._lowstate_fresh = False

        result = {
            "observation.state": sport,
            "observation.joint_positions": jpos,
            "observation.joint_velocities": jvel,
            "observation.joint_torques": jtau,
            "observation.motor_temperatures": mtemp,
            "observation.power": pwr,
            "observation.lowstate_fresh": np.array(
                [1.0 if lowstate_fresh else 0.0], dtype=np.float32
            ),
        }

        if include_lidar and self.enable_lidar:
            lpose, _ = self.lidar_pose.get()
            result["observation.lidar_pose"] = lpose

        if include_camera and self.enable_camera:
            frame = self.camera.get()
            if frame is not None:
                result["observation.images.front"] = frame

        return result

    def print_status(self):
        """Print a summary of received message counts."""
        for topic, count in sorted(self._msg_counts.items()):
            print(f"  {topic}: {count} messages")
        if self.enable_camera:
            print(f"  camera: {self.camera.count} frames")
