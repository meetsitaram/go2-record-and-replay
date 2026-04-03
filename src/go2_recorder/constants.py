"""
LeRobot dataset constants: feature schema, dataset FPS, and helpers.

Robot-level constants (buttons, topics, joints, etc.) live in go2_driver.constants.
"""

from go2_driver.constants import NUM_JOINTS, JOINT_NAMES

# ── Dataset config ───────────────────────────────────────────────────────────

DATASET_FPS = 20

# ── LeRobot feature schema ───────────────────────────────────────────────────

LEROBOT_FEATURES = {
    "action": {
        "dtype": "float32",
        "shape": [4],
        "names": ["lx", "ly", "rx", "ry"],
    },
    "action.buttons": {
        "dtype": "int32",
        "shape": [1],
        "names": ["keys_bitmask"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": [16],
        "names": [
            "pos_x", "pos_y", "pos_z",
            "vel_x", "vel_y", "vel_z",
            "yaw_speed",
            "imu_quat_w", "imu_quat_x", "imu_quat_y", "imu_quat_z",
            "foot_force_0", "foot_force_1", "foot_force_2", "foot_force_3",
            "battery_level",
        ],
    },
    "observation.joint_positions": {
        "dtype": "float32",
        "shape": [NUM_JOINTS],
        "names": [f"{j}_q" for j in JOINT_NAMES],
    },
    "observation.joint_velocities": {
        "dtype": "float32",
        "shape": [NUM_JOINTS],
        "names": [f"{j}_dq" for j in JOINT_NAMES],
    },
    "observation.joint_torques": {
        "dtype": "float32",
        "shape": [NUM_JOINTS],
        "names": [f"{j}_tau" for j in JOINT_NAMES],
    },
    "observation.motor_temperatures": {
        "dtype": "float32",
        "shape": [NUM_JOINTS],
        "names": [f"{j}_temp" for j in JOINT_NAMES],
    },
    "observation.power": {
        "dtype": "float32",
        "shape": [4],
        "names": ["voltage_v", "current_a", "battery_soc", "battery_current_ma"],
    },
    "observation.lowstate_fresh": {
        "dtype": "float32",
        "shape": [1],
        "names": ["is_fresh"],
    },
    "observation.lidar_pose": {
        "dtype": "float32",
        "shape": [6],
        "names": ["x", "y", "z", "roll", "pitch", "yaw"],
    },
    "observation.images.front": {
        "dtype": "video",
        "shape": [3, 720, 1280],
    },
}


def features_without_camera():
    """Return feature schema with camera excluded."""
    return {k: v for k, v in LEROBOT_FEATURES.items() if k != "observation.images.front"}


def features_without_lidar():
    """Return feature schema with lidar_pose excluded."""
    return {k: v for k, v in LEROBOT_FEATURES.items() if k != "observation.lidar_pose"}
