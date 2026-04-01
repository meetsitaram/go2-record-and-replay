"""
Shared constants: Unitree button bitmasks, WebRTC topics, LeRobot feature schema,
safety rules, and gamepad configuration.
"""

# ── Unitree button bitmasks (from unitree_legged_sdk joystick.h) ──────────────

KEY_R1     = 0x0001
KEY_L1     = 0x0002
KEY_START  = 0x0004
KEY_SELECT = 0x0008
KEY_R2     = 0x0010
KEY_L2     = 0x0020
KEY_F1     = 0x0040
KEY_F2     = 0x0080
KEY_A      = 0x0100
KEY_B      = 0x0200
KEY_X      = 0x0400
KEY_Y      = 0x0800
KEY_UP     = 0x1000
KEY_RIGHT  = 0x2000
KEY_DOWN   = 0x4000
KEY_LEFT   = 0x8000

ALL_SHOULDERS = KEY_L1 | KEY_L2 | KEY_R1 | KEY_R2
ANY_FACE      = KEY_A  | KEY_B  | KEY_X  | KEY_Y

# evdev button codes -> Unitree bitmask
BUTTON_MAP = {
    304: KEY_A,       # BTN_SOUTH / BTN_A
    305: KEY_B,       # BTN_EAST  / BTN_B
    307: KEY_X,       # BTN_NORTH / BTN_X
    308: KEY_Y,       # BTN_WEST  / BTN_Y
    310: KEY_L1,      # BTN_TL  (LB)
    311: KEY_R1,      # BTN_TR  (RB)
    312: KEY_L2,      # BTN_TL2 (LT digital)
    313: KEY_R2,      # BTN_TR2 (RT digital)
    314: KEY_SELECT,  # BTN_SELECT (Back / View)
    315: KEY_START,   # BTN_START  (Menu)
    317: KEY_F1,      # BTN_THUMBL (Left stick click)
    318: KEY_F2,      # BTN_THUMBR (Right stick click)
}

# ── Safety: blocked combos and emergency stop ─────────────────────────────────
# (combo_mask, strip_mask, description)
BLOCKED_COMBOS = [
    (KEY_L2 | KEY_B, KEY_B, "Damp (LT+B) — motors off, robot collapses"),
    (KEY_R1 | KEY_A, KEY_A, "Jump forward (RB+A)"),
    (KEY_R1 | KEY_X, KEY_X, "Pounce (RB+X)"),
]

COUNTDOWN_SECS = 1.8
PULSE_TIMES    = [0.0, 0.6, 1.2]
PULSE_MS       = 250

# Known Go2 button-to-action mapping (longest combo first)
BUTTON_ACTIONS = [
    (KEY_L2 | KEY_A,       "Lock posture (stand/crouch toggle)"),
    (KEY_L2 | KEY_B,       "Damp (motors off)"),
    (KEY_L2 | KEY_X,       "Stand up from fall"),
    (KEY_L2 | KEY_SELECT,  "Searchlight toggle"),
    (KEY_R2 | KEY_A,       "Stretch"),
    (KEY_R2 | KEY_B,       "Shake hands"),
    (KEY_R2 | KEY_Y,       "Love"),
    (KEY_R1 | KEY_X,       "Pounce"),
    (KEY_R1 | KEY_A,       "Jump forward"),
    (KEY_R1 | KEY_B,       "Sit down"),
    (KEY_L1 | KEY_A,       "Greet"),
    (KEY_L1 | KEY_B,       "Dance"),
    (KEY_RIGHT | KEY_START, "Stair mode 1 (fwd up / bwd down)"),
    (KEY_LEFT | KEY_SELECT, "Stair mode 2 (fwd down)"),
    (KEY_L1 | KEY_SELECT,   "Endurance mode"),
    (KEY_START,             "Walking mode"),
    (KEY_SELECT,            "Standing mode"),
]

# ── Analog stick config ───────────────────────────────────────────────────────

STICK_CENTER    = 32768
STICK_DEADZONE  = 4096
STICK_RANGE     = 32768.0
TRIGGER_THRESHOLD = 300
TRIGGER_MAX     = 1023.0

# ── WebRTC data topics ────────────────────────────────────────────────────────

TOPIC_SPORTMODE  = "rt/lf/sportmodestate"
TOPIC_LOWSTATE   = "rt/lf/lowstate"
TOPIC_LIDAR_STATE = "rt/utlidar/lidar_state"
TOPIC_ROBOT_POSE = "rt/utlidar/robot_pose"

SEND_RATE = 0.05   # 20 Hz
DATASET_FPS = 20

# ── LeRobot feature schema ───────────────────────────────────────────────────

NUM_JOINTS = 12  # Go2: 4 legs x 3 joints (hip, thigh, calf)

JOINT_NAMES = [
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
]

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
