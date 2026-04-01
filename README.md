# go2-record-and-replay

Record and replay teleoperation data for the **Unitree Go2 Air** quadruped robot,
stored natively in [LeRobot v3.0](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) dataset format.

## Go2 Air limitations

The Go2 Air does **not** support Unitree's secondary development SDK (EDU-only).
There is no access to CycloneDDS topics, no low-level motor control (`LowCmd`),
and no high-frequency `/lowstate` (500 Hz). All data is collected over **WebRTC**
using `unitree_webrtc_connect`, which is the only interface available on the Air.

This means some data streams are limited compared to EDU/Pro models:

- **Motor joint state** (`rt/lf/lowstate`) arrives at only **~1 Hz** (vs. 500 Hz on EDU via DDS)
- **Sport mode state** must use `rt/lf/sportmodestate` (the non-`lf` topic is silent on Air)
- **Controller injection** goes through the WebRTC data channel (not DDS)
- **Camera** is ~12 fps H.264 over WebRTC (not direct RTSP)

We record everything that **is** accessible and make the best of it. If you later
upgrade to a Go2 EDU, the dataset schema stays the same -- you'd just get higher
rate lowstate data and the freshness flag would be 1.0 on nearly every frame.

## What it does

- **Record**: Teleoperate the Go2 with an Xbox controller while simultaneously
  capturing all accessible data streams (controls, robot state, camera, LiDAR
  pose, motor joints) into a LeRobot dataset at 20 Hz.
- **Replay**: Send recorded actions back to the robot at original timing.
- **Train**: Datasets are directly usable with LeRobot-compatible training
  pipelines (ACT, Diffusion Policy, pi0, etc.) and can be pushed to HF Hub.

## Data captured

| Feature | Source | Native rate | Shape |
|---------|--------|-------------|-------|
| `action` | Xbox gamepad (evdev) | 20 Hz | [4] lx, ly, rx, ry |
| `action.buttons` | Xbox gamepad (evdev) | 20 Hz | [1] bitmask |
| `observation.state` | `rt/lf/sportmodestate` | ~20 Hz | [16] pos, vel, IMU, foot forces |
| `observation.joint_positions` | `rt/lf/lowstate` | ~1 Hz | [12] joint angles (rad) |
| `observation.joint_velocities` | `rt/lf/lowstate` | ~1 Hz | [12] joint velocities (rad/s)* |
| `observation.joint_torques` | `rt/lf/lowstate` | ~1 Hz | [12] estimated torques (Nm)* |
| `observation.motor_temperatures` | `rt/lf/lowstate` | ~1 Hz | [12] motor temps (C) |
| `observation.power` | `rt/lf/lowstate` | ~1 Hz | [4] voltage, current, battery |
| `observation.lowstate_fresh` | derived | 20 Hz | [1] 1.0 = new lowstate data |
| `observation.lidar_pose` | `rt/utlidar/robot_pose` | ~19 Hz | [6] x,y,z,r,p,y |
| `observation.images.front` | WebRTC video | ~12 fps | [3, 720, 1280] |

All streams are sample-and-held to the 20 Hz master clock. Low-rate data
(lowstate at ~1 Hz) includes a freshness flag for training-time masking.

*\* Go2 Air's lowstate only provides `q` (position) and `temperature`.
Velocities (`dq`) and torques (`tau_est`) are not present in the Air's
WebRTC messages — these columns will be zero. EDU models provide all fields.*

## Setup

```bash
cd go2-record-and-replay
uv venv --python 3.12
uv pip install -e .
```

> **Note:** If `opencv-python-headless` gets installed (pulled by lerobot), remove it
> so the GUI-capable `opencv-python` is used instead:
> `uv pip uninstall opencv-python-headless`

## Quick start

```bash
# 1. Record an episode
.venv/bin/python scripts/record.py --mode sta --ip 192.168.1.133 \
    --repo-id go2-walk --task "walk forward"

# 2. Visualize the recording (renders an annotated MP4)
.venv/bin/python scripts/visualize.py --dataset ./data/go2-walk

# 3. Replay on the robot
.venv/bin/python scripts/replay.py --dataset ./data/go2-walk --episode 0 \
    --mode sta --ip 192.168.1.133
```

## Usage

### Record

```bash
# Record via WiFi STA mode (robot on same router)
.venv/bin/python scripts/record.py --mode sta --ip 192.168.1.133 \
    --repo-id go2-teleop --task "walk around"

# Dry run (gamepad only, no robot connection or recording)
.venv/bin/python scripts/record.py --dry-run

# Without camera or lidar
.venv/bin/python scripts/record.py --mode sta --ip 192.168.1.133 --no-camera --no-lidar

# AP mode (connected to Go2's own hotspot)
.venv/bin/python scripts/record.py --mode ap --repo-id go2-teleop
```

**Controls during recording:**
- Left stick: walk / strafe
- Right stick: yaw / look
- Start: walking mode
- Select: standing mode
- **F1 (left stick click): toggle recording -- saves episode on stop**
- Ctrl+C: finalize dataset and exit

### Visualize

Renders the camera feed with overlaid joystick positions, button actions (with
fade-out), robot state, lidar pose, joint angles, and a progress bar into an MP4.

```bash
# Visualize episode 0 (default)
.venv/bin/python scripts/visualize.py --dataset ./data/go2-teleop

# Specific episode, custom output path
.venv/bin/python scripts/visualize.py --dataset ./data/go2-teleop --episode 1 -o viz.mp4
```

### Replay

Sends recorded actions back to the robot at the original 20 Hz timing.

```bash
# Replay on robot
.venv/bin/python scripts/replay.py --dataset ./data/go2-teleop --episode 0 \
    --mode sta --ip 192.168.1.133

# Dry run (print actions without sending)
.venv/bin/python scripts/replay.py --dataset ./data/go2-teleop --episode 0 --dry-run

# Half speed replay
.venv/bin/python scripts/replay.py --dataset ./data/go2-teleop --episode 0 --speed 0.5
```

### Inspect dataset

```bash
.venv/bin/python scripts/inspect_episode.py --dataset ./data/go2-teleop
.venv/bin/python scripts/inspect_episode.py --dataset ./data/go2-teleop --episode 0 --plot
```

## Safety

All safety rules from the Go2 Xbox controller are preserved:
- Dangerous combos (Damp, Jump, Pounce) are blocked by default
- `--allow-all` enables them with a 3-vibration countdown
- Emergency stop: hold LB+LT+RB+RT + any face button
- Speed limiting via `--speed-limit` (default 50%)
