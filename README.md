# go2-record-and-replay

Record and replay teleoperation data for the **Unitree Go2** (Air & Pro) quadruped robot,
stored natively in [LeRobot v3.0](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) dataset format.

<p align="center">
  <i>(screenshot removed — see cheatsheet.md for controller reference)</i>
</p>

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
- **Replay**: Send recorded actions back to the robot at original timing, with
  optional music sync and closed-loop position hold during balancing modes.
- **Multi-robot choreography**: Replay episodes on multiple robots simultaneously
  with D-pad controller takeover for live repositioning during shows.
- **Pro/Air compatibility**: Automatically substitutes Pro-only moves (Handstand,
  Erect) with Air-compatible alternatives during multi-robot playback.
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

This project depends on the sibling repo [`go2-driver`](https://github.com/meetsitaram/go2-driver),
which is referenced as a path-dependency (`../go2-driver`) — **not** a git submodule.
You clone the two repos as siblings under one parent directory:

```
go-explore/                  ← any parent dir name is fine
├── go2-driver/              ← shared driver (gamepad, WebRTC, streams)
└── go2-record-and-replay/   ← this repo
```

### 1. Prerequisites

#### Common (all platforms)

- **Python 3.12** — required (CUDA wheels on Jetson are cp312 only; LeRobot also
  pins newer ranges; older Pythons will resolve incorrectly).
- **[uv](https://docs.astral.sh/uv/)** — the package manager this project uses.
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **git** and **ffmpeg** — used at runtime for music playback (`ffplay`) and the
  visualizer's video encoder (`ffmpeg`).
- An **Xbox-compatible USB or Bluetooth gamepad** for teleop / takeover. The
  driver uses Linux's `evdev`, so on macOS the gamepad path is read-only and you
  cannot record (replay/choreo still works, just no joystick input).

#### Ubuntu / Debian (Jetson, x86_64, etc.)

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv git ffmpeg \
                    libevdev-dev libudev-dev pkg-config build-essential

# Optional (only if you want to AirPlay music to a Sonos / Apple TV during shows)
sudo apt install -y avahi-daemon avahi-utils pipewire pipewire-pulse wireplumber

# Optional but useful for development debugging
sudo apt install -y arp-scan iputils-ping
```

> **Gamepad permissions on Linux:** to read `/dev/input/event*` without `sudo`,
> add yourself to the `input` group once: `sudo usermod -aG input $USER`,
> then log out/in.

#### macOS (Apple Silicon or Intel)

```bash
# Homebrew (skip if already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install python@3.12 git ffmpeg uv
```

> **macOS limitations:**
> - `evdev` is Linux-only. The `record.py` script will fail at gamepad init on Mac.
> - `replay.py`, `replay_teleop.py`, `choreo_multi.py` work fine on Mac for replay
>   and visualization (no gamepad input needed if you let the show run unattended,
>   or use D-pad-only via `pygame` if you swap the gamepad backend).
> - For development on Mac, the typical workflow is: develop and visualize on Mac,
>   record/replay on a Linux host (the Jetson, an x86 laptop, or a Pi).

#### Jetson Thor / NVIDIA Jetson

This project runs on JetPack 7 (CUDA 13 / aarch64-sbsa). The dependencies above
are enough for the choreo runner; if you also want the LeRobot training side
(GPU PyTorch), the project ships a `[[tool.uv.index]]` entry pointing at
`https://pypi.jetson-ai-lab.io/sbsa/cu130/+simple/` so `uv sync` resolves the
correct CUDA wheels automatically. Set
`TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` if you run vLLM-adjacent code.

### 2. Clone

```bash
mkdir -p ~/projects/go-explore && cd ~/projects/go-explore
git clone https://github.com/meetsitaram/go2-driver.git
git clone https://github.com/meetsitaram/go2-record-and-replay.git
```

The two repos must end up as siblings; otherwise the `path = "../go2-driver"`
reference in `pyproject.toml` won't resolve.

### 3. Create the venv and install

The fastest path is the bundled bootstrap script, which handles steps 1–3
above plus the venv, dependency install, and (on Linux) the `input`-group
setup:

```bash
cd go2-record-and-replay
./install.sh
```

Pass `--no-system` to skip apt/brew, `--no-clone` if `../go2-driver` is
already in place, or `--no-input` to skip touching group membership.

If you'd rather do it manually:

```bash
cd go2-record-and-replay
uv venv --python 3.12
uv sync
```

`uv sync` installs both this package and `go2-driver` (in editable mode) plus
all dependencies. The first run takes a few minutes (it pulls LeRobot, PyArrow,
etc.).

> **Note on opencv:** if `opencv-python-headless` ends up installed (LeRobot
> sometimes pulls it transitively), uninstall it so the GUI-capable
> `opencv-python` from `go2-driver` is used:
> ```bash
> uv pip uninstall opencv-python-headless
> ```

### 4. Verify the install

```bash
.venv/bin/python -c "from go2_driver.gamepad import find_gamepad; print('driver OK')"
.venv/bin/python -c "import lerobot, pandas, numpy; print('deps OK')"
.venv/bin/python scripts/record.py --dry-run    # prints gamepad menu, no robot needed
```

If `find_gamepad()` returns `None` on Linux, your user probably isn't in the
`input` group yet (see prerequisites above) or the controller isn't paired.

### 5. (Optional) AirPlay music output to a Sonos / Apple TV

If you want the show's music to play through a Sonos speaker (Beam Gen 2,
Arc, etc.) on the network rather than the local sound card, enable PipeWire's
RAOP discovery once:

```bash
mkdir -p ~/.config/pipewire/pipewire.conf.d
cat > ~/.config/pipewire/pipewire.conf.d/30-raop-discover.conf <<'EOF'
context.modules = [
    { name = libpipewire-module-raop-discover
      args = { raop.latency.ms = 1000 } }
]
EOF
systemctl --user restart pipewire pipewire-pulse wireplumber
```

Confirm the speaker shows up in the sink list (`wpctl status`) and select it
as default with `wpctl set-default <id>`. The choreo runner then plays music
through whatever the system default sink is.

### 6. (Optional) Robot AES key

Go2 robots running firmware ≥ 1.1.15 require a per-device AES-128 key for the
LAN WebRTC handshake. See [Robot encryption key setup](#robot-encryption-key-setup)
below for how to fetch it. Older robots don't need a key.

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

### Multi-robot replay

Replay the same episode simultaneously on multiple robots. Robots are defined
in a YAML config file. Before replay starts, each robot is commanded to stand
and the body height is verified.

```bash
# Replay on all robots in config
.venv/bin/python scripts/replay_multi.py --dataset ./data/go2-teleop --episode 0

# Custom config file
.venv/bin/python scripts/replay_multi.py --dataset ./data/go2-teleop --episode 0 \
    --config config/robots.yaml

# Single robot override (no config needed)
.venv/bin/python scripts/replay_multi.py --dataset ./data/go2-teleop --episode 0 \
    --ip 192.168.1.133

# Half speed, skip posture check
.venv/bin/python scripts/replay_multi.py --dataset ./data/go2-teleop --episode 0 \
    --speed 0.5 --skip-posture-check

# Dry run
.venv/bin/python scripts/replay_multi.py --dataset ./data/go2-teleop --episode 0 --dry-run
```

**Config file** (`config/robots.yaml`):

```yaml
posture:
  standing_height_min: 0.28   # body height threshold to confirm standing (m)
  crouched_height_max: 0.12   # body height threshold to confirm crouched (m)

robots:
  - name: go2_55149_air1
    ip: 192.168.1.133
  - name: go2_50905_pro1
    ip: 192.168.1.246
    aes_key: 2c09e23856fa423ed680313dd939a3f0  # required for newer firmware
```

**Flags:**
- `--skip-posture-check` — skip the StandUp verification before replay
- `--posture-timeout <seconds>` — max wait for posture confirmation (default: 8)
- `--speed <multiplier>` — playback speed (0.5 = half speed, 2.0 = double)

## Robot encryption key setup

Go2 robots with firmware >= 1.1.15 use per-device AES-128 encryption for the
LAN WebRTC handshake. You must fetch your robot's key from the Unitree cloud
(one-time step).

### Fetch your AES key

Requires `unitree-webrtc-connect >= 2.1.0` (already included as a dependency):

```bash
# Interactive (prompts for password)
.venv/bin/unitree-fetch-aes-key --email YOUR_EMAIL --device-type Go2

# Non-interactive
.venv/bin/unitree-fetch-aes-key --email YOUR_EMAIL --password YOUR_PASSWORD --device-type Go2

# Get key for a specific robot serial number
.venv/bin/unitree-fetch-aes-key --email YOUR_EMAIL --device-type Go2 --sn B42D2000XXXXXXXX -q
```

This uses the same credentials as the Unitree Go mobile app. The output shows
all robots bound to your account with their serial numbers and AES keys.

### How to tell if your robot needs a key

If connection fails with:
```
AesKeyRequiredError: This robot speaks data2=3 — the per-device AES-128 key is required
```

Then add the `aes_key` field to your robot entry in `config/robots.yaml`.

Robots with older firmware (data2 <= 2) do not need a key and will connect without one.

### Inspect dataset

```bash
.venv/bin/python scripts/inspect_episode.py --dataset ./data/go2-teleop
.venv/bin/python scripts/inspect_episode.py --dataset ./data/go2-teleop --episode 0 --plot
```

## Safety

All safety rules from the Go2 Xbox controller are preserved:
- Dangerous combos (Damp, Jump, Pounce) are blocked by default
- `--allow-all` enables them with a 3-vibration countdown
- `--allow-all --no-countdown` fires them instantly (no hold required)
- Emergency stop: hold LB+LT+RB+RT + any face button
- Speed limiting via `--speed-limit` (default 50%)

## Teleop replay with music

Replay a recorded episode with synchronized music playback:

```bash
# Basic replay with song
.venv/bin/python scripts/replay_teleop.py 192.168.1.246 \
    --aes-key KEY --dataset data/dance-song-1/data/chunk-000 --episode 3 \
    --song ../assets/first-song.m4a --audio-head-start 0.5

# Without music (press ENTER to start moves manually)
.venv/bin/python scripts/replay_teleop.py 192.168.1.246 \
    --aes-key KEY --dataset data/dance-song-1/data/chunk-000 --episode 3 --no-music

# Disable position hold during handstand/erect
.venv/bin/python scripts/replay_teleop.py 192.168.1.246 --no-hold
```

Features:
- Auto-detects starting posture from recording (crouched/standing)
- Closed-loop position hold during balancing modes (handstand/erect)
- Saves replay log to `data/replay_log.parquet` with live robot state

## Multi-robot choreography show

Run synchronized dance on multiple robots with live D-pad takeover:

```bash
.venv/bin/python scripts/choreo_multi.py config/choreo_show.yaml --audio-head-start 4.0
```

**Show config** (`config/choreo_show.yaml`):

```yaml
song: ../assets/first-song.m4a

robots:
  - name: pro1
    ip: 192.168.1.246
    aes_key: 2c09e23856fa423ed680313dd939a3f0
    pro: true
    dataset: data/dance-song-1/data/chunk-000
    episode: 3

  - name: air1
    ip: 192.168.1.133
    pro: false
    dataset: data/dance-song-1/data/chunk-000
    episode: 3
```

**Show flow:**
1. Connects all robots, sets starting postures
2. D-pad / R3 active for pre-show positioning
3. Press ENTER to start music + dance
4. Live takeover during show (D-pad single-robot, R3 all-robots)
5. After show ends, D-pad remains active for repositioning
6. Ctrl+C to exit

**Takeover controls:**
- **D-pad Up/Down/Left/Right** — exclusive single-robot takeover (robots 1–4).
  First press grabs, second press releases.
- **R3 (right stick click)** — toggle ALL robots into MANUAL together. The
  same stick input is broadcast to every robot in lock-step (useful for line
  formations, simultaneous walks). Press again to release everyone.
- **Sticks** — drive whichever robot(s) are currently in MANUAL.
- **Rumble feedback:** single pulse = release, double = grab one robot,
  triple = grab all robots.

**Pro/Air handling:**
- Air robots automatically substitute Pro-only moves (double-click R2 = Erect)
  with sit/stand transitions to stay in sync.

**Show telemetry log:**
Every show writes a JSONL trace to `data/showlogs/show_<YYYYmmdd-HHMMSS>.jsonl`
(default-on, 5 Hz). Inspect the latest run with:

```bash
.venv/bin/python scripts/inspect_show_log.py --latest
```

The log captures pose, foot-force, tilt warnings, motor temps, send errors,
and step boundaries — useful for diagnosing trips or disconnects after the
fact. Disable with `--no-show-log` if needed.
