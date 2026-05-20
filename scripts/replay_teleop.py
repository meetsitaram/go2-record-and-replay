#!/usr/bin/env python3
"""
Replay a recorded teleop episode with music.

Reads the parquet recording and sends the controller actions back to the robot
at the original 20Hz rate, while simultaneously playing the song.

Usage:
    .venv/bin/python scripts/replay_teleop.py 192.168.1.246 --aes-key KEY
    .venv/bin/python scripts/replay_teleop.py 192.168.1.246 --aes-key KEY --episode 0
"""

import argparse
import asyncio
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD
from go2_driver.connection import Go2Connection

DATASET_PATH = Path("data/go2-teleop-v1/data/chunk-000")
SONG_PATH = Path("../assets/Dog-song.m4a")
SEND_RATE = 1.0 / 20  # 20Hz = 50ms per frame

# Body height thresholds for detecting balancing modes
NORMAL_HEIGHT_MAX = 0.45
HANDSTAND_HEIGHT_MIN = 0.50
POSITION_HOLD_GAIN = 0.3
YAW_HOLD_GAIN = 0.5
MAX_CORRECTION = 0.15


class RobotState:
    """Tracks live robot state from sportmodestate topic."""

    def __init__(self):
        self.position = [0.0, 0.0, 0.0]
        self.yaw = 0.0
        self.body_height = 0.0
        self.mode = 0
        self.last_update = 0.0

    def on_message(self, msg):
        try:
            data = msg if isinstance(msg, dict) else json.loads(msg)
            d = data.get("data", data)
            if isinstance(d, str):
                d = json.loads(d)

            pos = d.get("position", None)
            if isinstance(pos, list) and len(pos) >= 3:
                self.position = [float(p) for p in pos]

            imu = d.get("imu_state", {})
            if isinstance(imu, dict):
                rpy = imu.get("rpy", None)
                if rpy and len(rpy) >= 3:
                    self.yaw = float(rpy[2])

            self.body_height = float(d.get("body_height", self.body_height))
            self.mode = int(d.get("mode", self.mode))
            self.last_update = time.monotonic()
        except Exception:
            pass


class PositionHold:
    """Closed-loop position/yaw hold for balancing modes (handstand/erect)."""

    def __init__(self, state: RobotState, enabled: bool = True):
        self.state = state
        self.enabled = enabled
        self.active = False
        self.target_pos = None
        self.target_yaw = None
        self._was_balancing = False

    def _is_balancing(self) -> bool:
        return self.state.body_height >= HANDSTAND_HEIGHT_MIN

    def compute_correction(self) -> tuple[float, float, float]:
        """Returns (lx_corr, ly_corr, rx_corr) to add to joystick values."""
        if not self.enabled:
            return 0.0, 0.0, 0.0

        balancing = self._is_balancing()

        if balancing and not self._was_balancing:
            self.target_pos = list(self.state.position)
            self.target_yaw = self.state.yaw
            self.active = True
        elif not balancing and self._was_balancing:
            self.active = False
            self.target_pos = None
            self.target_yaw = None

        self._was_balancing = balancing

        if not self.active or self.target_pos is None:
            return 0.0, 0.0, 0.0

        dx = self.target_pos[0] - self.state.position[0]
        dy = self.target_pos[1] - self.state.position[1]

        cos_y = math.cos(self.state.yaw)
        sin_y = math.sin(self.state.yaw)
        vx_body = cos_y * dx + sin_y * dy
        vy_body = -sin_y * dx + cos_y * dy

        dyaw = self.target_yaw - self.state.yaw
        while dyaw > math.pi:
            dyaw -= 2 * math.pi
        while dyaw < -math.pi:
            dyaw += 2 * math.pi

        lx_corr = max(-MAX_CORRECTION, min(MAX_CORRECTION, vx_body * POSITION_HOLD_GAIN))
        ly_corr = max(-MAX_CORRECTION, min(MAX_CORRECTION, vy_body * POSITION_HOLD_GAIN))
        rx_corr = max(-MAX_CORRECTION, min(MAX_CORRECTION, dyaw * YAW_HOLD_GAIN))

        return lx_corr, ly_corr, rx_corr


async def replay_with_music(conn, actions, buttons, state, play_music=True,
                            song_path=None, audio_delay=0.5, position_hold=True,
                            start_posture="standing"):
    """Replay recorded actions at 20Hz with music."""
    if song_path is None:
        song_path = SONG_PATH
    num_frames = len(actions)
    duration = num_frames * SEND_RATE

    hold = PositionHold(state, enabled=position_hold)

    print(f"\n  Replaying {num_frames} frames ({duration:.1f}s)")
    print(f"  Song: {song_path}")
    print(f"  Audio head start: {audio_delay:.2f}s")
    print(f"  Position hold (handstand/erect): {'ON' if position_hold else 'OFF'}")
    print()

    # Match starting posture from recording
    if start_posture == "crouched":
        print("  Setting robot to CROUCHED (matching recording start)...")
        await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["StandDown"]}
        )
    else:
        print("  Setting robot to STANDING (matching recording start)...")
        await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["StandUp"]}
        )
    await asyncio.sleep(2)

    # Wait for user to be ready
    print("  Press ENTER to start playback...")
    await asyncio.get_event_loop().run_in_executor(None, input)

    # Start music first - Bluetooth audio has ~200ms latency plus ffplay startup
    audio_proc = None
    if play_music and song_path.exists():
        audio_proc = subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", str(song_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        # Wait for audio pipeline to fill (BT latency + ffplay decode buffer)
        await asyncio.sleep(audio_delay)

    t_start = time.monotonic()

    # Replay log: capture commands sent + live robot state
    replay_log = []

    for i in range(num_frames):
        lx, ly, rx, ry = actions[i].tolist()
        keys = int(buttons[i, 0]) if buttons.ndim > 1 else int(buttons[i])

        # Apply position-hold correction during balancing modes
        lx_c, ly_c, rx_c = hold.compute_correction()
        lx_send = max(-1.0, min(1.0, lx + lx_c))
        ly_send = max(-1.0, min(1.0, ly + ly_c))
        rx_send = max(-1.0, min(1.0, rx + rx_c))

        state_dict = {"lx": lx_send, "ly": ly_send, "rx": rx_send, "ry": ry, "keys": keys}
        msg = json.dumps({
            "type": "msg",
            "topic": "rt/wirelesscontroller",
            "data": state_dict,
        })

        try:
            conn.datachannel.channel.send(msg)
        except Exception:
            pass

        # Log frame
        elapsed = time.monotonic() - t_start
        replay_log.append({
            "time": elapsed,
            "frame": i,
            "lx_orig": lx, "ly_orig": ly, "rx_orig": rx, "ry_orig": ry,
            "lx_sent": lx_send, "ly_sent": ly_send, "rx_sent": rx_send, "ry_sent": ry,
            "keys": keys,
            "robot_x": state.position[0], "robot_y": state.position[1], "robot_z": state.position[2],
            "robot_yaw": state.yaw, "body_height": state.body_height,
            "hold_active": hold.active,
        })

        # Progress display
        if i % 20 == 0 or i == num_frames - 1:
            pct = (i + 1) / num_frames * 100
            hold_str = " [HOLD]" if hold.active else ""
            sys.stdout.write(
                f"\r  [{pct:5.1f}%] {elapsed:.1f}s  "
                f"lx={lx_send:+.2f} ly={ly_send:+.2f} rx={rx_send:+.2f} ry={ry:+.2f} "
                f"keys={keys:#06x}{hold_str}"
                "\033[K"
            )
            sys.stdout.flush()

        # Maintain 20Hz timing
        expected_time = (i + 1) * SEND_RATE
        elapsed = time.monotonic() - t_start
        sleep_time = expected_time - elapsed
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)

    elapsed = time.monotonic() - t_start
    print(f"\n\n  Replay complete in {elapsed:.1f}s")

    # Save replay log
    log_df = pd.DataFrame(replay_log)
    log_path = Path("data/replay_log.parquet")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_parquet(log_path)
    print(f"  Replay log saved: {log_path} ({len(log_df)} frames)")

    if audio_proc:
        audio_proc.terminate()


async def main():
    parser = argparse.ArgumentParser(description="Replay teleop recording with music")
    parser.add_argument("ip", help="Robot IP address")
    parser.add_argument("--aes-key", default=None, help="AES-128 key")
    parser.add_argument("--dataset", default=None, help="Dataset path (default: data/go2-teleop-v1/data/chunk-000)")
    parser.add_argument("--episode", type=int, default=0, help="Episode number to replay")
    parser.add_argument("--no-music", action="store_true", help="Skip music playback")
    parser.add_argument("--song", default=None, help="Path to song file (default: ../assets/Dog-song.m4a)")
    parser.add_argument("--no-hold", action="store_true", help="Disable position hold during handstand/erect")
    parser.add_argument("--audio-head-start", type=float, default=0.5,
                       help="Seconds to let audio play before starting moves (compensates BT latency, increase if audio still lags)")
    args = parser.parse_args()

    # Load recording
    dataset_path = Path(args.dataset) if args.dataset else DATASET_PATH
    episode_file = dataset_path / f"episode_{args.episode:06d}.parquet"
    clean_file = dataset_path / f"episode_{args.episode:06d}_clean.parquet"
    if clean_file.exists():
        episode_file = clean_file
        print(f"Using cleaned recording: {clean_file.name}")
    if not episode_file.exists():
        print(f"ERROR: {episode_file} not found")
        sys.exit(1)

    df = pd.read_parquet(episode_file)
    actions = np.array([a for a in df["action"].values])
    buttons = np.array([b for b in df["action.buttons"].values])
    print(f"Loaded episode {args.episode}: {len(df)} frames ({len(df)/20:.1f}s)")

    # Determine starting posture from recording
    first_state = np.array(df["observation.state"].iloc[0])
    start_height = first_state[2]  # pos_z
    if start_height < 0.15:
        start_posture = "crouched"
    else:
        start_posture = "standing"
    print(f"Recording started in {start_posture} posture (height={start_height:.3f}m)")

    # Connect
    go2 = Go2Connection("sta", args.ip, aes_key=args.aes_key)
    print(f"Connecting to {args.ip}...")
    conn = await go2.async_connect()
    print("Connected!")

    # State tracking for position hold
    state = RobotState()
    conn.datachannel.pub_sub.subscribe("rt/lf/sportmodestate", state.on_message)

    await asyncio.sleep(1)  # let state populate

    await replay_with_music(conn, actions, buttons, state,
                            play_music=not args.no_music,
                            song_path=Path(args.song) if args.song else SONG_PATH,
                            audio_delay=args.audio_head_start,
                            position_hold=not args.no_hold,
                            start_posture=start_posture)

    await go2.async_disconnect()


if __name__ == "__main__":
    asyncio.run(main())
