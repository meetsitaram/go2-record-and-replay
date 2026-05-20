#!/usr/bin/env python3
"""
Multi-robot choreography with live controller takeover.

Replays recorded episodes on multiple robots simultaneously. During playback,
the operator can use the D-pad to take manual control of any robot:
  - D-pad Up/Down/Left/Right = select robot 1/2/3/4
  - First press: take manual control (robot pauses its dance)
  - Second press: release back to dance (resumes from current position in timeline)

Usage:
    .venv/bin/python scripts/choreo_multi.py config/choreo_show.yaml
    .venv/bin/python scripts/choreo_multi.py config/choreo_show.yaml --no-music
"""

import argparse
import asyncio
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD
from go2_driver.connection import Go2Connection
from go2_driver.gamepad import (
    ControllerState, find_gamepad, validate_gamepad, gamepad_loop
)
from go2_driver.constants import KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT

SEND_RATE = 1.0 / 20  # 20Hz

# D-pad to robot index mapping
DPAD_MAP = {
    KEY_UP: 0,
    KEY_DOWN: 1,
    KEY_LEFT: 2,
    KEY_RIGHT: 3,
}

# Named move -> button bitmask for overrides
MOVE_KEYS = {
    "stretch": 0x0010 | 0x0100,    # RT + A
    "shake_hands": 0x0010 | 0x0200, # RT + B
    "love": 0x0010 | 0x0800,       # RT + Y
    "greet": 0x0002 | 0x0100,      # LB + A
    "dance": 0x0002 | 0x0200,      # LB + B
    "jump": 0x0001 | 0x0100,       # RB + A
    "sit": 0x0001 | 0x0200,        # RB + B
    "pounce": 0x0001 | 0x0400,     # RB + X
    "stand_up": 0x0020 | 0x0400,   # LT + X
    "crouch": 0x0020 | 0x0100,     # LT + A
    "none": 0,                      # neutral (no buttons)
}


class RobotPlayer:
    """Manages connection and replay state for a single robot."""

    def __init__(self, name: str, ip: str, aes_key: str | None,
                 actions: np.ndarray, buttons: np.ndarray, start_posture: str,
                 overrides: list | None = None, is_pro: bool = True):
        self.name = name
        self.ip = ip
        self.aes_key = aes_key
        self.actions = actions
        self.buttons = buttons
        self.start_posture = start_posture
        self.num_frames = len(actions)
        self._go2 = Go2Connection("sta", ip, aes_key=aes_key)
        self.conn = None
        self.frame_index = 0
        self.manual_override = False
        self.connected = False
        self.overrides = overrides or []
        self.is_pro = is_pro
        self._in_pro_mode = False
        self._prev_r2 = False
        self._last_r2_frame = None

    async def connect(self):
        print(f"  [{self.name}] Connecting to {self.ip}...")
        self.conn = await self._go2.async_connect()
        self.connected = True
        print(f"  [{self.name}] Connected!")

    async def set_start_posture(self):
        if not self.connected:
            return
        if self.start_posture == "crouched":
            await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["StandDown"]}
            )
        else:
            await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["StandUp"]}
            )

    # Pro-only button combos that toggle a balancing mode
    PRO_ONLY_COMBOS = [
        (0x0010 | 0x0400),  # RT + X (Handstand) — toggle
    ]
    SIT_KEYS = 0x0001 | 0x0200    # RB + B (Sit down)
    STAND_KEYS = 0x0020 | 0x0400  # LT + X (Stand up from fall)
    DOUBLE_CLICK_FRAMES = 8  # 0.4s at 20Hz

    def get_frame(self) -> dict | None:
        """Get the next frame from the recording, or None if done."""
        if self.frame_index >= self.num_frames:
            return None
        lx, ly, rx, ry = self.actions[self.frame_index].tolist()
        keys = int(self.buttons[self.frame_index, 0]) if self.buttons.ndim > 1 else int(self.buttons[self.frame_index])
        self.frame_index += 1

        # On Air robots, detect double-click R2 (Pro erect toggle) and substitute
        if not self.is_pro:
            r2_now = bool(keys & 0x0010)  # KEY_R2
            r2_rising = r2_now and not self._prev_r2

            if r2_rising:
                if (self._last_r2_frame is not None and
                        (self.frame_index - self._last_r2_frame) <= self.DOUBLE_CLICK_FRAMES):
                    # Double-click R2 detected
                    if not self._in_pro_mode:
                        self._in_pro_mode = True
                        keys = self.SIT_KEYS
                    else:
                        self._in_pro_mode = False
                        keys = self.STAND_KEYS
                    self._last_r2_frame = None
                else:
                    self._last_r2_frame = self.frame_index
            self._prev_r2 = r2_now

            # While Pro is in erect mode, Air stays sitting
            if self._in_pro_mode and not (keys == self.SIT_KEYS):
                keys = 0

        return {"lx": lx, "ly": ly, "rx": rx, "ry": ry, "keys": keys}

    def send(self, state_dict: dict):
        """Send controller state to robot."""
        if not self.connected:
            return
        msg = json.dumps({
            "type": "msg",
            "topic": "rt/wirelesscontroller",
            "data": state_dict,
        })
        try:
            self.conn.datachannel.channel.send(msg)
        except Exception:
            pass

    async def disconnect(self):
        await self._go2.async_disconnect()


def load_show_config(path: str) -> dict:
    """Load the show configuration YAML."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_episode(dataset_path: str, episode: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Load an episode and return (actions, buttons, start_posture)."""
    p = Path(dataset_path)
    ep_file = p / f"episode_{episode:06d}.parquet"
    clean_file = p / f"episode_{episode:06d}_clean.parquet"
    if clean_file.exists():
        ep_file = clean_file

    if not ep_file.exists():
        print(f"ERROR: {ep_file} not found")
        sys.exit(1)

    df = pd.read_parquet(ep_file)
    actions = np.array([a for a in df["action"].values])
    buttons = np.array([b for b in df["action.buttons"].values])

    first_state = np.array(df["observation.state"].iloc[0])
    start_height = first_state[2]
    start_posture = "crouched" if start_height < 0.15 else "standing"

    return actions, buttons, start_posture


async def run_show(config: dict, no_music: bool = False, audio_head_start: float = 0.5):
    """Main show loop."""
    robots_cfg = config["robots"]
    song_path = config.get("song")

    # Load episodes and create robot players
    players: list[RobotPlayer] = []
    for rcfg in robots_cfg:
        actions, buttons, start_posture = load_episode(
            rcfg["dataset"], rcfg.get("episode", 0)
        )
        player = RobotPlayer(
            name=rcfg["name"],
            ip=rcfg["ip"],
            aes_key=rcfg.get("aes_key"),
            actions=actions,
            buttons=buttons,
            start_posture=start_posture,
            is_pro=rcfg.get("pro", False),
        )
        players.append(player)
        print(f"  [{player.name}] Loaded {player.num_frames} frames "
              f"({player.num_frames * SEND_RATE:.1f}s), start={start_posture}")

    # Connect all robots
    print("\nConnecting to robots...")
    connect_tasks = [p.connect() for p in players]
    results = await asyncio.gather(*connect_tasks, return_exceptions=True)
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"  [{players[i].name}] FAILED: {r}")

    connected_players = [p for p in players if p.connected]
    if not connected_players:
        print("ERROR: No robots connected!")
        return

    # Set starting postures
    print("\nSetting start postures...")
    await asyncio.gather(*[p.set_start_posture() for p in connected_players])
    await asyncio.sleep(2)

    # Setup gamepad for takeover control
    gamepad_state = ControllerState()
    stop_event = threading.Event()
    gamepad_dev = find_gamepad()
    gamepad_thread = None

    if gamepad_dev:
        warnings = validate_gamepad(gamepad_dev)
        if not warnings:
            gamepad_thread = threading.Thread(
                target=gamepad_loop,
                args=(gamepad_dev, gamepad_state, stop_event),
                daemon=True,
            )
            gamepad_thread.start()
            print(f"\n  Gamepad connected: {gamepad_dev.name}")
        else:
            print(f"\n  Gamepad warnings: {warnings}")
    else:
        print("\n  No gamepad found — takeover disabled")

    # Print controls
    print("\n" + "=" * 60)
    print("  SHOW CONTROLS:")
    print("    D-pad Up    -> takeover/release robot 1" +
          (f" ({players[0].name})" if len(players) > 0 else ""))
    print("    D-pad Down  -> takeover/release robot 2" +
          (f" ({players[1].name})" if len(players) > 1 else ""))
    print("    D-pad Left  -> takeover/release robot 3" +
          (f" ({players[2].name})" if len(players) > 2 else ""))
    print("    D-pad Right -> takeover/release robot 4" +
          (f" ({players[3].name})" if len(players) > 3 else ""))
    print("    Ctrl+C      -> stop show")
    print("=" * 60)

    def handle_dpad(connected_players, prev_dpad, gamepad_state, phase=""):
        """Handle D-pad input: exclusive single-robot takeover. Returns new prev_dpad."""
        gstate = gamepad_state.to_dict()
        current_keys = gstate["keys"]

        for dpad_key, robot_idx in DPAD_MAP.items():
            if robot_idx >= len(connected_players):
                continue
            pressed_now = bool(current_keys & dpad_key)
            was_pressed = bool(prev_dpad & dpad_key)

            if pressed_now and not was_pressed:
                p = connected_players[robot_idx]
                if p.manual_override:
                    # Same robot pressed again — release
                    p.manual_override = False
                    sys.stdout.write(f"\n  [{p.name}] -> AUTO")
                else:
                    # Release any other robot first
                    for other in connected_players:
                        other.manual_override = False
                    p.manual_override = True
                    sys.stdout.write(f"\n  [{p.name}] -> MANUAL (controlling)")

                # Print active list
                active = [pp.name for pp in connected_players if pp.manual_override]
                if active:
                    sys.stdout.write(f"  |  Active: {', '.join(active)}")
                else:
                    sys.stdout.write(f"  |  All on AUTO")
                sys.stdout.write("\n")
                sys.stdout.flush()

        return current_keys & (KEY_UP | KEY_DOWN | KEY_LEFT | KEY_RIGHT)

    # Pre-show positioning loop: D-pad takeover active, waiting for Enter
    print("\n  D-pad active for positioning. Press ENTER to start the show...")
    enter_pressed = asyncio.Event()

    def _wait_enter():
        input()
        enter_pressed.set()

    asyncio.get_event_loop().run_in_executor(None, _wait_enter)

    prev_dpad = 0
    while not enter_pressed.is_set():
        if gamepad_thread:
            prev_dpad = handle_dpad(connected_players, prev_dpad, gamepad_state)

            for p in connected_players:
                if p.manual_override:
                    gstate = gamepad_state.to_dict()
                    gstate["keys"] = gstate["keys"] & ~(KEY_UP | KEY_DOWN | KEY_LEFT | KEY_RIGHT)
                    p.send(gstate)

        await asyncio.sleep(SEND_RATE)

    # Reset all overrides for show start
    for p in connected_players:
        p.manual_override = False

    # Start music
    audio_proc = None
    if not no_music and song_path and Path(song_path).exists():
        audio_proc = subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", str(song_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        await asyncio.sleep(audio_head_start)

    # Main playback loop
    max_frames = max(p.num_frames for p in connected_players)
    t_start = time.monotonic()
    prev_dpad = 0

    print("\n  Show started!\n")

    for frame_i in range(max_frames):
        # Check D-pad for takeover toggles
        if gamepad_thread:
            prev_dpad = handle_dpad(connected_players, prev_dpad, gamepad_state)

        # Send frames to each robot
        for p in connected_players:
            if p.manual_override:
                # Send live gamepad state (without D-pad keys)
                if gamepad_thread:
                    gstate = gamepad_state.to_dict()
                    gstate["keys"] = gstate["keys"] & ~(KEY_UP | KEY_DOWN | KEY_LEFT | KEY_RIGHT)
                    p.send(gstate)
                else:
                    p.send({"lx": 0, "ly": 0, "rx": 0, "ry": 0, "keys": 0})
                # Keep frame index advancing so it stays in sync when released
                if p.frame_index < p.num_frames:
                    p.frame_index += 1
            else:
                frame = p.get_frame()
                if frame:
                    p.send(frame)

        # Progress display
        if frame_i % 20 == 0:
            elapsed = time.monotonic() - t_start
            statuses = []
            for p in connected_players:
                pct = min(100, p.frame_index / p.num_frames * 100)
                mode = "M" if p.manual_override else "A"
                statuses.append(f"{p.name[:8]}:{pct:.0f}%[{mode}]")
            sys.stdout.write(f"\r  {elapsed:.1f}s | {' | '.join(statuses)}\033[K")
            sys.stdout.flush()

        # Maintain 20Hz timing
        expected_time = (frame_i + 1) * SEND_RATE
        elapsed = time.monotonic() - t_start
        sleep_time = expected_time - elapsed
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)

    elapsed = time.monotonic() - t_start
    print(f"\n\n  Show complete in {elapsed:.1f}s")
    print("  D-pad still active for repositioning. Ctrl+C to exit.")

    if audio_proc:
        audio_proc.terminate()

    # Post-show positioning loop: keep D-pad active until Ctrl+C
    prev_dpad = 0
    while True:
        if gamepad_thread:
            prev_dpad = handle_dpad(connected_players, prev_dpad, gamepad_state)

            for p in connected_players:
                if p.manual_override:
                    gstate = gamepad_state.to_dict()
                    gstate["keys"] = gstate["keys"] & ~(KEY_UP | KEY_DOWN | KEY_LEFT | KEY_RIGHT)
                    p.send(gstate)

        await asyncio.sleep(SEND_RATE)


async def main():
    parser = argparse.ArgumentParser(description="Multi-robot choreography show")
    parser.add_argument("config", help="Show config YAML file")
    parser.add_argument("--no-music", action="store_true", help="Skip music playback")
    parser.add_argument("--audio-head-start", type=float, default=0.5,
                       help="Seconds to let audio play before moves start")
    args = parser.parse_args()

    config = load_show_config(args.config)
    await run_show(config, no_music=args.no_music, audio_head_start=args.audio_head_start)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n  Show ended. Disconnecting...")
