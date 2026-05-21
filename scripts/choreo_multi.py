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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD
from go2_driver.connection import Go2Connection
from go2_driver.gamepad import (
    ControllerState, find_gamepad, validate_gamepad, gamepad_loop, RumbleHelper
)
from go2_driver.constants import KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT
from go2_driver.streams import RobotStreams

# Local imports: algo move catalogue + scaling helpers.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
from go2_recorder.algo_moves_lib import (  # noqa: E402
    CATALOGUE_BY_NAME, Move, clamp_params,
)
from go2_recorder.show_logger import ShowLogger  # noqa: E402

SEND_RATE = 1.0 / 20  # 20Hz

# D-pad to robot index mapping
DPAD_MAP = {
    KEY_UP: 0,
    KEY_DOWN: 1,
    KEY_LEFT: 2,
    KEY_RIGHT: 3,
}

# Controller key bits for mode switching (mirrors algo_moves.py).
KEY_SELECT = 0x0008  # = "Standing mode" (BalanceStand)
KEY_START = 0x0004   # = "Walking mode"

# Human-readable names for the common controller bits, used by the recording
# logger to make sense of button presses replayed during a recording step.
_KEY_NAMES = {
    0x0001: "R1",   0x0002: "L1",   0x0004: "Start", 0x0008: "Select",
    0x0010: "R2",   0x0020: "L2",   0x0040: "F1",    0x0080: "F2",
    0x0100: "A",    0x0200: "B",    0x0400: "X",     0x0800: "Y",
    0x1000: "Up",   0x2000: "Right", 0x4000: "Down", 0x8000: "Left",
}


def _describe_keys(k: int) -> str:
    names = [n for m, n in _KEY_NAMES.items() if k & m]
    return "+".join(names) if names else "none"


# ─── Audio player with sticky-song semantics ────────────────────────────

class AudioPlayer:
    """Wraps ffplay with per-step song switching.

    State machine driven by `apply_step(step)`:
      - step.song is _SONG_UNSET: do nothing, current song keeps playing
      - step.song is None:        stop the current song (silent)
      - step.song is a path:      if different from current, kill + restart
                                   if same (and same offset), do nothing
    """

    def __init__(self, head_start: float = 0.5):
        self.head_start = head_start
        self._proc: subprocess.Popen | None = None
        self._current_song: str | None = None
        self._current_offset: float = 0.0

    def _stop(self):
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass
        self._proc = None
        self._current_song = None
        self._current_offset = 0.0

    def _start(self, song_path: str, start_offset: float):
        path = Path(song_path)
        if not path.exists():
            sys.stdout.write(f"\n  WARN: song file not found: {song_path}\n")
            sys.stdout.flush()
            return
        cmd = ["ffplay", "-nodisp", "-autoexit"]
        if start_offset > 0:
            cmd += ["-ss", f"{start_offset:.3f}"]
        cmd.append(str(path))
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._current_song = str(path)
        self._current_offset = start_offset

    async def apply_step(self, step):
        """Apply the step's audio directive. May block for head_start seconds
        when launching a new song to let the BT pipeline fill up."""
        if step.song is _SONG_UNSET:
            return  # sticky: keep what's playing

        if step.song is None:
            if self._proc is not None:
                sys.stdout.write("\n  [audio] stopping music\n")
                sys.stdout.flush()
                self._stop()
            return

        new_path = str(Path(step.song))
        # Already playing the same song from the same offset? Leave it alone.
        if (new_path == self._current_song
                and abs(step.song_start - self._current_offset) < 0.001):
            return

        # Switch.
        if self._proc is not None:
            self._stop()
        sys.stdout.write(
            f"\n  [audio] start {Path(step.song).name}"
            f"{f' @{step.song_start:g}s' if step.song_start else ''}"
            f"  (head_start={self.head_start:.2f}s)\n"
        )
        sys.stdout.flush()
        self._start(new_path, step.song_start)
        if self.head_start > 0:
            await asyncio.sleep(self.head_start)

    def shutdown(self):
        self._stop()


# ─── Shared step abstraction ─────────────────────────────────────────────

# Sentinel: "song field not specified in YAML" -> keep current music as-is.
# Distinguish from `song = None` which means "explicitly stop music".
_SONG_UNSET = object()


@dataclass
class RecordingStep:
    """One step: replay a single .parquet on all robots."""
    dataset: str
    episode: int
    actions: np.ndarray = field(repr=False)
    buttons: np.ndarray = field(repr=False)
    start_posture: str
    # Audio control (sticky semantics, see AudioPlayer):
    #   - song = _SONG_UNSET  : no change, current music keeps playing
    #   - song = None         : stop the current music (silent)
    #   - song = "path"       : start (or switch to) this song
    song: object = _SONG_UNSET
    song_start: float = 0.0

    @property
    def kind(self) -> str:
        return "recording"

    @property
    def duration(self) -> float:
        return len(self.actions) * SEND_RATE

    @property
    def num_frames(self) -> int:
        return len(self.actions)

    @property
    def label(self) -> str:
        return f"recording {Path(self.dataset).name}/ep{self.episode:03d}"


@dataclass
class AlgoStep:
    """One step: play an algorithmic move on all robots."""
    move: Move
    duration: float
    tempo: float = 1.0
    amplitude: float = 1.0
    # Audio control -- same semantics as RecordingStep.
    song: object = _SONG_UNSET
    song_start: float = 0.0

    @property
    def kind(self) -> str:
        return "algo"

    @property
    def num_frames(self) -> int:
        return int(round(self.duration / SEND_RATE))

    @property
    def label(self) -> str:
        tag = ""
        if abs(self.tempo - 1.0) > 1e-3 or abs(self.amplitude - 1.0) > 1e-3:
            tag = f" [tempo x{self.tempo:.2f} amp x{self.amplitude:.2f}]"
        return f"algo {self.move.name}{tag}"


@dataclass
class ActionStep:
    """One step: fire a single Sport API command on all robots.

    Fire-and-forget: we publish the api_id once to every connected robot
    and move on. Use this for one-shot built-in actions (Stretch, Hello,
    Sit, WiggleHips, FingerHeart, Pose, Scrape, Dance1, Dance2 ...).

    Set `duration:` to insert a wait after firing so the action has time
    to play before the next step. Defaults to 0 (no wait).
    """
    name: str             # SPORT_CMD key, e.g. "Stretch"
    api_id: int           # resolved from SPORT_CMD[name]
    duration: float = 0.0
    # Audio control -- same semantics as RecordingStep / AlgoStep.
    song: object = _SONG_UNSET
    song_start: float = 0.0

    @property
    def kind(self) -> str:
        return "action"

    @property
    def num_frames(self) -> int:
        return int(round(self.duration / SEND_RATE))

    @property
    def label(self) -> str:
        return f"action {self.name} (api_id={self.api_id})"


def _load_episode_data(dataset_path: str, episode: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Read a parquet episode into (actions, buttons, start_posture)."""
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


def build_script(config: dict) -> list:
    """Turn a config dict into a list of Step objects.

    Supports two formats:
      1. NEW: top-level `script:` list with {type: algo|recording, ...} entries.
         All robots play the same script.
      2. LEGACY: each robot has its own `dataset:` + `episode:`. We build a
         one-step recording script using the FIRST robot's entries (they
         should all match in the legacy case anyway -- choreo always played
         one episode per robot).
    """
    if "script" in config:
        raw_steps = config["script"]
        steps: list = []
        for i, entry in enumerate(raw_steps, 1):
            # Parse song fields shared by both step types.
            # Use _SONG_UNSET if the field is absent (sticky: keep prev music).
            # `song: null` -> Python None -> "stop music".
            song = entry["song"] if "song" in entry else _SONG_UNSET
            song_start = float(entry.get("song_start", 0.0))

            t = entry.get("type", "recording")
            if t == "recording":
                ds = entry["dataset"]
                ep = int(entry.get("episode", 0))
                actions, buttons, posture = _load_episode_data(ds, ep)
                steps.append(RecordingStep(
                    dataset=ds, episode=ep,
                    actions=actions, buttons=buttons,
                    start_posture=posture,
                    song=song, song_start=song_start,
                ))
            elif t == "algo":
                name = entry["name"]
                if name not in CATALOGUE_BY_NAME:
                    print(f"ERROR: step {i}: unknown algo move '{name}'")
                    print(f"Available: {sorted(CATALOGUE_BY_NAME)}")
                    sys.exit(1)
                move = CATALOGUE_BY_NAME[name]
                steps.append(AlgoStep(
                    move=move,
                    duration=float(entry.get("duration", move.duration)),
                    tempo=float(entry.get("tempo", 1.0)),
                    amplitude=float(entry.get("amplitude", 1.0)),
                    song=song, song_start=song_start,
                ))
            elif t == "action":
                name = entry["name"]
                if name not in SPORT_CMD:
                    print(f"ERROR: step {i}: unknown action '{name}'")
                    print(f"Available SPORT_CMD names: {sorted(SPORT_CMD)}")
                    sys.exit(1)
                steps.append(ActionStep(
                    name=name,
                    api_id=SPORT_CMD[name],
                    duration=float(entry.get("duration", 0.0)),
                    song=song, song_start=song_start,
                ))
            else:
                print(f"ERROR: step {i}: unknown type '{t}'")
                sys.exit(1)
        return steps

    # Legacy path: one recording-step per robot. Since all robots share the
    # same script now, take the first robot's dataset/episode and warn if
    # robots disagree.
    robots = config["robots"]
    first = robots[0]
    ds = first["dataset"]
    ep = int(first.get("episode", 0))
    for r in robots[1:]:
        if r.get("dataset") != ds or int(r.get("episode", 0)) != ep:
            print(f"  WARN: legacy config has different episodes per robot; "
                  f"using {ds}/ep{ep:03d} for all.")
            break
    actions, buttons, posture = _load_episode_data(ds, ep)
    return [RecordingStep(
        dataset=ds, episode=ep,
        actions=actions, buttons=buttons,
        start_posture=posture,
    )]


def print_script(steps: list) -> None:
    total = 0.0
    print(f"\n  Script: {len(steps)} step(s)")
    print(f"  {'#':>2}  {'kind':<10} {'dur':>6}  detail")
    print(f"  {'-'*2}  {'-'*10} {'-'*6}  {'-'*40}")
    for i, s in enumerate(steps, 1):
        d = s.duration
        total += d
        # Render the song hint per step so it's clear what audio behavior
        # to expect: -- = sticky (no change), [silent] = stop, [path] = play.
        if s.song is _SONG_UNSET:
            song_hint = ""
        elif s.song is None:
            song_hint = "  [audio: STOP]"
        else:
            offset = f" @{s.song_start:g}s" if s.song_start else ""
            song_hint = f"  [audio: {Path(s.song).name}{offset}]"
        print(f"  {i:>2}  {s.kind:<10} {d:>5.1f}s  {s.label}{song_hint}")
    print(f"  {'':>2}  {'TOTAL':<10} {total:>5.1f}s\n")

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
    """Manages the connection + send state for a single robot.

    No longer owns playback data: steps (RecordingStep / AlgoStep) live at the
    show level and are dispatched to every player in parallel each step.
    """

    def __init__(self, name: str, ip: str, aes_key: str | None,
                 is_pro: bool = True):
        self.name = name
        self.ip = ip
        self.aes_key = aes_key
        self._go2 = Go2Connection("sta", ip, aes_key=aes_key)
        self.conn = None
        self.connected = False
        self.is_pro = is_pro
        self.manual_override = False
        # Pro/Air R2 double-click tracking (used during recording replay).
        self._in_pro_mode = False
        self._prev_r2 = False
        self._last_r2_frame = None
        # Per-step frame counter; reset at the start of each recording step.
        self.frame_index = 0
        self.send_errors = 0

    def is_alive(self) -> bool:
        return self.connected and self._go2.is_alive()

    async def connect(self):
        print(f"  [{self.name}] Connecting to {self.ip}...")
        self.conn = await self._go2.async_connect()
        self.connected = True

        def _on_close(state: str):
            self.connected = False
            sys.stdout.write(
                f"\n  ERROR: [{self.name} @ {self.ip}] WebRTC {state}. "
                "Robot dropped from show.\n"
            )
            sys.stdout.flush()

        self._go2.on_close(_on_close)
        print(f"  [{self.name}] Connected!")

    async def set_start_posture(self, posture: str):
        if not self.connected:
            return
        if posture == "crouched":
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

    def reset_step(self):
        """Reset per-step counters (frame index, R2 tracking) before a new step."""
        self.frame_index = 0
        self._in_pro_mode = False
        self._prev_r2 = False
        self._last_r2_frame = None
        self._prev_rec_keys = 0  # for one-shot button compression

    def get_recording_frame(self, step: RecordingStep) -> dict | None:
        """Get the next frame from the current recording step, or None if done."""
        if self.frame_index >= step.num_frames:
            return None
        lx, ly, rx, ry = step.actions[self.frame_index].tolist()
        b = step.buttons
        keys = int(b[self.frame_index, 0]) if b.ndim > 1 else int(b[self.frame_index])
        self.frame_index += 1

        # TOGGLE-COMBO COMPRESSION (narrow)
        # Some Pro firmware combos toggle state on each "press" event the
        # firmware sees. When the recording holds the combo for 4-6 frames
        # (a normal thumb-press of 200-300ms), the firmware cycles state
        # back and forth. We collapse ONLY these specific combos to a
        # single rising-edge pulse; all other buttons pass through unchanged.
        TOGGLE_COMBOS = {
            0x0120,   # L2 + A : Lock posture toggle (stand <-> crouch)
        }
        raw_keys = keys  # remember the actual keys for the next-frame compare
        if keys in TOGGLE_COMBOS and keys == self._prev_rec_keys:
            keys = 0   # suppress this repeat frame
        self._prev_rec_keys = raw_keys

        # On Air robots, detect double-click R2 (Pro erect toggle) and substitute.
        if not self.is_pro:
            r2_now = bool(keys & 0x0010)  # KEY_R2
            r2_rising = r2_now and not self._prev_r2

            if r2_rising:
                if (self._last_r2_frame is not None and
                        (self.frame_index - self._last_r2_frame) <= self.DOUBLE_CLICK_FRAMES):
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
        except Exception as e:
            self.send_errors += 1
            if self.send_errors <= 3 or self.send_errors % 50 == 0:
                sys.stdout.write(
                    f"\n  WARN: [{self.name}] send failed "
                    f"(#{self.send_errors}, {type(e).__name__}: {e})\n"
                )
                sys.stdout.flush()

    def send_controller_keys(self, keys: int):
        """Send a raw controller frame with only a keys field (and sticks zero).

        Used to emulate Select / Start button presses for mode switching, which
        is the reliable way to put a Pro into BalanceStand or Walking mode.
        """
        self.send({"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0, "keys": keys})

    async def send_sport(self, api_id: int, params: dict | None = None):
        """Send a sport-mode API request (Euler / BodyHeight / Move / ...)."""
        if not self.connected:
            return
        opts = {"api_id": api_id}
        if params is not None:
            opts["parameter"] = params
        try:
            await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"], opts
            )
        except Exception as e:
            self.send_errors += 1
            if self.send_errors <= 3 or self.send_errors % 50 == 0:
                sys.stdout.write(
                    f"\n  WARN: [{self.name}] sport send failed "
                    f"(#{self.send_errors}, {type(e).__name__}: {e})\n"
                )
                sys.stdout.flush()

    async def disconnect(self):
        await self._go2.async_disconnect()


# ─── Algo command helpers ────────────────────────────────────────────────

def _scale_cmd(cmd: dict, amp: float) -> dict:
    """Multiply a command's spatial params by `amp` (amplitude multiplier).

    Mirrors algo_moves.py._scale_cmd so the per-step amplitude flag works
    identically here.
    """
    if abs(amp - 1.0) < 1e-6:
        return cmd
    t = cmd.get("type")
    if t == "euler":
        p = cmd["params"]
        return {"type": "euler", "params": {
            "x": p.get("x", 0.0) * amp,
            "y": p.get("y", 0.0) * amp,
            "z": p.get("z", 0.0) * amp,
        }}
    if t == "body_height":
        return {"type": "body_height",
                "params": {"data": cmd["params"].get("data", 0.0) * amp}}
    if t == "move":
        p = cmd["params"]
        return {"type": "move", "params": {
            "x": p.get("x", 0.0) * amp,
            "y": p.get("y", 0.0) * amp,
            "z": p.get("z", 0.0) * amp,
        }}
    if t == "compound":
        return {"type": "compound",
                "params": [_scale_cmd(sub, amp) for sub in cmd["params"]]}
    return cmd


async def _broadcast_sport_cmd(players: list, cmd: dict):
    """Send one algo-frame command to every (live, non-overridden) player."""
    cmd = clamp_params(cmd)
    t = cmd["type"]
    p = cmd["params"]
    if t == "euler":
        api_id = SPORT_CMD["Euler"]
        await asyncio.gather(*[pl.send_sport(api_id, p)
                               for pl in players if pl.is_alive()
                               and not pl.manual_override])
    elif t == "body_height":
        api_id = SPORT_CMD["BodyHeight"]
        await asyncio.gather(*[pl.send_sport(api_id, p)
                               for pl in players if pl.is_alive()
                               and not pl.manual_override])
    elif t == "move":
        api_id = SPORT_CMD["Move"]
        await asyncio.gather(*[pl.send_sport(api_id, p)
                               for pl in players if pl.is_alive()
                               and not pl.manual_override])
    elif t == "compound":
        for sub in p:
            await _broadcast_sport_cmd(players, sub)


async def _broadcast_mode_press(players: list, walking: bool):
    """Press Select (balance) or Start (walking) on every robot at once.

    Pressing the controller-emulated mode button is the reliable way to put
    the Pro into the right mode before an algo step; direct API calls are
    silently ignored on some firmwares.

    Robots under manual_override are skipped -- the operator owns the
    gamepad state for those robots and we must not inject phantom presses.
    """
    keys = KEY_START if walking else KEY_SELECT
    targets = [pl for pl in players if pl.is_alive() and not pl.manual_override]
    # Hold ~150ms (3 frames at 20Hz) then release, mirroring algo_moves.py.
    for _ in range(3):
        for pl in targets:
            pl.send_controller_keys(keys)
        await asyncio.sleep(0.05)
    for pl in targets:
        pl.send_controller_keys(0)


def load_show_config(path: str) -> dict:
    """Load the show configuration YAML."""
    with open(path) as f:
        return yaml.safe_load(f)


# ─── Step runners (called by run_show for each step) ────────────────────

# Set once by run_show() after the gamepad is discovered. Read by
# _handle_dpad() to provide haptic feedback on takeover transitions.
# Kept at module scope so we don't have to thread `rumble` through every
# step runner's signature.
_RUMBLE: "RumbleHelper | None" = None

# Gap (seconds) between the two pulses of the "grab" double-tap. PULSE_MS
# from the driver is 250ms, so 0.35s gap gives a distinct two-thump feel.
_DOUBLE_PULSE_GAP = 0.35


def _rumble_double_pulse(rumble: "RumbleHelper | None"):
    """Fire two pulses ~PULSE_MS apart on a background thread.

    Backgrounded so we don't block the D-pad polling loop while the
    pulse plays out (FF_RUMBLE effects are non-blocking on upload, but
    we still need to sleep between firings).
    """
    if rumble is None:
        return

    def _runner():
        rumble.pulse()
        time.sleep(_DOUBLE_PULSE_GAP)
        rumble.pulse()

    threading.Thread(target=_runner, daemon=True).start()


def _handle_dpad(connected_players, prev_dpad, gamepad_state,
                  rumble: "RumbleHelper | None" = None):
    """Handle D-pad input: exclusive single-robot takeover. Returns new mask.

    Rumble feedback fires on every AUTO<->MANUAL transition so the operator
    gets tactile confirmation that they just grabbed/released a robot
    (especially useful since the screen may be 10ft away during a show):
      - AUTO -> MANUAL : double-pulse (grab)
      - MANUAL -> AUTO : single pulse (release)
    Source: explicit `rumble` arg if given, else the module-level _RUMBLE.
    """
    rumble = rumble if rumble is not None else _RUMBLE
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
                p.manual_override = False
                sys.stdout.write(f"\n  [{p.name}] -> AUTO")
                # Single pulse: "letting go".
                if rumble is not None:
                    rumble.pulse()
            else:
                for other in connected_players:
                    other.manual_override = False
                p.manual_override = True
                sys.stdout.write(f"\n  [{p.name}] -> MANUAL (controlling)")
                # Double pulse: "I just grabbed this robot, careful".
                _rumble_double_pulse(rumble)

            active = [pp.name for pp in connected_players if pp.manual_override]
            if active:
                sys.stdout.write(f"  |  Active: {', '.join(active)}")
            else:
                sys.stdout.write(f"  |  All on AUTO")
            sys.stdout.write("\n")
            sys.stdout.flush()

    return current_keys & (KEY_UP | KEY_DOWN | KEY_LEFT | KEY_RIGHT)


def _send_takeover_frame(players, gamepad_state, gamepad_active):
    """Send live gamepad state to any robot under manual override."""
    if not gamepad_active:
        return
    gstate = gamepad_state.to_dict()
    gstate["keys"] = gstate["keys"] & ~(KEY_UP | KEY_DOWN | KEY_LEFT | KEY_RIGHT)
    for p in players:
        if p.manual_override and p.is_alive():
            p.send(gstate)


async def run_recording_step(step: RecordingStep, players: list,
                              gamepad_state, gamepad_active: bool,
                              step_idx: int, total_steps: int,
                              t_offset: float, t_start_show: float,
                              audio: "AudioPlayer | None" = None):
    """Replay one parquet on every robot in lock-step.

    All robots play the SAME recorded actions/buttons. They start together at
    frame 0 and end together at the last frame. Manual takeover still works.
    """
    for p in players:
        p.reset_step()
    # Apply per-step audio directive (start/stop/switch song) before any
    # robot frames go out, so the audio head-start lines up with frame 0.
    if audio is not None:
        await audio.apply_step(step)
    # IMPORTANT: do NOT inject our own Select/Start press before a recording
    # step. The recording carries its own button presses and an interfering
    # Select press can leave the Pro in a state where the recording's stand-up
    # is ignored.
    #
    # However, recordings often start with `L2+A` (Lock posture toggle), which
    # depends on the robot's current internal state and may be ignored when
    # the robot was freshly StandDown'd by our pre-show setup. To make the
    # initial stand-up reliable, we proactively send `L2+X` (Stand up from
    # fall -- unconditional) before letting the recording's own keys run.
    # This is safe because L2+X on an already-standing robot is a no-op.
    sys.stdout.write(
        f"\n  [step {step_idx}/{total_steps}] recording start "
        f"({step.num_frames} frames, {step.duration:.1f}s) -- "
        f"sending L2+X (Stand up from fall) for reliable wake-up\n"
    )
    sys.stdout.flush()
    # L2+X = 0x0020 | 0x0400 -- mirrors RobotPlayer.STAND_KEYS.
    # IMPORTANT: send for exactly 1 frame (50ms) then release. Multi-frame
    # holds trigger the firmware's repeat/toggle handler and cycle state.
    # Skip robots under manual_override -- the operator owns them.
    wake_targets = [pl for pl in players
                    if pl.is_alive() and not pl.manual_override]
    for pl in wake_targets:
        pl.send_controller_keys(0x0020 | 0x0400)
    await asyncio.sleep(0.05)
    for pl in wake_targets:
        pl.send_controller_keys(0)
    # Give the firmware ~1.5s to actually execute the stand-up before the
    # recording's first frames start streaming.
    await asyncio.sleep(1.5)

    num_frames = step.num_frames
    t_step_start = time.monotonic()
    prev_dpad = 0
    # Track previous-frame keys per player so we can log button presses as
    # they happen. Useful for debugging "why didn't the robot stand up".
    prev_keys: dict[str, int] = {p.name: 0 for p in players}

    for frame_i in range(num_frames):
        if gamepad_active:
            prev_dpad = _handle_dpad(players, prev_dpad, gamepad_state)

        live = [p for p in players if p.is_alive()]
        if not live:
            sys.stdout.write(f"\n\n  ERROR: All robots disconnected mid-step. Aborting.\n")
            sys.stdout.flush()
            return False

        for p in live:
            if p.manual_override:
                _send_takeover_frame([p], gamepad_state, gamepad_active)
                if p.frame_index < num_frames:
                    p.frame_index += 1
            else:
                frame = p.get_recording_frame(step)
                if frame:
                    p.send(frame)
                    # Log every rising-edge button change so we can see what
                    # the recording is asking the robot to do at this moment.
                    keys_now = int(frame.get("keys", 0))
                    pressed = keys_now & ~prev_keys[p.name]
                    if pressed:
                        t_in_step = frame_i * SEND_RATE
                        sys.stdout.write(
                            f"\n  [t={t_in_step:5.2f}s {p.name}] "
                            f"button press keys={keys_now:#06x} "
                            f"({_describe_keys(keys_now)})\n"
                        )
                        sys.stdout.flush()
                    prev_keys[p.name] = keys_now

        if frame_i % 20 == 0:
            now = time.monotonic()
            elapsed_step = now - t_step_start
            elapsed_show = now - t_start_show
            pct = (frame_i + 1) / num_frames * 100
            statuses = []
            for p in players:
                if not p.is_alive():
                    mode = "X"
                else:
                    mode = "M" if p.manual_override else "A"
                statuses.append(f"{p.name[:8]}[{mode}]")
            sys.stdout.write(
                f"\r  show={elapsed_show:5.1f}s step{step_idx}/{total_steps} "
                f"({step.kind} {pct:5.1f}%) | {' | '.join(statuses)}\033[K"
            )
            sys.stdout.flush()

        expected = t_step_start + (frame_i + 1) * SEND_RATE
        sleep_for = expected - time.monotonic()
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)

    return True


async def run_algo_step(step: AlgoStep, players: list,
                         gamepad_state, gamepad_active: bool,
                         step_idx: int, total_steps: int,
                         t_offset: float, t_start_show: float,
                         audio: "AudioPlayer | None" = None,
                         prep_mode: bool = True,
                         settle_after: bool = True):
    """Play one algorithmic move on every robot in lock-step.

    Each tick we evaluate move.frame(t, ctx), scale by amplitude, clamp, and
    broadcast as a sport request to every (live, non-overridden) robot.

    prep_mode    -- if False, skip the Select/Start press + 400ms settle.
                    Use when the previous step already left the robot in
                    the correct gait mode (e.g. consecutive Euler-only
                    algo steps). Saves ~550ms of dead air per transition.
    settle_after -- if False, skip the zero-Euler "leak-prevention" send
                    at the end of the step. Use when the next step is
                    another algo with the same requires_walking, so the
                    very next frame will overwrite the pose anyway.
    """
    transition = "seamless" if not prep_mode else (
        "Start (walking) + RecoveryStand"
        if step.move.requires_walking
        else "Select (standing) + RecoveryStand"
    )
    sys.stdout.write(
        f"\n  [step {step_idx}/{total_steps}] algo {step.move.name} "
        f"({step.duration:.1f}s, tempo x{step.tempo:.2f} amp x{step.amplitude:.2f}, "
        f"requires_walking={step.move.requires_walking}) -- "
        f"mode: {transition}\n"
    )
    sys.stdout.flush()
    # Apply per-step audio directive before mode prep so they overlap in
    # time (mode prep is ~400ms which roughly matches audio head_start).
    if audio is not None:
        await audio.apply_step(step)
    # Switch mode (Select for pose / Start for walking) for all robots --
    # but only when needed. Skipping this is the key to seamless algo->algo
    # transitions: the previous step already put the robot in the right
    # mode, so the very next frame can drop straight into the new motion.
    if prep_mode:
        # Defensive trip-recovery: fire RecoveryStand on every live, non-
        # overridden robot. RecoveryStand (api 1006) animates a stand-up
        # only if the robot has actually fallen; on an upright robot it's
        # a no-op. Done CONCURRENTLY with the controller-mode press so it
        # adds zero wall-clock time to this branch -- both the recovery
        # request and the Select/Start press complete inside the existing
        # ~600ms window. Seamless algo->algo (prep_mode=False) paths are
        # untouched and remain gap-free.
        recovery_targets = [pl for pl in players
                            if pl.is_alive() and not pl.manual_override]
        recovery_task = asyncio.gather(*[
            pl.send_sport(SPORT_CMD["RecoveryStand"])
            for pl in recovery_targets
        ])
        await asyncio.gather(
            recovery_task,
            _broadcast_mode_press(players, walking=step.move.requires_walking),
        )
        await asyncio.sleep(0.4)

    ctx = {"beat_hz": 1.0 * step.tempo, "beat_phase": 0.0}
    num_frames = step.num_frames
    t_step_start = time.monotonic()
    prev_dpad = 0

    for frame_i in range(num_frames):
        if gamepad_active:
            prev_dpad = _handle_dpad(players, prev_dpad, gamepad_state)

        live = [p for p in players if p.is_alive()]
        if not live:
            sys.stdout.write(f"\n\n  ERROR: All robots disconnected mid-step. Aborting.\n")
            sys.stdout.flush()
            return False

        # Takeover frames (live gamepad state for overridden robots).
        _send_takeover_frame(live, gamepad_state, gamepad_active)

        # Algo command for the rest.
        t_in_step = (frame_i + 1) * SEND_RATE  # advance based on planned frame
        cmd = step.move.frame(t_in_step, ctx)
        cmd = _scale_cmd(cmd, step.amplitude)
        await _broadcast_sport_cmd(live, cmd)

        if frame_i % 20 == 0:
            now = time.monotonic()
            elapsed_step = now - t_step_start
            elapsed_show = now - t_start_show
            pct = (frame_i + 1) / num_frames * 100
            statuses = []
            for p in players:
                if not p.is_alive():
                    mode = "X"
                else:
                    mode = "M" if p.manual_override else "A"
                statuses.append(f"{p.name[:8]}[{mode}]")
            sys.stdout.write(
                f"\r  show={elapsed_show:5.1f}s step{step_idx}/{total_steps} "
                f"({step.kind} {pct:5.1f}%) | {' | '.join(statuses)}\033[K"
            )
            sys.stdout.flush()

        expected = t_step_start + (frame_i + 1) * SEND_RATE
        sleep_for = expected - time.monotonic()
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)

    # Zero out Euler/BodyHeight/Move so we don't leak pose state into the
    # next step. Recording steps will overwrite immediately anyway. For
    # consecutive algo steps in the same mode we skip this (settle_after=
    # False) so the new move can drive straight in without a 50ms blip
    # of "flat pose".
    if settle_after:
        await _broadcast_sport_cmd(live, {"type": "euler",
                                           "params": {"x": 0.0, "y": 0.0, "z": 0.0}})
    return True


async def run_action_step(step: ActionStep, players: list,
                           gamepad_state, gamepad_active: bool,
                           step_idx: int, total_steps: int,
                           t_offset: float, t_start_show: float,
                           audio: "AudioPlayer | None" = None):
    """Fire a single Sport API command on every live robot.

    No mode prep, no waveform loop -- we just publish the api_id once and
    optionally wait `duration` seconds so the next step doesn't pre-empt
    the action mid-play. Use `duration: 0` (default) to immediately move on.
    """
    sys.stdout.write(
        f"\n  [step {step_idx}/{total_steps}] action {step.name} "
        f"(api_id={step.api_id}, duration={step.duration:.1f}s)\n"
    )
    sys.stdout.flush()

    if audio is not None:
        await audio.apply_step(step)

    live = [p for p in players if p.is_alive() and not p.manual_override]
    if not live:
        sys.stdout.write("  WARN: no live, non-overridden robots -- skipping action\n")
        sys.stdout.flush()
        # Still honour duration so step timing stays correct in the show clock.
        if step.duration > 0:
            await asyncio.sleep(step.duration)
        return True

    # Broadcast the api_id to all live robots in parallel.
    async def _fire(p):
        try:
            await p.send_sport(step.api_id)
        except Exception as exc:
            sys.stdout.write(f"  [{p.name}] action {step.name} FAILED: {exc}\n")
            sys.stdout.flush()

    await asyncio.gather(*[_fire(p) for p in live])

    # Wait the requested duration (in small slices so a disconnect is
    # noticed and we can abort if every robot drops). Throughout the wait
    # we keep D-pad takeover responsive so the operator can grab any robot
    # mid-action -- useful for e.g. pulling one dog away while the others
    # stretch.
    prev_dpad = 0
    if step.duration > 0:
        t_end = time.monotonic() + step.duration
        while time.monotonic() < t_end:
            if not any(p.is_alive() for p in players):
                sys.stdout.write("\n  All robots disconnected -- aborting.\n")
                sys.stdout.flush()
                return False
            if gamepad_active:
                prev_dpad = _handle_dpad(players, prev_dpad, gamepad_state)
                _send_takeover_frame(players, gamepad_state, gamepad_active)
            await asyncio.sleep(min(SEND_RATE, t_end - time.monotonic()))

    return True


# ─── Main show orchestrator ──────────────────────────────────────────────

async def run_show(config: dict, no_music: bool = False, audio_head_start: float = 0.5,
                    show_log_path: str | None = "",
                    show_log_rate: float = 5.0):
    """Main show loop: connect once, run each step in sequence on all robots.

    show_log_path semantics:
      - None      -> logging disabled
      - ""        -> logging enabled, auto-generated filename under data/showlogs/
      - <path>    -> logging enabled, write to the given path
    """
    robots_cfg = config["robots"]
    song_path = config.get("song")

    # 1. Build the script up front (loads parquet files, validates names).
    steps = build_script(config)
    if not steps:
        print("ERROR: empty script")
        return
    print_script(steps)

    # 2. Build player objects (no per-robot script anymore -- it's shared).
    players: list[RobotPlayer] = []
    for rcfg in robots_cfg:
        players.append(RobotPlayer(
            name=rcfg["name"], ip=rcfg["ip"],
            aes_key=rcfg.get("aes_key"),
            is_pro=rcfg.get("pro", False),
        ))

    # 3. Connect everyone in parallel.
    print("Connecting to robots...")
    connect_tasks = [p.connect() for p in players]
    results = await asyncio.gather(*connect_tasks, return_exceptions=True)
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"  [{players[i].name}] FAILED: {r}")

    connected_players = [p for p in players if p.connected]
    if not connected_players:
        print("ERROR: No robots connected!")
        return

    # Telemetry logger: subscribe each robot's WebRTC streams (sportmode +
    # lowstate) and write a JSONL trace to data/showlogs/. Default-on so we
    # always have an artefact when something looks wrong on the floor.
    show_logger: ShowLogger | None = None
    robot_streams: list[RobotStreams] = []
    if show_log_path is not None:
        try:
            show_logger = ShowLogger.create(
                path=(show_log_path or None),
                rate_hz=show_log_rate,
            )
            print(f"\n  Show log: {show_logger.path}  ({show_log_rate:g} Hz)")
            for p in connected_players:
                # Camera/lidar are off: we want pose + lowstate only, cheap.
                streams = RobotStreams(
                    enable_camera=False, enable_lidar=False, enable_voxel=False,
                )
                streams.attach(p.conn)
                robot_streams.append(streams)
                show_logger.attach(p, streams)
        except Exception as exc:
            print(f"\n  WARN: show log setup failed ({type(exc).__name__}: {exc}); "
                  f"continuing without telemetry log")
            show_logger = None

    # 4. Set the starting posture (from the FIRST step only, which is the
    # entry into the show -- subsequent steps handle their own mode prep).
    first_step = steps[0]
    if isinstance(first_step, RecordingStep):
        start_posture = first_step.start_posture
    else:
        start_posture = "standing"   # algo steps need standing
    print(f"\nSetting start posture: {start_posture}")
    await asyncio.gather(*[p.set_start_posture(start_posture)
                           for p in connected_players])
    await asyncio.sleep(2)

    # 5. Setup gamepad for takeover control.
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
            # Initialise haptic feedback for takeover transitions.
            # Silently degrades if the pad lacks FF_RUMBLE support.
            global _RUMBLE
            _RUMBLE = RumbleHelper(gamepad_dev)
            if _RUMBLE.available:
                print(f"  Haptic feedback: ON (will pulse on takeover)")
            else:
                print(f"  Haptic feedback: not supported by this pad")
        else:
            print(f"\n  Gamepad warnings: {warnings}")
    else:
        print("\n  No gamepad found — takeover disabled")

    gamepad_active = gamepad_thread is not None

    # 6. Pre-show controls printout.
    print("\n" + "=" * 60)
    print("  SHOW CONTROLS (active before, during, and after the show):")
    for i, p in enumerate(connected_players[:4]):
        dirs = ["Up", "Down", "Left", "Right"]
        print(f"    D-pad {dirs[i]:<5} -> takeover/release {p.name}")
    print("    Sticks       -> teleop the robot currently under takeover")
    print("    Ctrl+C       -> exit (script keeps D-pad alive until you Ctrl+C)")
    print("=" * 60)

    # 7. Pre-show positioning loop: D-pad takeover active, waiting for ENTER.
    print("\n  D-pad active for positioning. Press ENTER to start the show...")
    enter_pressed = asyncio.Event()

    def _wait_enter():
        input()
        enter_pressed.set()

    asyncio.get_event_loop().run_in_executor(None, _wait_enter)

    prev_dpad = 0
    while not enter_pressed.is_set():
        if gamepad_active:
            prev_dpad = _handle_dpad(connected_players, prev_dpad, gamepad_state)
            _send_takeover_frame(connected_players, gamepad_state, gamepad_active)
        await asyncio.sleep(SEND_RATE)

    # Clear overrides for show start.
    for p in connected_players:
        p.manual_override = False

    # 8. Audio player. We pre-seed it with the global `song:` so steps that
    # don't specify their own song still get music. Per-step `song:` overrides.
    audio = AudioPlayer(head_start=audio_head_start) if not no_music else None
    if audio is not None and song_path:
        # Synthesise an initial "step" that just starts the global song. We
        # reuse AudioPlayer.apply_step by faking a minimal object with the
        # right two fields.
        class _InitialAudio:
            song = song_path
            song_start = 0.0
        await audio.apply_step(_InitialAudio())

    # 9. Run each step in sequence.
    print("\n  Show started!\n")
    t_show_start = time.monotonic()
    t_step_offset = 0.0
    aborted = False

    if show_logger is not None:
        show_logger.start(t_show_start=t_show_start)

    for i, step in enumerate(steps, 1):
        if show_logger is not None:
            show_logger.note_step(i, len(steps), step)
        # Look at neighbours to decide whether this step can transition
        # seamlessly from the previous one / into the next one.
        prev = steps[i - 2] if i > 1 else None
        nxt = steps[i] if i < len(steps) else None

        if isinstance(step, RecordingStep):
            ok = await run_recording_step(
                step, connected_players, gamepad_state, gamepad_active,
                i, len(steps), t_step_offset, t_show_start, audio=audio,
            )
        elif isinstance(step, AlgoStep):
            # prep_mode: skip the Select/Start press + 400ms settle when the
            # previous step is another AlgoStep with the same gait
            # requirement. Recording/Action steps leave the robot in an
            # unknown mode (recordings can change mode mid-clip), so any
            # non-algo predecessor forces a fresh prep.
            prep_mode = not (
                isinstance(prev, AlgoStep)
                and prev.move.requires_walking == step.move.requires_walking
            )
            # settle_after: skip the zero-Euler send when the next step is
            # another same-mode AlgoStep that will overwrite immediately.
            settle_after = not (
                isinstance(nxt, AlgoStep)
                and nxt.move.requires_walking == step.move.requires_walking
            )
            ok = await run_algo_step(
                step, connected_players, gamepad_state, gamepad_active,
                i, len(steps), t_step_offset, t_show_start, audio=audio,
                prep_mode=prep_mode, settle_after=settle_after,
            )
        elif isinstance(step, ActionStep):
            ok = await run_action_step(
                step, connected_players, gamepad_state, gamepad_active,
                i, len(steps), t_step_offset, t_show_start, audio=audio,
            )
        else:
            print(f"\n  WARN: unknown step type {type(step).__name__}, skipping")
            continue

        if not ok:
            aborted = True
            break
        t_step_offset += step.duration

    elapsed = time.monotonic() - t_show_start
    if aborted:
        print(f"\n\n  Show aborted after {elapsed:.1f}s.")
        if show_logger is not None:
            show_logger.note_event("show_aborted", elapsed_s=round(elapsed, 3))
    else:
        print(f"\n\n  Show complete in {elapsed:.1f}s.")
        if show_logger is not None:
            show_logger.note_event("show_complete", elapsed_s=round(elapsed, 3))
    print("  D-pad still active for repositioning. Ctrl+C to exit.")

    if audio is not None:
        audio.shutdown()

    # 10. Post-show positioning loop: keep D-pad active until Ctrl+C.
    # Wrap in try/finally so the JSONL log is flushed/closed cleanly even
    # when the operator hits Ctrl+C (the most common exit path here).
    try:
        prev_dpad = 0
        while True:
            if gamepad_active:
                prev_dpad = _handle_dpad(connected_players, prev_dpad, gamepad_state)
                _send_takeover_frame(connected_players, gamepad_state, gamepad_active)
            await asyncio.sleep(SEND_RATE)
    finally:
        if show_logger is not None:
            try:
                show_logger.note_event("ctrl_c_or_exit")
                await show_logger.aclose()
            except Exception:
                pass


async def main():
    parser = argparse.ArgumentParser(description="Multi-robot choreography show")
    parser.add_argument("config", help="Show config YAML file")
    parser.add_argument("--no-music", action="store_true", help="Skip music playback")
    parser.add_argument("--audio-head-start", type=float, default=0.5,
                       help="Seconds to let audio play before moves start")
    parser.add_argument("--no-show-log", action="store_true",
                       help="Disable per-show telemetry log (default: log to data/showlogs/show_<timestamp>.jsonl)")
    parser.add_argument("--show-log",
                       help="Override path for the show log file (implies enabled)")
    parser.add_argument("--show-log-rate", type=float, default=5.0,
                       help="Show log sampling rate in Hz (default: 5)")
    args = parser.parse_args()

    config = load_show_config(args.config)
    await run_show(
        config,
        no_music=args.no_music,
        audio_head_start=args.audio_head_start,
        show_log_path=(None if args.no_show_log else (args.show_log or "")),
        show_log_rate=args.show_log_rate,
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n  Show ended. Disconnecting...")
