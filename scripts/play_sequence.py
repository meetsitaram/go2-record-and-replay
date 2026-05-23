#!/usr/bin/env python3
"""
Play a sequence of algorithmic moves back-to-back on a single robot.

Reads a sequence file describing each move's name, duration, tempo, and
amplitude, connects once, plays everything, then disconnects cleanly.
Optionally plays a song in parallel.

Sequence file format -- YAML (preferred):

    moves:
      - name: headbang
        duration: 8.0
        tempo: 0.5
        amplitude: 1.0
      - name: roll_double_headbang
        duration: 8.0
        tempo: 2.0
        amplitude: 2.0

Or plain text (one move per line):

    # comments and blank lines are ignored
    headbang             duration=8 tempo=0.5 amplitude=1.0
    roll_double_headbang duration=8 tempo=2.0 amplitude=2.0
    head_swivel          duration=8 tempo=1.0 amplitude=1.5
    head_swivel_nod_sync duration=8 tempo=2.0 amplitude=2.0

Usage:
    .venv/bin/python scripts/play_sequence.py \
        --ip 192.168.1.246 --aes-key KEY \
        --sequence config/test_sequence.yaml

    .venv/bin/python scripts/play_sequence.py \
        --ip 192.168.1.246 --aes-key KEY \
        --sequence config/test_sequence.yaml \
        --song ../assets/third-song.m4a
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from unitree_webrtc_connect.constants import SPORT_CMD
from go2_driver.connection import Go2Connection

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
from go2_recorder.algo_moves_lib import CATALOGUE_BY_NAME  # noqa: E402

# Reuse the heavy lifting from algo_moves.py
from algo_moves import (  # noqa: E402
    StateMonitor, run_move, press_select, _send_sport,
)


@dataclass
class SequenceStep:
    name: str
    duration: float
    tempo: float = 1.0
    amplitude: float = 1.0


def parse_kv_line(line: str) -> SequenceStep:
    """Parse a text-format line like 'name k=v k=v ...'."""
    parts = line.split()
    name = parts[0]
    kvs = {}
    for tok in parts[1:]:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        kvs[k.strip()] = v.strip()
    return SequenceStep(
        name=name,
        duration=float(kvs.get("duration", 8.0)),
        tempo=float(kvs.get("tempo", 1.0)),
        amplitude=float(kvs.get("amplitude", 1.0)),
    )


def load_sequence(path: Path) -> list[SequenceStep]:
    text = path.read_text()
    steps: list[SequenceStep] = []

    # Try YAML first.
    try:
        data = yaml.safe_load(text)
        if isinstance(data, dict) and "moves" in data:
            for entry in data["moves"]:
                steps.append(SequenceStep(
                    name=entry["name"],
                    duration=float(entry.get("duration", 8.0)),
                    tempo=float(entry.get("tempo", 1.0)),
                    amplitude=float(entry.get("amplitude", 1.0)),
                ))
            return steps
    except yaml.YAMLError:
        pass  # fall through to plain-text parser

    # Plain text fallback: one move per line, kv pairs after the name.
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        steps.append(parse_kv_line(line))
    return steps


def validate_steps(steps: list[SequenceStep]) -> None:
    unknown = [s.name for s in steps if s.name not in CATALOGUE_BY_NAME]
    if unknown:
        print(f"ERROR: unknown move name(s) in sequence: {unknown}")
        print(f"Available: {sorted(CATALOGUE_BY_NAME)}")
        sys.exit(1)


def print_plan(steps: list[SequenceStep]) -> None:
    total = sum(s.duration for s in steps)
    print(f"\n  Sequence: {len(steps)} moves, total {total:.1f}s")
    print(f"  {'#':>2}  {'move':<24} {'dur':>5}  {'tempo':>5}  {'amp':>4}")
    print(f"  {'-'*2}  {'-'*24} {'-'*5}  {'-'*5}  {'-'*4}")
    for i, s in enumerate(steps, 1):
        print(f"  {i:>2}  {s.name:<24} {s.duration:>4.1f}s  "
              f"{s.tempo:>5.2f}  {s.amplitude:>4.2f}")
    print()


async def play_sequence(go2: Go2Connection, conn, monitor: StateMonitor,
                        steps: list[SequenceStep], send_hz: float = 20.0):
    """Run each step sequentially with seamless transitions.

    To eliminate the ~1s pause between moves, we skip the per-move mode-prep
    and settle phases unless the mode actually needs to change (walking <->
    non-walking). Result: consecutive Euler-only moves switch instantly.
    """
    t_start = time.monotonic()
    for i, step in enumerate(steps, 1):
        print(f"\n========== Step {i}/{len(steps)} ==========")
        move = CATALOGUE_BY_NAME[step.name]

        prev = CATALOGUE_BY_NAME[steps[i - 2].name] if i > 1 else None
        nxt = CATALOGUE_BY_NAME[steps[i].name] if i < len(steps) else None

        # Prep mode only if we just connected (first step) or the previous
        # move had a different gait requirement than this one.
        prep_mode = (prev is None) or (prev.requires_walking != move.requires_walking)
        # Settle only if this is the last step, or the NEXT move needs a
        # different gait (so we have to drop into balance-stand to transition).
        settle_after = (nxt is None) or (nxt.requires_walking != move.requires_walking)

        ok = await run_move(
            conn, move, go2, monitor,
            duration_override=step.duration,
            send_hz=send_hz,
            tempo=step.tempo,
            amplitude=step.amplitude,
            prep_mode=prep_mode,
            settle_after=settle_after,
        )
        if not ok:
            print(f"  [step {i}] aborted (connection lost). Stopping sequence.")
            return

    elapsed = time.monotonic() - t_start
    print(f"\n  Sequence complete in {elapsed:.1f}s.")


def start_music(song_path: Path, head_start: float) -> subprocess.Popen | None:
    if not song_path.exists():
        print(f"  WARN: song file not found: {song_path}")
        return None
    proc = subprocess.Popen(
        ["ffplay", "-nodisp", "-autoexit", str(song_path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if head_start > 0:
        time.sleep(head_start)
    return proc


async def main():
    parser = argparse.ArgumentParser(description="Play a sequence of algo moves")
    parser.add_argument("--ip", required=True, help="Robot IP address")
    parser.add_argument("--aes-key", default=None, help="AES-128 key (newer firmware)")
    parser.add_argument("--sequence", required=True,
                        help="Path to sequence YAML or text file")
    parser.add_argument("--rate", type=float, default=20.0,
                        help="Send rate in Hz (default 20)")
    parser.add_argument("--song", default=None,
                        help="Optional song file to play in parallel")
    parser.add_argument("--audio-head-start", type=float, default=0.5,
                        help="Seconds to let audio play before moves start "
                             "(compensates Bluetooth latency)")
    parser.add_argument("--no-wait", action="store_true",
                        help="Skip the 'press ENTER to start' prompt")
    args = parser.parse_args()

    seq_path = Path(args.sequence)
    if not seq_path.exists():
        print(f"ERROR: sequence file not found: {seq_path}")
        sys.exit(1)

    steps = load_sequence(seq_path)
    if not steps:
        print("ERROR: sequence file has no steps")
        sys.exit(1)
    validate_steps(steps)
    print_plan(steps)

    print(f"Connecting to {args.ip}...")
    go2 = Go2Connection("sta", args.ip, aes_key=args.aes_key)
    conn = await go2.async_connect()
    print("Connected!")

    monitor = StateMonitor()
    conn.datachannel.pub_sub.subscribe("rt/lf/sportmodestate", monitor.on_msg)
    await asyncio.sleep(1.0)

    audio_proc: subprocess.Popen | None = None
    try:
        # Bring up to BalanceStand before the sequence starts.
        await press_select(conn)
        await asyncio.sleep(2.0)
        print(f"  Robot ready. mode={monitor.mode_name()} "
              f"height={monitor.body_height:.3f}m")

        if not args.no_wait:
            print("\n  Press ENTER to start the sequence...")
            await asyncio.get_event_loop().run_in_executor(None, input)

        if args.song:
            print(f"  Starting music: {args.song}")
            audio_proc = start_music(Path(args.song), args.audio_head_start)

        await play_sequence(go2, conn, monitor, steps, send_hz=args.rate)

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        if audio_proc and audio_proc.poll() is None:
            audio_proc.terminate()
        try:
            await _send_sport(conn, SPORT_CMD["Euler"], {"x": 0.0, "y": 0.0, "z": 0.0})
            await _send_sport(conn, SPORT_CMD["BodyHeight"], {"data": 0.0})
            await press_select(conn)
            await asyncio.sleep(0.5)
        except Exception:
            pass
        await go2.async_disconnect()


if __name__ == "__main__":
    asyncio.run(main())
