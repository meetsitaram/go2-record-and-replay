#!/usr/bin/env python3
"""
Interactive explorer for algorithmic dance moves on the Go2 (Air / Pro).

Walks through a curated catalogue of parametric moves (sine-wave Euler tilts,
BodyHeight bounces, Move strafes, and combinations thereof). For each move:

    ENTER  -> play it
    r      -> replay the last move
    s      -> save it to config/favorite_moves.yaml
    n      -> skip / advance to next
    q      -> quit

Each move sends Euler / BodyHeight / Move sport requests at 20 Hz while it
plays, then returns the robot to BalanceStand before the next prompt.

Usage:
    .venv/bin/python scripts/algo_moves.py --ip 192.168.1.246 \\
        --aes-key 2c09e23856fa423ed680313dd939a3f0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import yaml

from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD
from go2_driver.connection import Go2Connection

# Make the local src/ importable so `algo_moves_lib` resolves without install.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
from go2_recorder.algo_moves_lib import (  # noqa: E402
    CATALOGUE, CATALOGUE_BY_NAME, Move, clamp_params,
)


SEND_RATE = 1.0 / 20  # 20 Hz
FAVORITES_PATH = _HERE.parent / "config" / "favorite_moves.yaml"


# ── Robot state monitor ─────────────────────────────────────────────────

class StateMonitor:
    """Tracks live robot state from rt/lf/sportmodestate."""

    def __init__(self):
        self.mode = -1
        self.body_height = 0.0
        self.gait_type = -1
        self.last_update = 0.0

    def on_msg(self, msg):
        try:
            data = msg.get("data", {})
            self.mode = data.get("mode", -1)
            self.body_height = data.get("body_height", 0.0)
            self.gait_type = data.get("gait_type", -1)
            self.last_update = time.monotonic()
        except Exception:
            pass

    def mode_name(self) -> str:
        # Common Go2 sportmode codes seen in practice.
        return {
            0: "idle",
            1: "balance_stand",
            2: "pose",
            3: "locomotion",
            4: "lay_down",
            5: "stand_up",
            6: "damping",
            7: "recovery",
            9: "sit",
        }.get(self.mode, f"unknown({self.mode})")


# ── Sport-request helpers ────────────────────────────────────────────────

# Track accept / reject counts per api_id across a move's playback.
_REJECT_COUNTS: dict[int, int] = {}
_ACCEPT_COUNTS: dict[int, int] = {}


async def _send_sport(conn, api_id: int, params=None,
                      check_status: bool = False, verbose: bool = False):
    """Send a sport request. If check_status, parse and report rejection codes."""
    opts = {"api_id": api_id}
    if params is not None:
        opts["parameter"] = params
    try:
        resp = await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"], opts
        )
        if check_status and resp is not None:
            try:
                code = resp.get("data", {}).get("header", {}).get("status", {}).get("code", 0)
                if code != 0:
                    cnt = _REJECT_COUNTS.get(api_id, 0)
                    if cnt < 3 or verbose:
                        sys.stdout.write(
                            f"\n  [REJECTED] api_id={api_id} code={code} params={params}\n"
                        )
                    _REJECT_COUNTS[api_id] = cnt + 1
                else:
                    _ACCEPT_COUNTS[api_id] = _ACCEPT_COUNTS.get(api_id, 0) + 1
            except Exception:
                pass
    except Exception as e:
        sys.stdout.write(f"\n  [send error] api_id={api_id}: {e}\n")


async def _send_controller(conn, keys: int = 0,
                           lx: float = 0.0, ly: float = 0.0,
                           rx: float = 0.0, ry: float = 0.0):
    """Send a raw controller message on rt/wirelesscontroller (same as record.py)."""
    msg = json.dumps({
        "type": "msg",
        "topic": "rt/wirelesscontroller",
        "data": {"lx": lx, "ly": ly, "rx": rx, "ry": ry, "keys": keys},
    })
    try:
        conn.datachannel.channel.send(msg)
    except Exception as e:
        sys.stdout.write(f"\n  [controller send error] keys={keys}: {e}\n")


# Controller key bits (from go2_driver.constants).
KEY_SELECT = 0x0008  # press = "Standing mode" (BalanceStand)
KEY_START = 0x0004   # press = "Walking mode"


async def press_select(conn):
    """Press Select on the controller -> robot enters BalanceStand."""
    # Hold for ~150ms so the firmware registers the press.
    for _ in range(3):
        await _send_controller(conn, keys=KEY_SELECT)
        await asyncio.sleep(0.05)
    await _send_controller(conn, keys=0)


async def press_start(conn):
    """Press Start on the controller -> robot enters Walking mode."""
    for _ in range(3):
        await _send_controller(conn, keys=KEY_START)
        await asyncio.sleep(0.05)
    await _send_controller(conn, keys=0)


async def balance_stand(conn):
    """Return robot to a clean balance stand via the controller channel."""
    await press_select(conn)


async def stop_move(conn):
    await _send_controller(conn, keys=0)


async def send_command(conn, cmd: dict, check_status: bool = False):
    """Translate one frame dict into one or more sport requests."""
    cmd = clamp_params(cmd)
    t = cmd["type"]
    p = cmd["params"]
    if t == "euler":
        await _send_sport(conn, SPORT_CMD["Euler"], p, check_status=check_status)
    elif t == "body_height":
        await _send_sport(conn, SPORT_CMD["BodyHeight"], p, check_status=check_status)
    elif t == "move":
        await _send_sport(conn, SPORT_CMD["Move"], p, check_status=check_status)
    elif t == "compound":
        # Send each sub-command. They go out near-simultaneously.
        for sub in p:
            await send_command(conn, sub, check_status=check_status)


# ── Move runner ──────────────────────────────────────────────────────────

def _scale_cmd(cmd: dict, amp: float) -> dict:
    """Multiply a command's spatial params by `amp` (amplitude multiplier).

    Recurses into compound commands. Final clamping in `send_command` /
    `clamp_params` still applies, so amp values that would exceed safety
    limits get capped (not an error -- just an upper bound).
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


async def run_move(conn, move: Move, conn_wrapper: Go2Connection,
                   monitor: StateMonitor,
                   duration_override: float | None = None,
                   send_hz: float = 20.0,
                   tempo: float = 1.0,
                   amplitude: float = 1.0,
                   prep_mode: bool = True,
                   settle_after: bool = True):
    """Play a move for its duration, sending frames at `send_hz`.

    tempo:        multiplier on ctx["beat_hz"] -- speeds up all sine frequencies.
    amplitude:    multiplier on all Euler/BodyHeight/Move param magnitudes
                  (clamped by safety limits in send_command).
    prep_mode:    if True (default), press the appropriate controller button
                  and wait 500ms before sending frames. Set False when chaining
                  consecutive moves that need the same mode (no mode switch).
    settle_after: if True (default), zero out Euler/BodyHeight/Move and press
                  Select for ~800ms after the move. Set False when chaining --
                  the next move's first frame takes over immediately.
    """
    period = 1.0 / send_hz
    duration = duration_override if duration_override is not None else move.duration
    # ctx["beat_hz"] feeds every move's frequency computation; multiplying it
    # by `tempo` cleanly speeds up or slows down the move without touching the
    # move's source. Default tempo=1.0 reproduces the original behaviour.
    ctx = {"beat_hz": 1.0 * tempo, "beat_phase": 0.0}

    tag = ""
    if abs(tempo - 1.0) > 1e-3 or abs(amplitude - 1.0) > 1e-3:
        tag = f"  [tempo x{tempo:.2f}  amp x{amplitude:.2f}]"
    print(f"\n  > playing {move.name} for {duration:.1f}s "
          f"(category {move.category}, beats/cycle={move.beats_per_cycle}){tag}")
    print(f"    pre-move state: mode={monitor.mode_name()} "
          f"height={monitor.body_height:.3f}m gait={monitor.gait_type}")

    # Press the appropriate controller button to switch mode. This is what
    # the actual Xbox controller does -- direct sport-API SelectMode/BalanceStand
    # calls don't fully transition the robot on the Pro.
    if prep_mode:
        if move.requires_walking:
            await press_start(conn)
        else:
            await press_select(conn)
        await asyncio.sleep(0.5)  # let the mode actually settle

    # Reset rejection counters so we re-report rejections for this move
    _REJECT_COUNTS.clear()
    _ACCEPT_COUNTS.clear()

    t_start = time.monotonic()
    next_tick = t_start
    last_progress = 0
    frame_count = 0
    while True:
        now = time.monotonic()
        t = now - t_start
        if t >= duration:
            break

        if not conn_wrapper.is_alive():
            sys.stdout.write("\n  [ERROR] connection lost, aborting move\n")
            return False

        cmd = move.frame(t, ctx)
        cmd = _scale_cmd(cmd, amplitude)
        # Check every frame's status so we can summarise accept/reject counts
        # at the end of the move. Rejection messages themselves are throttled
        # to the first 3 per api_id to avoid spam.
        await send_command(conn, cmd, check_status=True)
        frame_count += 1

        # Progress bar
        pct = int(100 * t / duration)
        if pct >= last_progress + 5:
            sys.stdout.write(f"\r    [{pct:3d}%] t={t:5.2f}s  "
                             f"mode={monitor.mode_name()} h={monitor.body_height:.3f}")
            sys.stdout.flush()
            last_progress = pct

        next_tick += period
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)

    sys.stdout.write(f"\r    [100%] t={duration:5.2f}s — done"
                     f"  ({frame_count} frames sent)\n")
    sys.stdout.flush()

    # Per-api accept/reject summary
    name_for_id = {SPORT_CMD["Euler"]: "Euler",
                   SPORT_CMD["BodyHeight"]: "BodyHeight",
                   SPORT_CMD["Move"]: "Move"}
    seen = set(_REJECT_COUNTS) | set(_ACCEPT_COUNTS)
    if seen:
        parts = []
        for aid in sorted(seen):
            n = name_for_id.get(aid, str(aid))
            ok = _ACCEPT_COUNTS.get(aid, 0)
            bad = _REJECT_COUNTS.get(aid, 0)
            parts.append(f"{n}: {ok} ok / {bad} rej")
        print("    " + " | ".join(parts))

    # Settle: clear Euler / BodyHeight / Move, return to balance stand.
    # Skipped when chaining consecutive moves to keep transitions seamless.
    if settle_after:
        await _send_sport(conn, SPORT_CMD["Euler"], {"x": 0.0, "y": 0.0, "z": 0.0})
        await _send_sport(conn, SPORT_CMD["BodyHeight"], {"data": 0.0})
        if move.requires_walking:
            await stop_move(conn)
        await balance_stand(conn)
        await asyncio.sleep(0.3)
    return True


# ── Favorites I/O ────────────────────────────────────────────────────────

def load_favorites() -> list[dict]:
    if not FAVORITES_PATH.exists():
        return []
    with open(FAVORITES_PATH) as f:
        data = yaml.safe_load(f) or {}
    return data.get("moves") or []


def save_favorite(move: Move, duration: float, notes: str = "",
                  tempo: float = 1.0, amplitude: float = 1.0):
    favs = load_favorites()
    entry = {
        "name": move.name,
        "catalog_id": move.name,
        "duration": float(duration),
        "category": move.category,
        "notes": notes,
    }
    # Only record tunings if they're non-default, to keep the YAML clean.
    if abs(tempo - 1.0) > 1e-3:
        entry["tempo"] = float(tempo)
    if abs(amplitude - 1.0) > 1e-3:
        entry["amplitude"] = float(amplitude)
    favs.append(entry)
    FAVORITES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FAVORITES_PATH, "w") as f:
        yaml.safe_dump({"moves": favs}, f, sort_keys=False)
    print(f"  -> saved to {FAVORITES_PATH}")


# ── Interactive loop ─────────────────────────────────────────────────────

async def ainput(prompt: str = "") -> str:
    """Async wrapper around blocking input()."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: input(prompt))


async def explorer(conn, conn_wrapper: Go2Connection, monitor: StateMonitor,
                   moves: list[Move],
                   duration_override: float | None, send_hz: float,
                   tempo: float = 1.0, amplitude: float = 1.0):
    print("\n" + "=" * 70)
    print("  ALGO MOVES EXPLORER")
    print("  Controls: [ENTER]=play  [r]=replay  [s]=save  [n]=skip  [q]=quit")
    if abs(tempo - 1.0) > 1e-3 or abs(amplitude - 1.0) > 1e-3:
        print(f"  Global multipliers: tempo x{tempo:.2f}  amplitude x{amplitude:.2f}")
    print("=" * 70)
    print(f"  {len(moves)} moves in catalogue, {len(load_favorites())} favorites saved")

    last_move: Move | None = None
    i = 0
    while i < len(moves):
        move = moves[i]
        gait_hint = "  (NEEDS WALKING)" if move.requires_walking else ""
        print(f"\n[{i+1:2d}/{len(moves)}] {move.name:20s} — {move.description}{gait_hint}")
        cmd = (await ainput("  > ")).strip().lower()

        if cmd == "q":
            print("  Quitting.")
            return
        if cmd == "n" or cmd == "skip":
            i += 1
            continue
        if cmd == "r" and last_move is not None:
            await run_move(conn, last_move, conn_wrapper, monitor,
                           duration_override, send_hz,
                           tempo=tempo, amplitude=amplitude)
            continue
        if cmd == "s" and last_move is not None:
            notes = (await ainput("    notes (optional): ")).strip()
            dur = duration_override if duration_override is not None else last_move.duration
            save_favorite(last_move, dur, notes,
                          tempo=tempo, amplitude=amplitude)
            continue
        if cmd == "" or cmd == "y" or cmd == "p":
            await run_move(conn, move, conn_wrapper, monitor,
                           duration_override, send_hz,
                           tempo=tempo, amplitude=amplitude)
            last_move = move
            i += 1
            continue
        print(f"  Unknown command '{cmd}'. Use: ENTER / r / s / n / q")


# ── Main ─────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Algorithmic moves explorer")
    parser.add_argument("--ip", required=True, help="Robot IP address")
    parser.add_argument("--aes-key", default=None, help="AES-128 key (newer firmware)")
    parser.add_argument("--start-at", default=None,
                        help="Skip ahead to this move name in the catalogue")
    parser.add_argument("--duration", type=float, default=None,
                        help="Override duration for all moves (seconds)")
    parser.add_argument("--rate", type=float, default=20.0,
                        help="Send rate in Hz (default 20)")
    parser.add_argument("--only", default=None,
                        help="Comma-separated list of move names to play")
    parser.add_argument("--tempo", type=float, default=1.0,
                        help="Tempo multiplier applied to every move's frequency "
                             "(2.0 = twice as fast, 0.5 = half speed). Default 1.0.")
    parser.add_argument("--amplitude", type=float, default=1.0,
                        help="Amplitude multiplier applied to all Euler/Move params "
                             "(1.5 = aggressive, 0.5 = subtle). Capped by safety clamps. "
                             "Default 1.0.")
    args = parser.parse_args()

    moves = list(CATALOGUE)
    if args.only:
        wanted = [n.strip() for n in args.only.split(",") if n.strip()]
        unknown = [n for n in wanted if n not in CATALOGUE_BY_NAME]
        if unknown:
            print(f"ERROR: unknown move name(s): {unknown}")
            print(f"Available: {[m.name for m in CATALOGUE]}")
            sys.exit(1)
        moves = [CATALOGUE_BY_NAME[n] for n in wanted]
    elif args.start_at:
        names = [m.name for m in moves]
        if args.start_at not in names:
            print(f"ERROR: --start-at '{args.start_at}' not in catalogue")
            print(f"Available: {names}")
            sys.exit(1)
        moves = moves[names.index(args.start_at):]

    print(f"Connecting to {args.ip}...")
    go2 = Go2Connection("sta", args.ip, aes_key=args.aes_key)
    conn = await go2.async_connect()
    print("Connected!")

    # Subscribe to robot state so we know what mode we're in.
    monitor = StateMonitor()
    conn.datachannel.pub_sub.subscribe("rt/lf/sportmodestate", monitor.on_msg)
    await asyncio.sleep(1.0)

    try:
        # Bring up to BalanceStand via Select button -- same as pressing
        # Select on the Xbox controller, which we know works.
        await press_select(conn)
        await asyncio.sleep(2.0)
        print(f"  Robot ready. mode={monitor.mode_name()} "
              f"height={monitor.body_height:.3f}m\n")

        await explorer(conn, go2, monitor, moves, args.duration, args.rate,
                       tempo=args.tempo, amplitude=args.amplitude)
    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        try:
            await _send_sport(conn, SPORT_CMD["Euler"], {"x": 0.0, "y": 0.0, "z": 0.0})
            await _send_sport(conn, SPORT_CMD["BodyHeight"], {"data": 0.0})
            await press_select(conn)  # back to standing
            await asyncio.sleep(0.5)
        except Exception:
            pass
        await go2.async_disconnect()


if __name__ == "__main__":
    asyncio.run(main())
