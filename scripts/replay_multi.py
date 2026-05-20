#!/usr/bin/env python3
"""
Replay a recorded episode on multiple Go2 robots simultaneously.

Connects to all robots listed in a YAML config file in parallel, then
replays the same episode with synchronized frame timing across all robots.
Robots that fail to connect are skipped.

Usage:
    python scripts/replay_multi.py --dataset ./data/go2-teleop --episode 0 --config config/robots.yaml
    python scripts/replay_multi.py --dataset ./data/go2-teleop --episode 0 --config config/robots.yaml --speed 0.5
    python scripts/replay_multi.py --dataset ./data/go2-teleop --episode 0 --config config/robots.yaml --dry-run
    python scripts/replay_multi.py --dataset ./data/go2-teleop --episode 0 --ip 192.168.1.133
"""

import argparse
import json
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from go2_driver.connection import Go2Connection
from go2_driver.constants import SEND_RATE
from go2_driver.posture import ensure_posture, POSTURE_STANDING, STANDING_HEIGHT_MIN, CROUCHED_HEIGHT_MAX
from go2_recorder.replayer import load_episode_actions


async def _async_send(conn, msg: str):
    conn.datachannel.channel.send(msg)


def connect_robot(name: str, ip: str, mode: str, aes_key: str | None = None) -> tuple[str, Go2Connection | None]:
    """Connect to a single robot. Returns (name, connection) or (name, None) on failure."""
    try:
        conn = Go2Connection(mode, ip, aes_key=aes_key)
        conn.connect()
        return name, conn
    except Exception as e:
        print(f"  WARNING: {name} ({ip}) failed to connect: {e}")
        return name, None


def connect_all(robots: list[dict], mode: str) -> list[tuple[str, Go2Connection]]:
    """Connect to all robots in parallel. Returns list of (name, conn) for successful connections."""
    results = []

    with ThreadPoolExecutor(max_workers=len(robots)) as pool:
        futures = [
            pool.submit(connect_robot, r["name"], r["ip"], mode, r.get("aes_key"))
            for r in robots
        ]
        for f in futures:
            name, conn = f.result()
            if conn is not None:
                results.append((name, conn))

    return results


def replay_synchronized(
    connections: list[tuple[str, Go2Connection]],
    actions: np.ndarray,
    buttons: np.ndarray,
    speed_multiplier: float = 1.0,
    dry_run: bool = False,
):
    """Replay an episode simultaneously on all connected robots."""
    num_frames = len(actions)
    interval = SEND_RATE / speed_multiplier
    robot_names = [name for name, _ in connections]

    print(f"\n  Replaying {num_frames} frames ({num_frames / 20:.1f}s at 1x)")
    if speed_multiplier != 1.0:
        print(f"  Speed: {speed_multiplier:.1f}x (interval: {interval*1000:.0f}ms)")
    print(f"  Robots: {', '.join(robot_names)}")
    print()

    send_counts = {name: 0 for name in robot_names}

    def send_to_robot(name: str, conn: Go2Connection, msg: str):
        try:
            conn.run_coroutine(_async_send(conn.conn, msg), timeout=1)
            send_counts[name] += 1
        except Exception:
            pass

    for i in range(num_frames):
        tick_start = time.monotonic()

        lx, ly, rx, ry = actions[i].tolist()
        keys = int(buttons[i, 0]) if buttons.ndim > 1 else int(buttons[i])

        state_dict = {"lx": lx, "ly": ly, "rx": rx, "ry": ry, "keys": keys}

        if dry_run:
            if i % 20 == 0 or i == num_frames - 1:
                pct = (i + 1) / num_frames * 100
                sys.stdout.write(
                    f"\r  [{pct:5.1f}%] frame {i+1}/{num_frames}  "
                    f"lx={lx:+.2f} ly={ly:+.2f} rx={rx:+.2f} ry={ry:+.2f} keys={keys:#06x}"
                    "\033[K"
                )
                sys.stdout.flush()
        else:
            msg = json.dumps({
                "type": "msg",
                "topic": "rt/wirelesscontroller",
                "data": state_dict,
            })

            with ThreadPoolExecutor(max_workers=len(connections)) as pool:
                for name, conn in connections:
                    pool.submit(send_to_robot, name, conn, msg)

            if i % 20 == 0:
                pct = (i + 1) / num_frames * 100
                sys.stdout.write(f"\r  [{pct:5.1f}%] frame {i+1}/{num_frames}\033[K")
                sys.stdout.flush()

        elapsed = time.monotonic() - tick_start
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    print(f"\n\n  Replay complete:")
    for name in robot_names:
        print(f"    {name}: {send_counts[name]} frames sent")


def main():
    p = argparse.ArgumentParser(
        description="Replay a recorded episode on multiple Go2 robots simultaneously"
    )
    p.add_argument("--dataset", required=True, help="Path to LeRobot dataset directory")
    p.add_argument("--episode", type=int, default=0, help="Episode index to replay")
    p.add_argument("--config", default=None,
                   help="Path to robots YAML config file (default: config/robots.yaml)")
    p.add_argument("--ip", default=None,
                   help="Single robot IP (overrides config, for quick single-robot replay)")
    p.add_argument("--mode", choices=["ap", "sta", "lan"], default="sta",
                   help="Connection mode (default: sta)")
    p.add_argument("--dry-run", action="store_true", help="Print actions without sending")
    p.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    p.add_argument("--skip-posture-check", action="store_true",
                   help="Skip initial posture verification (StandUp before replay)")
    p.add_argument("--posture-timeout", type=float, default=8.0,
                   help="Seconds to wait for posture confirmation (default: 8)")

    args = p.parse_args()

    print("=" * 60)
    print(f"  go2-record-and-replay: MULTI REPLAY  [{args.mode.upper()} mode]"
          + ("  [DRY RUN]" if args.dry_run else ""))
    print("=" * 60)

    # Load episode data
    actions, buttons = load_episode_actions(args.dataset, args.episode)
    if len(actions) == 0:
        print("  No frames to replay.")
        sys.exit(1)

    # Determine robot list
    config = {}
    if args.ip:
        robots = [{"name": f"go2@{args.ip}", "ip": args.ip}]
    else:
        config_path = Path(args.config) if args.config else Path("config/robots.yaml")
        if not config_path.exists():
            print(f"  ERROR: Config file not found: {config_path}")
            print("  Provide --config <path> or --ip <address>")
            sys.exit(1)
        with open(config_path) as f:
            config = yaml.safe_load(f)
        robots = config.get("robots", [])
        if not robots:
            print(f"  ERROR: No robots defined in {config_path}")
            sys.exit(1)

    print(f"\n  Target robots ({len(robots)}):")
    for r in robots:
        print(f"    {r['name']:20s}  {r['ip']}")

    # Connect to all robots
    if not args.dry_run:
        try:
            from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection  # noqa: F401
        except ImportError:
            print("  ERROR: unitree_webrtc_connect not installed.")
            sys.exit(1)

        print(f"\n  Connecting to {len(robots)} robot(s) in parallel...")
        connections = connect_all(robots, args.mode)

        if not connections:
            print("  ERROR: No robots connected successfully. Aborting.")
            sys.exit(1)

        connected_names = [name for name, _ in connections]
        skipped = [r["name"] for r in robots if r["name"] not in connected_names]

        print(f"\n  Connected: {len(connections)}/{len(robots)}")
        for name, _ in connections:
            print(f"    {name}: OK")
        for name in skipped:
            print(f"    {name}: SKIPPED")
    else:
        connections = [(r["name"], None) for r in robots]

    # Ensure all robots are in the standing posture before replay
    if not args.dry_run and not args.skip_posture_check:
        posture_cfg = config.get("posture", {}) if not args.ip else {}
        standing_min = posture_cfg.get("standing_height_min", STANDING_HEIGHT_MIN)
        crouched_max = posture_cfg.get("crouched_height_max", CROUCHED_HEIGHT_MAX)

        print(f"\n  Ensuring standing posture (timeout: {args.posture_timeout}s, "
              f"threshold: {standing_min}m)...")
        for name, conn in connections:
            ok = ensure_posture(
                conn,
                target=POSTURE_STANDING,
                timeout=args.posture_timeout,
                standing_height_min=standing_min,
                crouched_height_max=crouched_max,
            )
            status = "OK" if ok else "TIMEOUT (proceeding anyway)"
            print(f"    {name}: {status}")
        print()

    # Replay
    try:
        replay_synchronized(
            connections=connections,
            actions=actions,
            buttons=buttons,
            speed_multiplier=args.speed,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        print("\n  Replay interrupted.")
    finally:
        if not args.dry_run:
            for name, conn in connections:
                try:
                    conn.disconnect()
                except Exception:
                    pass

    print("  Done.")


if __name__ == "__main__":
    main()
