#!/usr/bin/env python3
"""
Replay a recorded episode on the Go2 robot.

Usage:
    python scripts/replay.py --dataset ./data/go2-teleop --episode 0 --mode sta --ip 192.168.1.133
    python scripts/replay.py --dataset ./data/go2-teleop --episode 0 --dry-run
    python scripts/replay.py --dataset ./data/go2-teleop --episode 0 --speed 0.5  # half speed
"""

import argparse
import sys

from go2_recorder.connection import Go2Connection
from go2_recorder.replayer import load_episode_actions, replay_episode


def main():
    p = argparse.ArgumentParser(description="Replay a recorded episode on the Go2")
    p.add_argument("--dataset", required=True, help="Path to LeRobot dataset directory")
    p.add_argument("--episode", type=int, default=0, help="Episode index to replay")
    p.add_argument("--mode", choices=["ap", "sta", "lan"], default="ap")
    p.add_argument("--ip", default=None, help="Go2 IP address (required for sta mode)")
    p.add_argument("--dry-run", action="store_true", help="Print actions without sending")
    p.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")

    args = p.parse_args()

    print("=" * 60)
    print(f"  go2-record-and-replay: REPLAY  [{args.mode.upper()} mode]"
          + ("  [DRY RUN]" if args.dry_run else ""))
    print("=" * 60)

    # Load episode data
    actions, buttons = load_episode_actions(args.dataset, args.episode)

    if len(actions) == 0:
        print("  No frames to replay.")
        sys.exit(1)

    # Connect to robot
    conn_wrapper = None
    if not args.dry_run:
        try:
            from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection
        except ImportError:
            print("  ERROR: unitree_webrtc_connect not installed.")
            sys.exit(1)

        conn_wrapper = Go2Connection(args.mode, args.ip)
        try:
            conn_wrapper.connect()
        except ConnectionError as e:
            print(f"  ERROR: {e}")
            sys.exit(1)

    # Replay
    try:
        replay_episode(
            conn_wrapper=conn_wrapper,
            actions=actions,
            buttons=buttons,
            speed_multiplier=args.speed,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        print("\n  Replay interrupted.")
    finally:
        if conn_wrapper:
            conn_wrapper.disconnect()

    print("  Done.")


if __name__ == "__main__":
    main()
