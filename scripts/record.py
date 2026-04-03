#!/usr/bin/env python3
"""
Record teleoperation episodes with an Xbox controller.

Connects to the Go2 via WebRTC, reads the Xbox gamepad, sends controls to the
robot, and simultaneously records all data streams into a LeRobot v3.0 dataset.

Episode control:
    F1 (left stick click)  -- toggle recording start/stop
    Ctrl+C                 -- finalize dataset and exit

Usage:
    python scripts/record.py --mode sta --ip 192.168.1.133 --repo-id user/go2-teleop
    python scripts/record.py --mode ap --dry-run
    python scripts/record.py --mode sta --ip 192.168.1.133 --no-camera --task "walk forward"
"""

import argparse
import asyncio
import json
import sys
import time
import threading

import numpy as np

from go2_driver.constants import KEY_F1, SEND_RATE
from go2_driver.connection import Go2Connection
from go2_driver.gamepad import (
    ControllerState, SafetyFilter, RumbleHelper,
    find_gamepad, validate_gamepad, check_device_permissions, gamepad_loop,
)
from go2_driver.streams import RobotStreams
from go2_recorder.constants import DATASET_FPS
from go2_recorder.recorder import EpisodeRecorder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record Go2 teleoperation episodes")
    p.add_argument("--mode", choices=["ap", "sta", "lan"], default="ap",
                   help="Connection mode (ap=hotspot, sta=same router, lan=ethernet)")
    p.add_argument("--ip", default=None, help="Go2 IP address (required for sta mode)")
    p.add_argument("--dry-run", action="store_true",
                   help="Run gamepad without connecting to robot or recording")
    p.add_argument("--allow-all", action="store_true",
                   help="Allow dangerous button combos (with countdown)")
    p.add_argument("--speed-limit", type=float, default=0.5, metavar="0.0-1.0",
                   help="Cap joystick output (default: 0.5 = half speed)")

    p.add_argument("--repo-id", default="go2-teleop",
                   help="LeRobot dataset repo ID (default: go2-teleop)")
    p.add_argument("--root", default="./data",
                   help="Local dataset root directory (default: ./data)")
    p.add_argument("--task", default="teleoperation",
                   help="Task description for episodes")
    p.add_argument("--num-episodes", type=int, default=0,
                   help="Stop after N episodes (0 = unlimited)")
    p.add_argument("--no-camera", action="store_true", help="Skip camera recording")
    p.add_argument("--no-lidar", action="store_true", help="Skip lidar pose recording")
    p.add_argument("--push-to-hub", action="store_true",
                   help="Push dataset to HF Hub after recording")

    args = p.parse_args()
    args.speed_limit = max(0.0, min(1.0, args.speed_limit))
    return args


def send_and_record_loop(
    conn_wrapper: Go2Connection | None,
    state: ControllerState,
    safety: SafetyFilter,
    streams: RobotStreams | None,
    recorder: EpisodeRecorder | None,
    stop_event: threading.Event,
    args: argparse.Namespace,
):
    """
    20 Hz loop: read controller, apply safety, send to robot, record frame.
    Runs in a dedicated thread.
    """
    sent = 0
    recording = False
    was_f1 = False
    episodes_done = 0

    use_camera = not args.no_camera and not args.dry_run
    use_lidar = not args.no_lidar and not args.dry_run

    while not stop_event.is_set():
        tick_start = time.monotonic()

        # Read and filter controller state
        raw = state.to_dict()
        filtered = safety.apply(raw)

        # F1 toggles recording
        f1_pressed = bool(filtered["keys"] & KEY_F1)
        if f1_pressed and not was_f1:
            if not recording:
                recording = True
                sys.stdout.write("\n  [REC] Recording started\n")
                sys.stdout.flush()
                if safety.rumble:
                    safety.rumble.pulse()
            else:
                recording = False
                sys.stdout.write("\n  [STOP] Recording stopped\n")
                sys.stdout.flush()
                if safety.rumble:
                    safety.rumble.pulse()
                if recorder:
                    recorder.save_episode()
                    episodes_done += 1
                    if args.num_episodes > 0 and episodes_done >= args.num_episodes:
                        sys.stdout.write(f"\n  Reached {args.num_episodes} episodes, stopping.\n")
                        stop_event.set()
                        break
        was_f1 = f1_pressed

        # Send to robot
        if not args.dry_run and conn_wrapper and conn_wrapper.conn:
            try:
                msg = json.dumps({
                    "type": "msg",
                    "topic": "rt/wirelesscontroller",
                    "data": filtered,
                })
                conn_wrapper.run_coroutine(
                    _async_send(conn_wrapper.conn, msg), timeout=1
                )
                sent += 1
            except Exception as e:
                if sent == 0:
                    sys.stdout.write(f"\n  Send failed: {e}\n")

        # Record frame
        if recording and recorder:
            action = np.array(
                [filtered["lx"], filtered["ly"], filtered["rx"], filtered["ry"]],
                dtype=np.float32,
            )
            buttons = np.array([filtered["keys"]], dtype=np.int32)

            if streams:
                obs = streams.snapshot(
                    include_camera=use_camera,
                    include_lidar=use_lidar,
                )
            else:
                obs = {}

            recorder.add_frame(action, buttons, obs, task=args.task)

        # Maintain 20 Hz
        elapsed = time.monotonic() - tick_start
        sleep_time = SEND_RATE - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


async def _async_send(conn, msg: str):
    conn.datachannel.channel.send(msg)


def main():
    args = parse_args()

    print("=" * 60)
    print(f"  go2-record-and-replay  [{args.mode.upper()} mode]"
          + ("  [DRY RUN]" if args.dry_run else ""))
    print("=" * 60)

    # ── Gamepad ──────────────────────────────────────────────
    try:
        import evdev
    except ImportError:
        print("  ERROR: 'evdev' not installed. Run: uv pip install evdev")
        sys.exit(1)

    device = find_gamepad()
    if not device:
        perms = check_device_permissions()
        if perms and not perms["in_input_group"]:
            print(f"  ERROR: Gamepad not detected (permissions issue).")
            print(f"  Fix: sudo usermod -aG input {perms['user']}  then re-login.")
        else:
            print("  ERROR: No gamepad detected. Connect an Xbox controller and try again.")
        sys.exit(1)

    print(f"  Gamepad: {device.name}  ({device.path})")

    rumble = RumbleHelper(device)
    if rumble.available:
        print("  Vibration: enabled")
    else:
        print("  Vibration: unavailable")

    warnings = validate_gamepad(device)
    for w in warnings:
        print(f"  Warning: {w}")

    # ── Connection ───────────────────────────────────────────
    conn_wrapper = None
    streams = None

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
            print("  Make sure the Unitree phone app is closed.")
            sys.exit(1)

        # ── Data streams ─────────────────────────────────────
        streams = RobotStreams(
            enable_camera=not args.no_camera,
            enable_lidar=not args.no_lidar,
        )
        streams.attach(conn_wrapper.conn)
        print("  Data streams attached")

        # Wait briefly for first data
        time.sleep(1.0)

    # ── Recorder ─────────────────────────────────────────────
    recorder = None
    if not args.dry_run:
        recorder = EpisodeRecorder(
            repo_id=args.repo_id,
            root=args.root,
            use_camera=not args.no_camera,
            use_lidar=not args.no_lidar,
        )
        recorder.create(task=args.task)

    # ── Safety ───────────────────────────────────────────────
    safety = SafetyFilter(
        allow_all=args.allow_all,
        speed_limit=args.speed_limit,
        rumble=rumble,
        conn=conn_wrapper.conn if conn_wrapper else None,
        loop=conn_wrapper.loop if conn_wrapper else None,
        dry_run=args.dry_run,
    )

    # ── Print controls ───────────────────────────────────────
    print()
    print("  Controls:")
    print("    Left stick   -> walk / strafe")
    print("    Right stick  -> yaw / look")
    print("    Start        -> walking mode")
    print("    Select       -> standing mode")
    print("    F1 (L-click) -> toggle recording")
    print("    Ctrl+C       -> stop and save")
    if args.speed_limit < 1.0:
        print(f"  Speed limit: {args.speed_limit:.0%}")
    if not args.dry_run:
        print(f"  Dataset: {args.repo_id}")
        print(f"  Task: {args.task}")
    print()

    # ── Run ──────────────────────────────────────────────────
    controller_state = ControllerState()
    stop_event = threading.Event()

    send_thread = threading.Thread(
        target=send_and_record_loop,
        args=(conn_wrapper, controller_state, safety, streams, recorder, stop_event, args),
        daemon=True,
    )
    send_thread.start()

    try:
        gamepad_loop(device, controller_state, stop_event)
    except KeyboardInterrupt:
        stop_event.set()

    # ── Shutdown ─────────────────────────────────────────────
    print()
    print("  Shutting down ...")

    send_thread.join(timeout=5)
    rumble.cleanup()

    if recorder:
        recorder.finalize()

    if conn_wrapper:
        conn_wrapper.disconnect()

    if streams:
        print("  Data received:")
        streams.print_status()

    if recorder and args.push_to_hub:
        print("  Pushing to Hub ...")
        recorder.push_to_hub()

    print("  Done.")


if __name__ == "__main__":
    main()
