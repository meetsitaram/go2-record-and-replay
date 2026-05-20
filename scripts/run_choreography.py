#!/usr/bin/env python3
"""
Run a choreography on the robot, synced with audio playback.

Reads config/choreography.yaml and executes moves in sequence,
playing the song simultaneously. Position corrections after
displacement-heavy moves use closed-loop feedback from sportmodestate.

Usage:
    .venv/bin/python scripts/run_choreography.py 192.168.1.246 --aes-key KEY
    .venv/bin/python scripts/run_choreography.py 192.168.1.246 --aes-key KEY --dry-run
"""

import argparse
import asyncio
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml

from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD

CHOREOGRAPHY_PATH = Path("config/choreography.yaml")


class RobotState:
    """Tracks live robot state from sportmodestate topic."""

    def __init__(self):
        self.position = [0.0, 0.0, 0.0]
        self.yaw = 0.0
        self.body_height = 0.0
        self.mode = 0
        self.velocity = [0.0, 0.0, 0.0]
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
            self.velocity = d.get("velocity", self.velocity)
            self.last_update = time.monotonic()
        except Exception:
            pass


async def closed_loop_correction(conn, state, target_pos, target_yaw, config):
    """
    Drive the robot back to target_pos/target_yaw using live feedback.
    
    Reads current position from state (updated by sportmodestate subscriber),
    computes error, and sends Move commands until within tolerance or timeout.
    """
    speed = config.get("speed", 0.2)
    yaw_speed = config.get("yaw_speed", 0.4)
    pos_tol = config.get("position_tolerance", 0.05)
    yaw_tol = config.get("yaw_tolerance", 0.05)
    max_dur = config.get("max_duration", 3.0)

    t0 = time.monotonic()

    while time.monotonic() - t0 < max_dur:
        # Compute position error in world frame
        dx = target_pos[0] - state.position[0]
        dy = target_pos[1] - state.position[1]
        dist = math.sqrt(dx**2 + dy**2)

        # Yaw error (shortest path)
        dyaw = target_yaw - state.yaw
        while dyaw > math.pi:
            dyaw -= 2 * math.pi
        while dyaw < -math.pi:
            dyaw += 2 * math.pi

        # Check if we're close enough
        if dist < pos_tol and abs(dyaw) < yaw_tol:
            break

        # Transform world-frame error into robot body frame
        cos_y = math.cos(state.yaw)
        sin_y = math.sin(state.yaw)
        # Body-frame velocity to reach target
        vx_body = cos_y * dx + sin_y * dy
        vy_body = -sin_y * dx + cos_y * dy

        # Clamp velocities
        v_mag = math.sqrt(vx_body**2 + vy_body**2)
        if v_mag > speed:
            scale = speed / v_mag
            vx_body *= scale
            vy_body *= scale

        vyaw = max(-yaw_speed, min(yaw_speed, dyaw * 2.0))

        # Send Move command
        await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"],
            {"api_id": SPORT_CMD["Move"],
             "parameter": json.dumps({"x": round(vx_body, 3),
                                       "y": round(vy_body, 3),
                                       "z": round(vyaw, 3)})}
        )
        await asyncio.sleep(0.1)

    # Stop movement
    await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["StopMove"]}
    )

    # Report final error
    final_dist = math.sqrt(
        (target_pos[0] - state.position[0])**2 +
        (target_pos[1] - state.position[1])**2
    )
    elapsed = time.monotonic() - t0
    return {
        "elapsed": round(elapsed, 2),
        "final_distance_error": round(final_dist, 4),
        "final_yaw_error": round(abs(target_yaw - state.yaw), 4),
        "converged": final_dist < pos_tol and abs(target_yaw - state.yaw) < yaw_tol,
    }


async def keepalive(conn, stop_event):
    """Keep WebRTC alive by subscribing periodically."""
    while not stop_event.is_set():
        try:
            conn.datachannel.pub_sub.publish_without_callback(
                topic="rt/lf/sportmodestate", msg_type="sub"
            )
        except Exception:
            pass
        await asyncio.sleep(2)


async def run_choreography(conn, choreography, dry_run=False):
    """Execute the choreography timeline."""
    moves = choreography["moves"]
    song_file = choreography["song"]["file"]

    # Set up state tracking
    state = RobotState()
    conn.datachannel.pub_sub.subscribe("rt/lf/sportmodestate", state.on_message)
    await asyncio.sleep(1)  # let state populate

    # Stand up
    if not dry_run:
        await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["StandUp"]}
        )
    await asyncio.sleep(2)

    print(f"\nInitial position: x={state.position[0]:.3f}, y={state.position[1]:.3f}, yaw={state.yaw:.3f}")
    print(f"Starting choreography ({len(moves)} entries)...\n")

    # Start audio playback
    audio_proc = None
    if not dry_run and Path(song_file).exists():
        audio_proc = subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", str(song_file)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    t_start = time.monotonic()
    saved_positions = {}  # move_index -> (position, yaw) before move

    for i, entry in enumerate(moves):
        entry_start = entry["start_time"]
        entry_type = entry.get("type", "builtin_move")

        # Wait until it's time
        elapsed = time.monotonic() - t_start
        wait = entry_start - elapsed
        if wait > 0:
            if wait > 2:
                print(f"  [{elapsed:>6.1f}s] Waiting {wait:.1f}s...")
            await asyncio.sleep(wait)

        elapsed = time.monotonic() - t_start

        if entry_type == "builtin_move":
            move_name = entry["move"]
            api_id = entry["api_id"]

            # Save position before move for later correction
            saved_positions[i] = (list(state.position), state.yaw)

            print(f"  [{elapsed:>6.1f}s] MOVE: {move_name} (api_id={api_id})")
            if not dry_run:
                await conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["SPORT_MOD"], {"api_id": api_id}
                )

        elif entry_type == "correction":
            reverses = entry.get("reverses", "")
            # Find the most recent move entry that this correction reverses
            target_pos = None
            target_yaw = None
            for j in range(i - 1, -1, -1):
                if moves[j].get("type") == "builtin_move" and moves[j].get("move") == reverses:
                    if j in saved_positions:
                        target_pos, target_yaw = saved_positions[j]
                        break

            if target_pos is None:
                print(f"  [{elapsed:>6.1f}s] CORRECT: <-{reverses} (no saved position, skipping)")
                continue

            print(f"  [{elapsed:>6.1f}s] CORRECT: <-{reverses} (target: x={target_pos[0]:.3f}, y={target_pos[1]:.3f}, yaw={target_yaw:.3f})")

            if not dry_run:
                result = await closed_loop_correction(conn, state, target_pos, target_yaw, entry)
                status = "OK" if result["converged"] else f"err={result['final_distance_error']:.3f}m"
                print(f"           -> {status} in {result['elapsed']:.1f}s")
            else:
                print(f"           -> (dry run, would correct for ~{entry['duration']:.1f}s)")

    # Done
    elapsed = time.monotonic() - t_start
    print(f"\n  Choreography complete in {elapsed:.1f}s")

    if audio_proc:
        audio_proc.terminate()

    # Final position report
    print(f"  Final position: x={state.position[0]:.3f}, y={state.position[1]:.3f}, yaw={state.yaw:.3f}")


async def main():
    parser = argparse.ArgumentParser(description="Run choreography on robot")
    parser.add_argument("ip", help="Robot IP address")
    parser.add_argument("--aes-key", default=None, help="AES-128 key")
    parser.add_argument("--dry-run", action="store_true", help="Print timeline without sending commands")
    args = parser.parse_args()

    # Load choreography
    with open(CHOREOGRAPHY_PATH) as f:
        choreography = yaml.safe_load(f)
    print(f"Loaded choreography: {len(choreography['moves'])} entries, song={choreography['song']['file']}")

    if args.dry_run:
        print("DRY RUN MODE - no robot commands will be sent\n")
        # Still connect to read state
        kwargs = {"ip": args.ip}
        if args.aes_key:
            kwargs["aes_128_key"] = args.aes_key
        conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, **kwargs)
        await asyncio.wait_for(conn.connect(), timeout=15)
        stop_ka = asyncio.Event()
        ka_task = asyncio.create_task(keepalive(conn, stop_ka))
        await run_choreography(conn, choreography, dry_run=True)
        stop_ka.set()
        ka_task.cancel()
        await conn.disconnect()
    else:
        kwargs = {"ip": args.ip}
        if args.aes_key:
            kwargs["aes_128_key"] = args.aes_key
        conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, **kwargs)
        await asyncio.wait_for(conn.connect(), timeout=15)
        print("Connected!")
        stop_ka = asyncio.Event()
        ka_task = asyncio.create_task(keepalive(conn, stop_ka))
        await run_choreography(conn, choreography, dry_run=False)
        stop_ka.set()
        ka_task.cancel()
        await conn.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
