#!/usr/bin/env python3
"""
Capture full robot state during each built-in move at ~20Hz.

Records body height, IMU (roll/pitch/yaw), velocity, foot forces, and mode
for each move. Saves both a summary YAML and full frame-by-frame JSON for
later use in beat-synced choreography.

Usage:
    .venv/bin/python scripts/capture_move_profiles.py 192.168.1.133
    .venv/bin/python scripts/capture_move_profiles.py 192.168.1.246 --aes-key 2c09e23856fa423ed680313dd939a3f0
"""

import argparse
import asyncio
import json
import time

import numpy as np
import yaml

from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD

MOVES = [
    ("Hello", 1016, 10),
    ("Stretch", 1017, 10),
    ("Dance1", 1022, 15),
    ("Dance2", 1023, 30),
    ("Pose", 1028, 10),
    ("Scrape", 1029, 12),
    ("FrontJump", 1031, 8),
    ("FrontPounce", 1032, 8),
    ("FingerHeart", 1036, 10),
    ("StandDown", 1005, 6),
]

STANDING_HEIGHT = 0.28
STABLE_NEEDED = 10


async def keepalive(conn, stop_event):
    while not stop_event.is_set():
        try:
            conn.datachannel.pub_sub.publish_without_callback(
                topic="rt/lf/sportmodestate", msg_type="sub"
            )
        except Exception:
            pass
        await asyncio.sleep(2)


async def capture_move(conn, name, api_id, max_wait):
    """Send a move and capture full state until it returns to standing."""
    frames = []
    done = asyncio.Event()
    stable_count = [0]
    started = [False]
    t0 = time.monotonic()

    def on_state(msg):
        try:
            data = msg if isinstance(msg, dict) else json.loads(msg)
            d = data.get("data", data)
            if isinstance(d, str):
                d = json.loads(d)

            t = time.monotonic() - t0
            frame = {
                "time": round(t, 4),
                "body_height": d.get("body_height", 0),
                "mode": d.get("mode", 0),
                "gait_type": d.get("gait_type", 0),
                "velocity": d.get("velocity", [0, 0, 0]),
                "yaw_speed": d.get("yaw_speed", 0),
            }

            pos = d.get("position", [0, 0, 0])
            if isinstance(pos, list) and len(pos) >= 3:
                frame["position"] = pos

            imu = d.get("imu_state", {})
            if isinstance(imu, dict):
                frame["rpy"] = imu.get("rpy", [0, 0, 0])
                frame["quaternion"] = imu.get("quaternion", [1, 0, 0, 0])
                frame["gyroscope"] = imu.get("gyroscope", [0, 0, 0])
                frame["accelerometer"] = imu.get("accelerometer", [0, 0, 0])

            frame["foot_force"] = d.get("foot_force", [0, 0, 0, 0])

            frames.append(frame)

            # Detect completion
            bh = frame["body_height"]
            vel = frame["velocity"]
            not_standing = (
                bh < STANDING_HEIGHT - 0.02
                or frame["mode"] != 0
                or abs(vel[0]) > 0.1
                or abs(vel[1]) > 0.1
            )
            if not_standing:
                started[0] = True
                stable_count[0] = 0
            elif started[0] and bh >= STANDING_HEIGHT and t > 2.0:
                stable_count[0] += 1
                if stable_count[0] >= STABLE_NEEDED:
                    done.set()
        except Exception:
            pass

    conn.datachannel.pub_sub.subscribe("rt/lf/sportmodestate", on_state)
    await asyncio.sleep(0.5)

    resp = await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"], {"api_id": api_id}
    )
    status = resp.get("data", {}).get("header", {}).get("status", {}).get("code", -1)

    if status != 0:
        return name, status, []

    try:
        await asyncio.wait_for(done.wait(), timeout=max_wait)
    except asyncio.TimeoutError:
        pass

    return name, 0, frames


async def main():
    parser = argparse.ArgumentParser(description="Capture robot state during built-in moves")
    parser.add_argument("ip", help="Robot IP address")
    parser.add_argument("--aes-key", default=None, help="AES-128 key for newer firmware")
    args = parser.parse_args()

    kwargs = {"ip": args.ip}
    if args.aes_key:
        kwargs["aes_128_key"] = args.aes_key

    print(f"Connecting to {args.ip}...")
    conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, **kwargs)
    await asyncio.wait_for(conn.connect(), timeout=15)
    print("Connected!\n")

    # Keepalive
    stop_ka = asyncio.Event()
    ka_task = asyncio.create_task(keepalive(conn, stop_ka))

    # Stand up first
    await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["StandUp"]}
    )
    await asyncio.sleep(3)
    print("Robot standing. Starting capture...\n")

    all_results = {}

    for name, api_id, max_wait in MOVES:
        print(f"  Capturing {name} (api_id={api_id})...", end="", flush=True)
        move_name, status, frames = await capture_move(conn, name, api_id, max_wait)

        if status != 0:
            print(f" REJECTED (code={status})")
            continue

        duration = frames[-1]["time"] - frames[0]["time"] if frames else 0
        print(f" OK: {len(frames)} frames, {duration:.1f}s")

        all_results[name] = {
            "api_id": api_id,
            "num_frames": len(frames),
            "duration_seconds": round(duration, 2),
            "frames": frames,
        }

        # Stand back up
        await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["StandUp"]}
        )
        await asyncio.sleep(3)

    stop_ka.set()
    ka_task.cancel()
    await conn.disconnect()

    # Save full frame data
    json_path = "config/move_profiles_full.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f)
    print(f"\n  Full frame data saved to {json_path}")

    # Create summary
    summary = {}
    for name, data in all_results.items():
        frames = data["frames"]
        if not frames:
            continue

        heights = [f["body_height"] for f in frames]
        vels_x = [f["velocity"][0] for f in frames]
        vels_y = [f["velocity"][1] for f in frames]
        rpys = [f.get("rpy", [0, 0, 0]) for f in frames]
        rolls = [r[0] for r in rpys]
        pitches = [r[1] for r in rpys]
        yaws = [r[2] for r in rpys]
        gyros = [f.get("gyroscope", [0, 0, 0]) for f in frames]
        gyro_magnitudes = [np.sqrt(g[0]**2 + g[1]**2 + g[2]**2) for g in gyros]

        summary[name] = {
            "api_id": data["api_id"],
            "duration": data["duration_seconds"],
            "num_frames": data["num_frames"],
            "fps": round(data["num_frames"] / data["duration_seconds"], 1) if data["duration_seconds"] > 0 else 0,
            "body_height": {
                "min": round(min(heights), 4),
                "max": round(max(heights), 4),
                "mean": round(float(np.mean(heights)), 4),
            },
            "velocity_x": {"min": round(min(vels_x), 4), "max": round(max(vels_x), 4)},
            "velocity_y": {"min": round(min(vels_y), 4), "max": round(max(vels_y), 4)},
            "roll": {"min": round(min(rolls), 4), "max": round(max(rolls), 4)},
            "pitch": {"min": round(min(pitches), 4), "max": round(max(pitches), 4)},
            "yaw_range": round(max(yaws) - min(yaws), 4),
            "angular_energy": round(float(np.mean(gyro_magnitudes)), 4),
            "peak_angular_velocity": round(float(max(gyro_magnitudes)), 4),
        }

    yaml_path = "config/move_profiles.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
    print(f"  Summary saved to {yaml_path}")

    # Print summary
    print("\n" + "=" * 75)
    print("  MOVE PROFILES")
    print("=" * 75)
    print(f"  {'Move':<14s} {'Dur':>5s} {'FPS':>4s} {'Height':>14s} "
          f"{'Roll':>12s} {'Pitch':>12s} {'AngEnergy':>9s}")
    print(f"  {'-'*13:<14s} {'-'*4:>5s} {'-'*3:>4s} {'-'*13:>14s} "
          f"{'-'*11:>12s} {'-'*11:>12s} {'-'*8:>9s}")
    for name, s in summary.items():
        h = s["body_height"]
        print(
            f"  {name:<14s} {s['duration']:>4.1f}s {s['fps']:>3.0f} "
            f" {h['min']:.3f}-{h['max']:.3f} "
            f" {s['roll']['min']:+.3f}/{s['roll']['max']:+.3f} "
            f" {s['pitch']['min']:+.3f}/{s['pitch']['max']:+.3f} "
            f" {s['angular_energy']:.3f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
