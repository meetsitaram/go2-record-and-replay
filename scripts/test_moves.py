#!/usr/bin/env python3
"""
Test all built-in dance/trick/gesture commands on the Go2.

Sends each command one at a time, monitors sportmodestate to detect when the
move completes (body returns to stable standing), and logs the actual duration.
Press Enter to send the next command, or 'q' to quit.
"""

import asyncio
import json
import sys
import time

from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD

MOVES_TO_TEST = [
    # (name, api_id, category, max_wait_seconds)
    ("Hello", 1016, "gesture", 10),
    ("Stretch", 1017, "gesture", 10),
    ("Dance1", 1022, "dance", 30),
    ("Dance2", 1023, "dance", 30),
    ("Pose", 1028, "gesture", 10),
    ("Scrape", 1029, "playful", 10),
    ("FrontJump", 1031, "trick", 10),
    ("FrontPounce", 1032, "trick", 10),
    ("FingerHeart", 1036, "gesture", 10),
    # Previously rejected on Air — retry on Pro
    ("Wallow", 1021, "playful", 15),
    ("WiggleHips", 1033, "gesture", 10),
    ("StandOut", 1039, "gesture", 10),
    ("CrossWalk", 1051, "step", 15),
    ("Handstand", 1301, "trick", 15),
    ("CrossStep", 1302, "step", 15),
    ("OnesidedStep", 1303, "step", 15),
    ("Bound", 1304, "trick", 10),
    ("MoonWalk", 1305, "dance", 15),
    # MCF API IDs for rejected moves (firmware >= 1.1.6)
    ("CrossStep_MCF", 2051, "step-mcf", 15),
    ("HandStand_MCF", 2044, "trick-mcf", 15),
    ("FreeBound_MCF", 2046, "trick-mcf", 10),
    ("FreeJump_MCF", 2047, "trick-mcf", 10),
    ("FreeWalk_MCF", 2045, "step-mcf", 15),
    ("ClassicWalk_MCF", 2049, "step-mcf", 15),
    ("WalkUpright_MCF", 2050, "step-mcf", 15),
]

# Dangerous moves - test separately with extra caution
DANGEROUS_MOVES = [
    ("FrontFlip", 1030, "acrobatic", 8),
    ("LeftFlip", 1042, "acrobatic", 8),
    ("RightFlip", 1043, "acrobatic", 8),
    ("BackFlip", 1044, "acrobatic", 8),
]

STANDING_HEIGHT = 0.28
STABLE_READINGS = 10  # consecutive stable readings (~0.5s at 20Hz) to confirm move ended


class StateMonitor:
    """Monitors sportmodestate to detect when a move starts and finishes."""

    def __init__(self):
        self.body_height = 0.0
        self.mode = 0
        self.gait_type = 0
        self.velocity = [0.0, 0.0, 0.0]
        self.last_update = 0.0
        self._stable_count = 0
        self._move_active = False
        self._move_started_at = 0.0
        self._move_ended_at = 0.0
        self._done_event = asyncio.Event()
        self._height_log = []
        self._left_standing = False
        self._min_duration = 2.0  # don't declare done before this many seconds

    def on_msg(self, msg):
        try:
            data = msg if isinstance(msg, dict) else json.loads(msg)
            d = data.get("data", data)
            if isinstance(d, str):
                d = json.loads(d)

            self.body_height = float(d.get("body_height", 0.0))
            self.mode = d.get("mode", 0)
            self.gait_type = d.get("gait_type", d.get("gaitType", 0))
            vel = d.get("velocity", [0, 0, 0])
            if isinstance(vel, list):
                self.velocity = vel
            self.last_update = time.monotonic()

            self._height_log.append((time.monotonic(), self.body_height, self.mode))

            if self._move_active:
                elapsed = time.monotonic() - self._move_started_at

                # Detect that the robot has left its resting state
                not_standing = (
                    self.body_height < STANDING_HEIGHT - 0.02
                    or self.mode != 0
                    or abs(self.velocity[0]) > 0.1
                    or abs(self.velocity[1]) > 0.1
                )
                if not_standing:
                    self._left_standing = True

                is_stable = (
                    self.body_height >= STANDING_HEIGHT
                    and self.mode == 0
                    and abs(self.velocity[0]) < 0.03
                    and abs(self.velocity[1]) < 0.03
                )

                # Only declare done if: left standing at some point, min time passed, stable
                if is_stable and self._left_standing and elapsed > self._min_duration:
                    self._stable_count += 1
                    if self._stable_count >= STABLE_READINGS:
                        self._move_ended_at = time.monotonic()
                        self._move_active = False
                        self._done_event.set()
                else:
                    self._stable_count = 0
        except Exception:
            pass

    def start_tracking(self):
        self._move_active = True
        self._move_started_at = time.monotonic()
        self._move_ended_at = 0.0
        self._stable_count = 0
        self._left_standing = False
        self._done_event.clear()
        self._height_log = []

    async def wait_for_completion(self, timeout: float) -> float | None:
        """Wait for the move to finish. Returns duration in seconds, or None on timeout."""
        try:
            await asyncio.wait_for(self._done_event.wait(), timeout=timeout)
            return self._move_ended_at - self._move_started_at
        except asyncio.TimeoutError:
            return None

    def get_height_range(self) -> tuple[float, float]:
        """Get min/max body height during the move."""
        if not self._height_log:
            return (0.0, 0.0)
        heights = [h for _, h, _ in self._height_log]
        return (min(heights), max(heights))

    def get_modes_seen(self) -> set:
        """Get all mode values observed during the move."""
        if not self._height_log:
            return set()
        return {m for _, _, m in self._height_log}


async def async_input(prompt: str) -> str:
    """Non-blocking input that keeps the event loop alive."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, prompt)


async def keepalive(conn, stop_event: asyncio.Event):
    """Send periodic heartbeat to prevent WebRTC connection timeout."""
    while not stop_event.is_set():
        try:
            if conn.pc and conn.pc.connectionState == "connected":
                conn.datachannel.pub_sub.publish_without_callback(
                    topic="rt/lf/sportmodestate",
                    msg_type="sub"
                )
        except Exception:
            pass
        await asyncio.sleep(2)


async def main():
    ip = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.133"
    aes_key = sys.argv[2] if len(sys.argv) > 2 else None
    # Optional: --from <name> to skip to a specific move
    start_from = None
    for i, arg in enumerate(sys.argv):
        if arg == "--from" and i + 1 < len(sys.argv):
            start_from = sys.argv[i + 1]

    kwargs = {"ip": ip}
    if aes_key:
        kwargs["aes_128_key"] = aes_key

    print(f"Connecting to {ip}...")
    conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, **kwargs)
    await asyncio.wait_for(conn.connect(), timeout=15)
    print("Connected!\n")

    # Subscribe to state
    monitor = StateMonitor()
    conn.datachannel.pub_sub.subscribe("rt/lf/sportmodestate", monitor.on_msg)
    await asyncio.sleep(1)

    # Start keepalive to prevent connection timeout
    stop_keepalive = asyncio.Event()
    keepalive_task = asyncio.create_task(keepalive(conn, stop_keepalive))

    # First ensure standing
    print("Ensuring robot is standing...")
    await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["StandUp"]}
    )
    await asyncio.sleep(3)
    print(f"Ready. Body height: {monitor.body_height:.3f}m, mode: {monitor.mode}\n")

    print("=" * 70)
    print("  BUILT-IN MOVE TEST (with duration measurement)")
    print("  Press ENTER to send next move, 's' to skip, 'q' to quit")
    print("=" * 70)

    results = []
    skipping = start_from is not None

    for name, api_id, category, max_wait in MOVES_TO_TEST:
        if skipping:
            if name == start_from:
                skipping = False
            else:
                continue

        print(f"\n  [{category.upper():10s}] {name} (api_id={api_id})")
        user_input = (await async_input("  > ")).strip().lower()

        if user_input == "q":
            break
        if user_input == "s":
            results.append((name, api_id, category, "SKIPPED", 0, "", ""))
            continue

        try:
            monitor.start_tracking()

            resp = await conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"], {"api_id": api_id}
            )
            status_code = resp.get("data", {}).get("header", {}).get("status", {}).get("code", -1)

            if status_code != 0:
                print(f"    REJECTED (code={status_code})")
                results.append((name, api_id, category, f"REJECTED({status_code})", 0, "", ""))
                await asyncio.sleep(1)
                continue

            print(f"    Sent OK. Waiting for move to complete (max {max_wait}s)...")
            t0 = time.monotonic()

            # Show live status while waiting
            while not monitor._done_event.is_set():
                elapsed = time.monotonic() - t0
                if elapsed >= max_wait:
                    break
                sys.stdout.write(
                    f"\r    [{elapsed:5.1f}s] height={monitor.body_height:.3f}m "
                    f"mode={monitor.mode} vel=({monitor.velocity[0]:+.2f},{monitor.velocity[1]:+.2f})"
                    f" left_stand={'Y' if monitor._left_standing else 'N'}\033[K"
                )
                sys.stdout.flush()
                await asyncio.sleep(0.25)

            duration = None
            if monitor._done_event.is_set():
                duration = monitor._move_ended_at - monitor._move_started_at
            h_min, h_max = monitor.get_height_range()
            modes = monitor.get_modes_seen()

            if duration is not None:
                print(f"\n    DONE in {duration:.1f}s  "
                      f"(height: {h_min:.3f}-{h_max:.3f}m, modes: {modes})")
                results.append((name, api_id, category, "OK",
                                round(duration, 2),
                                f"{h_min:.3f}-{h_max:.3f}",
                                str(modes)))
            else:
                elapsed = time.monotonic() - t0
                print(f"\n    TIMEOUT after {elapsed:.1f}s  "
                      f"(height: {h_min:.3f}-{h_max:.3f}m, modes: {modes})")
                results.append((name, api_id, category, "TIMEOUT",
                                round(elapsed, 2),
                                f"{h_min:.3f}-{h_max:.3f}",
                                str(modes)))

            # Stand back up before next move
            await conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["StandUp"]}
            )
            await asyncio.sleep(3)

        except Exception as e:
            print(f"    ERROR: {e}")
            results.append((name, api_id, category, f"ERROR", 0, "", str(e)))

    stop_keepalive.set()
    keepalive_task.cancel()
    await conn.disconnect()

    # Summary
    print("\n\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Command':<16s} {'Category':<10s} {'Status':<12s} {'Duration':>8s}  {'Height range'}")
    print(f"  {'-'*15:<16s} {'-'*9:<10s} {'-'*11:<12s} {'-'*8:>8s}  {'-'*15}")
    for name, api_id, category, status, dur, heights, modes in results:
        dur_str = f"{dur:.1f}s" if dur > 0 else ""
        print(f"  {name:<16s} {category:<10s} {status:<12s} {dur_str:>8s}  {heights}")

    # Save results to file
    import yaml
    output = {
        "robot_ip": ip,
        "test_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "moves": []
    }
    for name, api_id, category, status, dur, heights, modes in results:
        output["moves"].append({
            "name": name,
            "api_id": api_id,
            "category": category,
            "status": status,
            "duration_seconds": dur if dur > 0 else None,
            "height_range": heights or None,
            "modes_observed": modes or None,
        })

    out_path = "config/move_test_results.yaml"
    with open(out_path, "w") as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
