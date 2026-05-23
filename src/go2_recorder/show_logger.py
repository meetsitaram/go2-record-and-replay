"""
Show logger: capture per-robot telemetry to a JSONL file during a choreo run.

Goal: when something looks wrong on the floor ("the robot tripped, the next
algo move did nothing"), have a file we can grep/replay later instead of
asking the operator to re-describe the moment by hand.

Design notes
- One JSONL line per (robot, sample). Cheap to append, trivial to parse.
- Rate is configurable; 5Hz is the default (200ms between samples) which is
  enough to catch a trip / mode-loss / foot-off-ground without producing
  large files. ~10MB for an hour of show time per robot.
- Step boundaries are emitted as their own `event` lines so the timeline can
  be reconstructed without reading the whole choreo config.
- All numeric arrays are flattened to plain Python lists so the file remains
  human-readable jsonl (no numpy artifacts, no base64).
- `RobotStreams.snapshot()` clears its `_fresh` flag; the logger keeps a
  shadow buffer via `peek()`-style reads that DON'T clear freshness, so the
  recording loop's normal use of snapshot() (in record.py) is unaffected if
  we ever share the streams instance with one.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import IO, Any

import numpy as np


# ── Heuristics for "robot is in trouble" annotations ──────────────────────
# Beyond these we set tilt_warning=True so a downstream parser can grep for
# trips without reasoning about quaternions itself.
_TILT_WARN_DEG = 25.0
_FOOT_OFF_NEWTONS = 5.0   # foot_force readings below this count as "in air"


def _quat_to_rpy_deg(quat: np.ndarray) -> tuple[float, float, float]:
    """Convert quaternion [w, x, y, z] to (roll, pitch, yaw) in degrees.

    Matches the convention used by RobotStreams._on_robot_pose. Returns zeros
    if the quaternion is degenerate (e.g. all zeros, which happens before any
    sportmode message has arrived).
    """
    if quat is None or len(quat) < 4:
        return 0.0, 0.0, 0.0
    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    norm2 = w * w + x * x + y * y + z * z
    if norm2 < 1e-9:
        return 0.0, 0.0, 0.0
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


@dataclass
class _RobotEntry:
    """Bookkeeping for one robot we are logging."""
    name: str
    player: Any            # RobotPlayer; we only read .name/.connected/.send_errors/.manual_override
    streams: Any           # RobotStreams instance attached to this robot
    last_send_errors: int = 0


@dataclass
class ShowLogger:
    """Append-only JSONL logger for show telemetry.

    Lifecycle:
        logger = ShowLogger.create(path)
        logger.attach(player, streams)   # one call per robot
        logger.start(t_show_start)       # call when the show clock begins
        logger.note_step(idx, total, step)   # call at each step boundary
        logger.note_event("ctrl_c", ...) # ad hoc events
        await logger.aclose()            # at end of show
    """

    path: str
    rate_hz: float = 5.0
    _fp: IO[str] | None = None
    _robots: list[_RobotEntry] = field(default_factory=list)
    _task: asyncio.Task | None = None
    _stop: asyncio.Event = field(default_factory=asyncio.Event)
    _t_show_start: float | None = None  # time.monotonic() reference, set by start()

    @classmethod
    def create(cls, path: str | None = None, *,
               rate_hz: float = 5.0,
               base_dir: str = "data/showlogs") -> "ShowLogger":
        """Open a new JSONL log. If `path` is None, auto-generate a timestamped name."""
        if path is None:
            os.makedirs(base_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            path = os.path.join(base_dir, f"show_{stamp}.jsonl")
        else:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fp = open(path, "w", buffering=1)  # line-buffered so a Ctrl+C still flushes
        logger = cls(path=path, rate_hz=rate_hz, _fp=fp)
        logger._write({
            "event": "log_open",
            "wall_time": datetime.now().isoformat(timespec="seconds"),
            "rate_hz": rate_hz,
        })
        return logger

    def attach(self, player, streams) -> None:
        """Register a (player, streams) pair to be sampled.

        Idempotent on player.name -- a second attach for the same name
        replaces the prior one (useful on reconnect, though not currently
        exercised by run_show).
        """
        for i, e in enumerate(self._robots):
            if e.name == player.name:
                self._robots[i] = _RobotEntry(name=player.name, player=player, streams=streams)
                return
        self._robots.append(_RobotEntry(name=player.name, player=player, streams=streams))
        self._write({"event": "robot_attached",
                     "robot": player.name,
                     "ip": getattr(player, "ip", None)})

    def start(self, t_show_start: float | None = None) -> None:
        """Begin the periodic sampling task. Safe to call only once."""
        if self._task is not None:
            return
        self._t_show_start = t_show_start if t_show_start is not None else time.monotonic()
        self._write({"event": "show_start", "wall_time": datetime.now().isoformat(timespec="seconds")})
        self._task = asyncio.get_event_loop().create_task(self._run())

    def note_step(self, step_idx: int, total: int, step) -> None:
        """Emit a step_start event line. step is RecordingStep|AlgoStep|ActionStep."""
        kind = getattr(step, "kind", type(step).__name__)
        # `label` exists on all three step classes but isn't required.
        label = getattr(step, "label", None)
        meta: dict[str, Any] = {
            "event": "step_start",
            "step_idx": step_idx,
            "step_total": total,
            "kind": kind,
            "label": label,
            "t_show": self._t_show(),
        }
        # Surface a few useful per-kind fields (cheap, makes log self-explanatory).
        for attr in ("name", "duration", "tempo", "amplitude", "dataset", "episode"):
            if hasattr(step, attr):
                v = getattr(step, attr)
                if isinstance(v, (int, float, str, bool)) or v is None:
                    meta[attr] = v
        self._write(meta)

    def note_event(self, name: str, **fields) -> None:
        """Generic event line (e.g. 'ctrl_c', 'recovery_stand_sent')."""
        msg = {"event": name, "t_show": self._t_show(), **fields}
        self._write(msg)

    async def aclose(self) -> None:
        if self._task is not None:
            self._stop.set()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._fp is not None:
            self._write({"event": "log_close",
                         "wall_time": datetime.now().isoformat(timespec="seconds")})
            try:
                self._fp.flush()
                self._fp.close()
            except Exception:
                pass
            self._fp = None

    # ── internals ────────────────────────────────────────────────────────

    async def _run(self) -> None:
        period = 1.0 / max(0.1, self.rate_hz)
        next_t = time.monotonic()
        while not self._stop.is_set():
            for entry in self._robots:
                try:
                    self._sample(entry)
                except Exception as exc:
                    self._write({"event": "sample_error",
                                 "robot": entry.name,
                                 "err": f"{type(exc).__name__}: {exc}"})
            next_t += period
            now = time.monotonic()
            sleep_for = next_t - now
            if sleep_for > 0:
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=sleep_for)
                except asyncio.TimeoutError:
                    pass
            else:
                # We're behind schedule. Resync and continue without spinning.
                next_t = now

    def _sample(self, entry: _RobotEntry) -> None:
        s = entry.streams
        player = entry.player
        # Use peek() so we don't drain freshness from any other consumer.
        sport = s.sport_state.peek() if hasattr(s, "sport_state") else None
        power = s.power.peek() if hasattr(s, "power") else None
        jtemp = s.motor_temperatures.peek() if hasattr(s, "motor_temperatures") else None
        jtau = s.joint_torques.peek() if hasattr(s, "joint_torques") else None

        sample: dict[str, Any] = {
            "t_show": self._t_show(),
            "robot": entry.name,
            "connected": bool(getattr(player, "connected", False))
                         and bool(getattr(player, "is_alive", lambda: False)()),
            "manual_override": bool(getattr(player, "manual_override", False)),
        }

        # Derived send-error delta is more useful than the absolute counter.
        send_errs = int(getattr(player, "send_errors", 0))
        sample["send_errors_delta"] = send_errs - entry.last_send_errors
        sample["send_errors_total"] = send_errs
        entry.last_send_errors = send_errs

        # Stream ages (seconds since last update). Useful to spot a topic
        # going dark while the WebRTC link is technically still alive.
        sample["sport_age"] = round(getattr(getattr(s, "sport_state", None), "age", float("inf")), 3)
        sample["lowstate_age"] = round(getattr(getattr(s, "joint_positions", None), "age", float("inf")), 3)

        if sport is not None and len(sport) >= 16:
            pos = [round(float(sport[0]), 4), round(float(sport[1]), 4), round(float(sport[2]), 4)]
            vel = [round(float(sport[3]), 4), round(float(sport[4]), 4), round(float(sport[5]), 4)]
            quat = sport[7:11]
            roll, pitch, yaw = _quat_to_rpy_deg(quat)
            ff = [round(float(sport[11+i]), 1) for i in range(4)]
            feet_off = sum(1 for f in ff if f < _FOOT_OFF_NEWTONS)
            tilt_warn = abs(roll) > _TILT_WARN_DEG or abs(pitch) > _TILT_WARN_DEG
            sample.update({
                "pos": pos, "vel": vel,
                "yaw_speed": round(float(sport[6]), 4),
                "rpy_deg": [round(roll, 1), round(pitch, 1), round(yaw, 1)],
                "foot_force": ff,
                "feet_off_ground": feet_off,
                "tilt_warning": tilt_warn,
                "battery_pct": round(float(sport[15]), 1),
            })

        if power is not None and len(power) >= 4:
            sample["power"] = {
                "v": round(float(power[0]), 2),
                "a": round(float(power[1]), 2),
                "soc": round(float(power[2]), 1),
                "i": round(float(power[3]), 2),
            }

        if jtemp is not None and len(jtemp) > 0:
            sample["motor_temp_max"] = round(float(np.max(jtemp)), 1)
        if jtau is not None and len(jtau) > 0:
            sample["motor_tau_absmax"] = round(float(np.max(np.abs(jtau))), 2)

        self._write(sample)

    def _t_show(self) -> float | None:
        if self._t_show_start is None:
            return None
        return round(time.monotonic() - self._t_show_start, 3)

    def _write(self, obj: dict[str, Any]) -> None:
        if self._fp is None:
            return
        try:
            self._fp.write(json.dumps(obj, separators=(",", ":")) + "\n")
        except Exception:
            # Logger must NEVER crash the show.
            pass
