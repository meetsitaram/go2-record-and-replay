#!/usr/bin/env python3
"""
Summarise a show log produced by choreo_multi.py's --show-log.

Reads a .jsonl file and prints:
  - per-step header and elapsed time
  - any tilt warnings (|pitch|>25deg or |roll|>25deg) with timestamp
  - any frames with all 4 feet off the ground (jump / fall)
  - any spikes in send_errors
  - disconnect / reconnect / ctrl_c events
  - end-of-show summary

Usage:
    .venv/bin/python scripts/inspect_show_log.py data/showlogs/show_20260521-160000.jsonl
    .venv/bin/python scripts/inspect_show_log.py --latest
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict


def _resolve_latest(base_dir: str = "data/showlogs") -> str | None:
    files = sorted(glob.glob(os.path.join(base_dir, "show_*.jsonl")))
    return files[-1] if files else None


def _load(path: str):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", help="Path to .jsonl log (omit with --latest)")
    parser.add_argument("--latest", action="store_true", help="Use most recent log under data/showlogs/")
    parser.add_argument("--robot", help="Filter samples to one robot name")
    parser.add_argument("--full", action="store_true", help="Print every interesting line (else summary only)")
    args = parser.parse_args()

    path = args.path
    if args.latest or path is None:
        path = _resolve_latest()
        if path is None:
            print("ERROR: no log files in data/showlogs/")
            return 1
    if not os.path.exists(path):
        print(f"ERROR: not found: {path}")
        return 1
    print(f"Log: {path}\n")

    n_samples = 0
    n_tilt = 0
    n_disconnect_samples = 0
    n_send_err_spikes = 0
    n_feet_in_air = 0
    sport_age_max = 0.0
    lowstate_age_max = 0.0
    by_robot_samples = defaultdict(int)
    by_robot_tilt = defaultdict(int)
    cur_step = None
    last_event_t = None

    for obj in _load(path):
        if args.robot and obj.get("robot") not in (None, args.robot):
            continue

        evt = obj.get("event")
        if evt:
            if evt == "step_start":
                cur_step = obj
                print(f"[t={obj.get('t_show'):>6}] step {obj['step_idx']}/{obj['step_total']}  "
                      f"{obj.get('kind','?'):<10} {obj.get('label') or obj.get('name','')}")
            elif evt in ("show_complete", "show_aborted", "ctrl_c_or_exit",
                         "robot_attached", "log_open", "log_close"):
                t = obj.get("t_show", obj.get("wall_time", ""))
                extras = {k: v for k, v in obj.items()
                          if k not in ("event", "t_show", "wall_time")}
                print(f"[t={t!r:>20}] {evt:<20} {extras if extras else ''}")
            elif args.full:
                print(f"  event {evt} {obj}")
            continue

        n_samples += 1
        robot = obj.get("robot", "?")
        by_robot_samples[robot] += 1

        # Track liveness / staleness
        if not obj.get("connected", True):
            n_disconnect_samples += 1
            if args.full:
                print(f"  [t={obj.get('t_show')}] {robot}: DISCONNECTED")
        if obj.get("send_errors_delta", 0) > 0:
            n_send_err_spikes += 1
            print(f"  [t={obj.get('t_show')}] {robot}: send_errors +{obj['send_errors_delta']} "
                  f"(total {obj.get('send_errors_total')})")
        sa = obj.get("sport_age", 0.0)
        la = obj.get("lowstate_age", 0.0)
        if isinstance(sa, (int, float)) and sa != float("inf"):
            sport_age_max = max(sport_age_max, sa)
        if isinstance(la, (int, float)) and la != float("inf"):
            lowstate_age_max = max(lowstate_age_max, la)

        if obj.get("tilt_warning"):
            n_tilt += 1
            by_robot_tilt[robot] += 1
            rpy = obj.get("rpy_deg", [None, None, None])
            ff = obj.get("foot_force")
            if args.full or n_tilt <= 20:
                step_label = (cur_step.get("label") or cur_step.get("name", ""))             if cur_step else "(pre-show)"
                step_kind = cur_step.get("kind", "?") if cur_step else "?"
                print(f"  [t={obj.get('t_show')}] {robot}: TILT roll={rpy[0]} pitch={rpy[1]} yaw={rpy[2]} "
                      f"feet_off={obj.get('feet_off_ground')} ff={ff} "
                      f"during step {cur_step.get('step_idx') if cur_step else '?'} "
                      f"({step_kind} {step_label})")
        elif obj.get("feet_off_ground", 0) >= 4:
            n_feet_in_air += 1
            if args.full or n_feet_in_air <= 5:
                print(f"  [t={obj.get('t_show')}] {robot}: ALL FEET IN AIR (jump/fall)")

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  total samples           : {n_samples}")
    for r, n in by_robot_samples.items():
        print(f"    {r}: {n} samples ({by_robot_tilt[r]} tilt warnings)")
    print(f"  tilt warnings (>25deg)  : {n_tilt}")
    print(f"  all-feet-in-air samples : {n_feet_in_air}")
    print(f"  samples with no link    : {n_disconnect_samples}")
    print(f"  send_error spike samples: {n_send_err_spikes}")
    print(f"  max sport_age (s)       : {sport_age_max:.2f}")
    print(f"  max lowstate_age (s)    : {lowstate_age_max:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
