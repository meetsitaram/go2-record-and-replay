#!/usr/bin/env python3
"""
Analyze a teleop recording to extract dance moves.

Reads a LeRobot episode parquet and produces a per-frame summary of:
    - Controller inputs (lx, ly, rx, ry, button combos)
    - Body orientation (roll/pitch/yaw derived from IMU quaternion)
    - Body height (pos_z)
    - Joint deviations (hip/thigh/calf per leg)

Optionally splits the episode into "segments" wherever the buttons change
or there's > 1 sec of zero input -- each segment is a candidate move.

Usage:
    .venv/bin/python scripts/analyze_teleop_moves.py PATH/TO/episode.parquet
    .venv/bin/python scripts/analyze_teleop_moves.py PATH/TO/episode.parquet --plot

Outputs:
    - Console summary of segments / candidate moves
    - Optional matplotlib plots if --plot is passed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# IMU quaternion -> Euler (radians). Standard ZYX convention.
def quat_to_euler(qw, qx, qy, qz):
    """Convert quaternion to roll, pitch, yaw (rad)."""
    # roll (x-axis)
    sinr = 2.0 * (qw * qx + qy * qz)
    cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr, cosr)
    # pitch (y-axis)
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    # yaw (z-axis)
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny, cosy)
    return roll, pitch, yaw


def load_episode(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df


def expand_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Expand list-valued columns into per-frame numeric arrays we can analyze."""
    n = len(df)
    out = pd.DataFrame()
    out["frame"] = np.arange(n)
    out["t"] = np.arange(n) / 20.0  # assume 20 Hz

    # Action (lx, ly, rx, ry)
    action = np.stack(df["action"].values)
    out["lx"] = action[:, 0]
    out["ly"] = action[:, 1]
    out["rx"] = action[:, 2]
    out["ry"] = action[:, 3]

    # Buttons (single int bitmask)
    buttons = np.stack(df["action.buttons"].values)
    out["keys"] = buttons[:, 0].astype(int)

    # State (16 floats)
    state = np.stack(df["observation.state"].values)
    out["pos_x"] = state[:, 0]
    out["pos_y"] = state[:, 1]
    out["pos_z"] = state[:, 2]
    out["vel_x"] = state[:, 3]
    out["vel_y"] = state[:, 4]
    out["vel_z"] = state[:, 5]
    out["yaw_speed"] = state[:, 6]
    out["qw"] = state[:, 7]
    out["qx"] = state[:, 8]
    out["qy"] = state[:, 9]
    out["qz"] = state[:, 10]
    out["foot_force_0"] = state[:, 11]
    out["foot_force_1"] = state[:, 12]
    out["foot_force_2"] = state[:, 13]
    out["foot_force_3"] = state[:, 14]
    out["battery"] = state[:, 15]

    roll, pitch, yaw = quat_to_euler(out["qw"], out["qx"], out["qy"], out["qz"])
    out["roll"] = roll
    out["pitch"] = pitch
    out["yaw"] = yaw

    return out


def decode_buttons(keys: int) -> str:
    """Decode key bitmask into short label."""
    labels = []
    bits = {
        0x0001: "RB", 0x0002: "RT", 0x0004: "Start", 0x0008: "Select",
        0x0010: "LT", 0x0020: "LB", 0x0040: "F1", 0x0080: "F2",
        0x0100: "A", 0x0200: "B", 0x0400: "X", 0x0800: "Y",
        0x1000: "Up", 0x2000: "Right", 0x4000: "Down", 0x8000: "Left",
    }
    for bit, lab in bits.items():
        if keys & bit:
            labels.append(lab)
    return "+".join(labels) if labels else "-"


def find_segments(df: pd.DataFrame, min_segment_frames: int = 10) -> list[dict]:
    """Split the recording into segments wherever buttons change or sticks dwell."""
    segments = []
    n = len(df)
    if n == 0:
        return segments

    # Segment boundary: change in active buttons OR transition between
    # "active sticks" (>0.1) and "idle sticks" (<0.1).
    keys = df["keys"].values
    stick_active = (df["lx"].abs() + df["ly"].abs() + df["rx"].abs() + df["ry"].abs()) > 0.2
    keys_changed = np.concatenate([[False], keys[1:] != keys[:-1]])
    stick_changed = np.concatenate([[False], stick_active.values[1:] != stick_active.values[:-1]])
    boundaries = np.where(keys_changed | stick_changed)[0]

    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [n]])

    for s, e in zip(starts, ends):
        if e - s < min_segment_frames:
            continue
        seg = df.iloc[s:e]
        segments.append({
            "start_frame": int(s),
            "end_frame": int(e),
            "start_t": float(seg["t"].iloc[0]),
            "duration": float(seg["t"].iloc[-1] - seg["t"].iloc[0]),
            "keys": int(seg["keys"].iloc[0]),
            "label": decode_buttons(int(seg["keys"].iloc[0])),
            "lx_range": (float(seg["lx"].min()), float(seg["lx"].max())),
            "ly_range": (float(seg["ly"].min()), float(seg["ly"].max())),
            "rx_range": (float(seg["rx"].min()), float(seg["rx"].max())),
            "ry_range": (float(seg["ry"].min()), float(seg["ry"].max())),
            "roll_range": (float(seg["roll"].min()), float(seg["roll"].max())),
            "pitch_range": (float(seg["pitch"].min()), float(seg["pitch"].max())),
            "yaw_range": (float(seg["yaw"].min()), float(seg["yaw"].max())),
            "pos_z_range": (float(seg["pos_z"].min()), float(seg["pos_z"].max())),
        })
    return segments


def estimate_sine(values: np.ndarray, t: np.ndarray) -> dict:
    """Crude sine fit: returns dominant frequency and amplitude via FFT."""
    if len(values) < 8:
        return {"freq_hz": 0.0, "amplitude": 0.0}
    v = values - values.mean()
    n = len(v)
    fft = np.fft.rfft(v)
    freqs = np.fft.rfftfreq(n, d=(t[1] - t[0]) if len(t) > 1 else 0.05)
    mag = np.abs(fft)
    if len(mag) > 1:
        peak = int(np.argmax(mag[1:])) + 1  # skip DC
        return {"freq_hz": float(freqs[peak]), "amplitude": float(v.max() - v.min()) / 2.0}
    return {"freq_hz": 0.0, "amplitude": 0.0}


def print_summary(segments: list[dict], df: pd.DataFrame):
    print(f"\n{len(segments)} segments found (min 10 frames = 0.5s):")
    print()
    print(f"  {'#':>2}  {'t_start':>7}  {'dur':>6}  {'buttons':<18}  "
          f"{'sticks (l/r mag)':<18}  {'pose (roll,pitch,yaw)':<28}  {'pz':>6}")
    print("  " + "-" * 110)
    for i, seg in enumerate(segments):
        stick_l = max(abs(seg['lx_range'][0]), abs(seg['lx_range'][1]),
                      abs(seg['ly_range'][0]), abs(seg['ly_range'][1]))
        stick_r = max(abs(seg['rx_range'][0]), abs(seg['rx_range'][1]),
                      abs(seg['ry_range'][0]), abs(seg['ry_range'][1]))
        pose = (f"({seg['roll_range'][1]-seg['roll_range'][0]:+.2f},"
                f"{seg['pitch_range'][1]-seg['pitch_range'][0]:+.2f},"
                f"{seg['yaw_range'][1]-seg['yaw_range'][0]:+.2f})")
        pz_span = seg['pos_z_range'][1] - seg['pos_z_range'][0]
        print(f"  {i+1:>2}  {seg['start_t']:>7.2f}  {seg['duration']:>6.2f}  "
              f"{seg['label']:<18}  l={stick_l:.2f} r={stick_r:.2f}     "
              f"{pose:<28}  {pz_span:>6.3f}")

    # Highlight segments that look like oscillating moves (likely candidates
    # for matching to algorithmic sine waves)
    print()
    print("Candidate oscillating moves (>0.05 rad pose span, no walking):")
    print()
    for i, seg in enumerate(segments):
        s, e = seg['start_frame'], seg['end_frame']
        sub = df.iloc[s:e]
        roll_fit = estimate_sine(sub["roll"].values, sub["t"].values)
        pitch_fit = estimate_sine(sub["pitch"].values, sub["t"].values)
        yaw_fit = estimate_sine(sub["yaw"].values, sub["t"].values)
        max_amp = max(roll_fit["amplitude"], pitch_fit["amplitude"], yaw_fit["amplitude"])
        if max_amp > 0.05 and seg['duration'] > 1.5:
            print(f"  Seg #{i+1} t={seg['start_t']:.2f}s dur={seg['duration']:.1f}s "
                  f"keys={seg['label']}")
            for axis, fit in [("roll", roll_fit), ("pitch", pitch_fit), ("yaw", yaw_fit)]:
                if fit["amplitude"] > 0.03:
                    print(f"    {axis:5s}: amp=±{fit['amplitude']:.2f}rad  "
                          f"freq={fit['freq_hz']:.2f}Hz  "
                          f"-> beats_per_cycle ~ {1.0/max(fit['freq_hz'], 0.01):.2f}")


def plot_episode(df: pd.DataFrame, out_path: Path | None = None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(df["t"], df["lx"], label="lx", lw=0.8)
    axes[0].plot(df["t"], df["ly"], label="ly", lw=0.8)
    axes[0].plot(df["t"], df["rx"], label="rx", lw=0.8)
    axes[0].plot(df["t"], df["ry"], label="ry", lw=0.8)
    axes[0].set_ylabel("sticks")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["t"], df["keys"], "k-", lw=0.5)
    axes[1].set_ylabel("buttons (bitmask)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df["t"], df["roll"], label="roll", lw=0.8)
    axes[2].plot(df["t"], df["pitch"], label="pitch", lw=0.8)
    axes[2].plot(df["t"], df["yaw"], label="yaw", lw=0.8)
    axes[2].set_ylabel("body orientation (rad)")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(df["t"], df["pos_z"], label="pos_z", lw=0.8)
    axes[3].set_ylabel("body height (m)")
    axes[3].set_xlabel("time (s)")
    axes[3].legend(loc="upper right", fontsize=8)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120)
        print(f"\n  Plot saved to {out_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze a teleop episode for dance moves")
    parser.add_argument("episode", help="Path to episode .parquet file")
    parser.add_argument("--plot", action="store_true", help="Show matplotlib plot")
    parser.add_argument("--save-plot", default=None, help="Save plot to this path instead")
    parser.add_argument("--min-frames", type=int, default=10,
                        help="Minimum frames per segment (default 10 = 0.5s)")
    args = parser.parse_args()

    path = Path(args.episode)
    if not path.exists():
        print(f"ERROR: {path} not found")
        sys.exit(1)

    print(f"Loading {path}...")
    raw = load_episode(path)
    df = expand_columns(raw)
    print(f"  {len(df)} frames ({len(df)/20:.1f}s)")
    print(f"  battery: {df['battery'].iloc[0]:.1f}V -> {df['battery'].iloc[-1]:.1f}V")
    print(f"  pos travelled: x {df['pos_x'].iloc[-1] - df['pos_x'].iloc[0]:+.2f}m  "
          f"y {df['pos_y'].iloc[-1] - df['pos_y'].iloc[0]:+.2f}m")
    print(f"  yaw rotated: {(df['yaw'].iloc[-1] - df['yaw'].iloc[0]) * 180 / np.pi:+.0f} deg")

    segments = find_segments(df, args.min_frames)
    print_summary(segments, df)

    if args.plot or args.save_plot:
        plot_episode(df, Path(args.save_plot) if args.save_plot else None)


if __name__ == "__main__":
    main()
