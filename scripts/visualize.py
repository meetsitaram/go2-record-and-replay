#!/usr/bin/env python3
"""
Visualize a recorded episode: renders overlays onto the video and saves as MP4.

Overlays include joystick positions, robot state, button actions, and a
progress bar — all baked into a single video file for easy playback.

Usage:
    python scripts/visualize.py --dataset ./data/go2-look-around
    python scripts/visualize.py --dataset ./data/go2-look-around --episode 0
    python scripts/visualize.py --dataset ./data/go2-look-around -o my_viz.mp4
"""

import argparse
import subprocess
import sys
from pathlib import Path

import av
import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from go2_driver.constants import JOINT_NAMES, BUTTON_ACTIONS


# ── Colors (BGR) ─────────────────────────────────────────────────────────────

BG_DARK = (30, 30, 30)
BG_PANEL = (40, 40, 40)
WHITE = (255, 255, 255)
GRAY = (140, 140, 140)
DIM = (80, 80, 80)
GREEN = (80, 220, 100)
RED = (80, 80, 240)
BLUE = (220, 160, 60)
CYAN = (220, 200, 60)
YELLOW = (60, 220, 240)
ORANGE = (40, 140, 240)
PROGRESS_BG = (60, 60, 60)
PROGRESS_FG = (220, 160, 60)
STICK_BG = (60, 60, 60)
STICK_RIM = (100, 100, 100)
STICK_DOT = (80, 220, 100)
STICK_CROSS = (70, 70, 70)


def load_episode_data(dataset_path: str, episode_index: int):
    """Load all data for an episode from parquet and video."""

    dataset_dir = Path(dataset_path)
    parquet_files = sorted(dataset_dir.rglob("*.parquet"))
    if not parquet_files:
        print(f"  No parquet files found in {dataset_dir}")
        sys.exit(1)

    all_tables = [pq.read_table(f) for f in parquet_files]
    table = pa.concat_tables(all_tables)

    ep_mask = np.array(table.column("episode_index").to_pylist()) == episode_index
    if not ep_mask.any():
        print(f"  Episode {episode_index} not found.")
        sys.exit(1)
    table = table.filter(ep_mask)

    data = {}
    for col in table.column_names:
        raw = table.column(col).to_pylist()
        if isinstance(raw[0], list):
            data[col] = np.array(raw, dtype=np.float32)
        else:
            data[col] = np.array(raw)

    video_files = sorted(dataset_dir.rglob(f"*episode_{episode_index:06d}.mp4"))
    frames = []
    if video_files:
        container = av.open(str(video_files[0]))
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format="bgr24")
            frames.append(img)
        container.close()

    return data, frames


def draw_stick(canvas, cx, cy, radius, sx, sy, label):
    """Draw a joystick visualization at (cx, cy)."""
    cv2.circle(canvas, (cx, cy), radius, STICK_BG, -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), radius, STICK_RIM, 2, cv2.LINE_AA)
    cv2.line(canvas, (cx - radius, cy), (cx + radius, cy), STICK_CROSS, 1)
    cv2.line(canvas, (cx, cy - radius), (cx, cy + radius), STICK_CROSS, 1)

    dx = int(sx * radius * 0.9)
    dy = int(-sy * radius * 0.9)
    dot_x = cx + dx
    dot_y = cy + dy
    cv2.circle(canvas, (dot_x, dot_y), 8, STICK_DOT, -1, cv2.LINE_AA)
    cv2.circle(canvas, (dot_x, dot_y), 8, WHITE, 1, cv2.LINE_AA)

    cv2.putText(canvas, label, (cx - radius, cy + radius + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1, cv2.LINE_AA)


def draw_progress_bar(canvas, x, y, w, h, progress, frame_idx, total_frames, timestamp):
    """Draw a progress bar with frame info."""
    cv2.rectangle(canvas, (x, y), (x + w, y + h), PROGRESS_BG, -1)
    fill_w = int(w * progress)
    if fill_w > 0:
        cv2.rectangle(canvas, (x, y), (x + fill_w, y + h), PROGRESS_FG, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), STICK_RIM, 1)

    text = f"Frame {frame_idx + 1}/{total_frames}  |  {timestamp:.2f}s"
    cv2.putText(canvas, text, (x + 8, y + h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)


def draw_text_line(canvas, x, y, label, value, color=WHITE, label_color=GRAY):
    """Draw a label: value line."""
    cv2.putText(canvas, label, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, label_color, 1, cv2.LINE_AA)
    cv2.putText(canvas, value, (x + 10 + len(label) * 9, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return y + 20


def get_button_names(keys_bitmask):
    """Decode bitmask to active button action names."""
    if keys_bitmask == 0:
        return "none"
    names = []
    for mask, desc in BUTTON_ACTIONS:
        if (keys_bitmask & mask) == mask:
            names.append(desc.split("(")[0].strip())
            keys_bitmask &= ~mask
    return ", ".join(names) if names else f"0x{keys_bitmask:04x}"


def get_button_action_list(keys_bitmask):
    """Return list of (mask, action_name) for all active buttons."""
    if keys_bitmask == 0:
        return []
    actions = []
    remaining = keys_bitmask
    for mask, desc in BUTTON_ACTIONS:
        if (remaining & mask) == mask:
            actions.append(desc)
            remaining &= ~mask
    if remaining and not actions:
        actions.append(f"Buttons: 0x{keys_bitmask:04x}")
    return actions


FADE_FRAMES = 50  # 2.5 seconds at 20 fps
HOLD_FRAMES = 10  # 0.5 seconds fully visible before fading


def build_action_events(data):
    """Pre-scan all frames to find button press events with their start frames."""
    events = []  # list of (start_frame, action_text)
    total = len(data["action.buttons"])
    prev_keys = 0

    for i in range(total):
        keys = int(data["action.buttons"][i][0])
        new_presses = keys & ~prev_keys
        if new_presses:
            actions = get_button_action_list(keys)
            for action_text in actions:
                events.append((i, action_text))
        prev_keys = keys

    return events


def draw_action_toast(canvas, events, frame_idx):
    """Draw fading action labels for recent button presses."""
    h, w = canvas.shape[:2]
    visible = []

    for start_frame, text in events:
        age = frame_idx - start_frame
        if age < 0 or age >= HOLD_FRAMES + FADE_FRAMES:
            continue

        if age < HOLD_FRAMES:
            alpha = 1.0
        else:
            alpha = 1.0 - (age - HOLD_FRAMES) / FADE_FRAMES

        visible.append((text, alpha))

    if not visible:
        return

    toast_x = w // 2
    base_y = 70

    for i, (text, alpha) in enumerate(visible[-3:]):
        y = base_y + i * 45

        font_scale = 0.75
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        tx = toast_x - tw // 2
        pad = 12

        bg_region_y1 = max(0, y - th - pad)
        bg_region_y2 = min(h, y + pad)
        bg_region_x1 = max(0, tx - pad)
        bg_region_x2 = min(w, tx + tw + pad)

        region = canvas[bg_region_y1:bg_region_y2, bg_region_x1:bg_region_x2]
        if region.size > 0:
            dark = np.full_like(region, (20, 20, 20))
            bg_alpha = 0.7 * alpha
            cv2.addWeighted(region, 1 - bg_alpha, dark, bg_alpha, 0, region)
            canvas[bg_region_y1:bg_region_y2, bg_region_x1:bg_region_x2] = region

        color = (
            int(YELLOW[0] * alpha),
            int(YELLOW[1] * alpha),
            int(YELLOW[2] * alpha),
        )
        shadow = (
            int(20 * alpha),
            int(20 * alpha),
            int(20 * alpha),
        )

        cv2.putText(canvas, text, (tx + 2, y + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, shadow, thickness + 1, cv2.LINE_AA)
        cv2.putText(canvas, text, (tx, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def draw_state_panel(canvas, x, y, data, frame_idx):
    """Draw robot state information panel."""
    panel_w = 300
    panel_h = 520
    overlay = canvas[y:y + panel_h, x:x + panel_w].copy()
    dark = np.full_like(overlay, BG_PANEL)
    cv2.addWeighted(overlay, 0.3, dark, 0.7, 0, overlay)
    canvas[y:y + panel_h, x:x + panel_w] = overlay

    tx, ty = x + 10, y + 22

    cv2.putText(canvas, "ROBOT STATE", (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, CYAN, 1, cv2.LINE_AA)
    ty += 28

    # Position
    state = data["observation.state"][frame_idx]
    has_state = np.any(state != 0)
    color = WHITE if has_state else DIM

    ty = draw_text_line(canvas, tx, ty, "Pos:", f"({state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f})", color)
    ty = draw_text_line(canvas, tx, ty, "Vel:", f"({state[3]:.2f}, {state[4]:.2f}, {state[5]:.2f})", color)
    ty = draw_text_line(canvas, tx, ty, "Yaw:", f"{state[6]:.3f} rad/s", color)
    ty = draw_text_line(canvas, tx, ty, "Batt:", f"{state[15]:.0f}%", GREEN if state[15] > 20 else RED)
    ty += 8

    # Lidar pose
    if "observation.lidar_pose" in data:
        pose = data["observation.lidar_pose"][frame_idx]
        has_pose = np.any(pose != 0)
        color = WHITE if has_pose else DIM
        cv2.putText(canvas, "LIDAR POSE", (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1, cv2.LINE_AA)
        ty += 22
        ty = draw_text_line(canvas, tx, ty, "XYZ:", f"({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f})", color)
        ty = draw_text_line(canvas, tx, ty, "RPY:", f"({pose[3]:.2f}, {pose[4]:.2f}, {pose[5]:.2f})", color)
        ty += 8

    # Joint positions (compact: 4 legs x 3 joints)
    if "observation.joint_positions" in data:
        jpos = data["observation.joint_positions"][frame_idx]
        has_joints = np.any(jpos != 0)
        color = WHITE if has_joints else DIM
        cv2.putText(canvas, "JOINTS (rad)", (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1, cv2.LINE_AA)
        ty += 22
        for leg_i, leg_name in enumerate(["FR", "FL", "RR", "RL"]):
            base = leg_i * 3
            vals = f"{jpos[base]:+.2f} {jpos[base+1]:+.2f} {jpos[base+2]:+.2f}"
            ty = draw_text_line(canvas, tx, ty, f"{leg_name}:", vals, color)
        ty += 8

    # Motor temperatures
    if "observation.motor_temperatures" in data:
        temps = data["observation.motor_temperatures"][frame_idx]
        has_temps = np.any(temps != 0)
        if has_temps:
            max_temp = np.max(temps)
            color = RED if max_temp > 60 else (YELLOW if max_temp > 45 else GREEN)
            cv2.putText(canvas, "MOTOR TEMP", (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1, cv2.LINE_AA)
            ty += 22
            ty = draw_text_line(canvas, tx, ty, "Range:",
                                f"{np.min(temps):.0f} - {max_temp:.0f} C", color)
            ty += 8

    # Power
    if "observation.power" in data:
        pwr = data["observation.power"][frame_idx]
        has_power = np.any(pwr != 0)
        if has_power:
            cv2.putText(canvas, "POWER", (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1, cv2.LINE_AA)
            ty += 22
            ty = draw_text_line(canvas, tx, ty, "V/A:", f"{pwr[0]:.1f}V  {pwr[1]:.2f}A", WHITE)

    # Freshness
    if "observation.lowstate_fresh" in data:
        fresh = data["observation.lowstate_fresh"][frame_idx]
        fresh_val = fresh[0] if len(fresh.shape) > 0 else fresh
        indicator = "LIVE" if fresh_val > 0.5 else "stale"
        color = GREEN if fresh_val > 0.5 else DIM
        ty = draw_text_line(canvas, tx, ty, "LowState:", indicator, color)


def render_frame(video_frame, data, frame_idx, total_frames, speed, paused, events=None):
    """Compose a full visualization frame."""
    h, w = video_frame.shape[:2]
    canvas = video_frame.copy()

    # ── Action overlay (bottom-left) ─────────────────────────────────────
    action = data["action"][frame_idx]
    lx, ly, rx, ry = action[0], action[1], action[2], action[3]
    keys = int(data["action.buttons"][frame_idx][0])

    panel_y = h - 160
    panel_h = 150
    panel_w = 320
    overlay = canvas[panel_y:panel_y + panel_h, 0:panel_w].copy()
    dark = np.full_like(overlay, BG_PANEL)
    cv2.addWeighted(overlay, 0.3, dark, 0.7, 0, overlay)
    canvas[panel_y:panel_y + panel_h, 0:panel_w] = overlay

    cv2.putText(canvas, "ACTIONS", (10, panel_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1, cv2.LINE_AA)

    r = 40
    draw_stick(canvas, 75, panel_y + 85, r, lx, ly, "L-Stick")
    draw_stick(canvas, 220, panel_y + 85, r, rx, ry, "R-Stick")

    btn_text = get_button_names(keys)
    cv2.putText(canvas, btn_text, (10, panel_y + 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, YELLOW if keys else DIM, 1, cv2.LINE_AA)

    # ── Button action toasts (top-center, fading) ────────────────────────
    if events:
        draw_action_toast(canvas, events, frame_idx)

    # ── State panel (top-right) ──────────────────────────────────────────
    draw_state_panel(canvas, w - 310, 10, data, frame_idx)

    # ── Top-left info ────────────────────────────────────────────────────
    info_overlay = canvas[0:30, 0:400].copy()
    dark = np.full_like(info_overlay, BG_PANEL)
    cv2.addWeighted(info_overlay, 0.3, dark, 0.7, 0, info_overlay)
    canvas[0:30, 0:400] = info_overlay

    status = "PAUSED" if paused else f"PLAYING {speed:.1f}x"
    color = YELLOW if paused else GREEN
    ts = data["timestamp"][frame_idx]
    cv2.putText(canvas, f"{status}  |  {ts:.2f}s  |  Frame {frame_idx+1}/{total_frames}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # ── Progress bar (very bottom) ───────────────────────────────────────
    bar_h = 6
    progress = (frame_idx + 1) / total_frames
    cv2.rectangle(canvas, (0, h - bar_h), (w, h), PROGRESS_BG, -1)
    fill_w = int(w * progress)
    if fill_w > 0:
        cv2.rectangle(canvas, (0, h - bar_h), (fill_w, h), PROGRESS_FG, -1)

    return canvas


def main():
    p = argparse.ArgumentParser(description="Visualize a recorded episode (renders to MP4)")
    p.add_argument("--dataset", required=True, help="Path to LeRobot dataset directory")
    p.add_argument("--episode", type=int, default=0, help="Episode index")
    p.add_argument("-o", "--output", default=None,
                   help="Output MP4 path (default: <dataset>/viz_episode_<N>.mp4)")
    args = p.parse_args()

    print(f"  Loading episode {args.episode} from {args.dataset} ...")
    data, video_frames = load_episode_data(args.dataset, args.episode)

    total_frames = len(data["timestamp"])
    has_video = len(video_frames) > 0
    print(f"  Data: {total_frames} frames, Video: {len(video_frames)} frames")

    if not has_video:
        print("  No video found. Creating blank frames.")
        video_frames = [np.zeros((720, 1280, 3), dtype=np.uint8)] * total_frames

    if len(video_frames) < total_frames:
        last = video_frames[-1] if video_frames else np.zeros((720, 1280, 3), dtype=np.uint8)
        video_frames.extend([last] * (total_frames - len(video_frames)))

    output_path = args.output or str(
        Path(args.dataset) / f"viz_episode_{args.episode:03d}.mp4"
    )

    h, w = video_frames[0].shape[:2]
    fps = 20

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path,
    ]

    events = build_action_events(data)
    print(f"  Button events: {len(events)}")
    print(f"  Rendering {total_frames} frames to {output_path} ...")

    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    for frame_idx in range(total_frames):
        canvas = render_frame(
            video_frames[frame_idx], data, frame_idx, total_frames,
            speed=1.0, paused=False, events=events,
        )
        proc.stdin.write(canvas.tobytes())

        if (frame_idx + 1) % 100 == 0 or frame_idx == total_frames - 1:
            pct = (frame_idx + 1) / total_frames * 100
            ts = data["timestamp"][frame_idx]
            sys.stdout.write(
                f"\r  [{pct:5.1f}%]  frame {frame_idx+1}/{total_frames}  ({ts:.1f}s)"
            )
            sys.stdout.flush()

    proc.stdin.close()
    proc.wait()
    print(f"\n  Saved: {output_path}")
    print("  Done.")


if __name__ == "__main__":
    main()
