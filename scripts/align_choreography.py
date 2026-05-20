#!/usr/bin/env python3
"""
Analyze song beats/structure and align robot moves to create a choreography.

Loads the song (Dog-song.m4a), detects beats, tempo, energy envelope, and
structural segments. Then maps available moves to appropriate time slots
based on energy level, duration fit, and beat alignment.

After Dance1, Dance2, FrontJump, and FrontPounce, inserts a position
correction step using the Move API to walk the robot back to its start.

Usage:
    .venv/bin/python scripts/align_choreography.py
"""

import json
from pathlib import Path

import librosa
import numpy as np
import yaml

SONG_PATH = Path("../assets/Dog-song.m4a")
PROFILES_PATH = Path("config/move_profiles.yaml")
PROFILES_FULL_PATH = Path("config/move_profiles_full.json")
OUTPUT_PATH = Path("config/choreography.yaml")

# Moves categorized by energy level and character
HIGH_ENERGY = ["Dance1", "Dance2"]
MEDIUM_ENERGY = ["Scrape", "FingerHeart", "FrontPounce", "FrontJump"]
LOW_ENERGY = ["Hello", "Stretch", "Pose", "StandDown"]

# Moves that need position correction afterward
NEEDS_CORRECTION = {"Dance1", "Dance2", "FrontJump", "FrontPounce"}

# Observed displacement per move (from captured profiles)
# Used to compute correction velocity * duration
MOVE_DISPLACEMENT = {
    "Dance1":      {"dx": 0.14, "dy": -0.13, "dyaw": 0.02},   # meters, radians
    "Dance2":      {"dx": 0.19, "dy":  0.03, "dyaw": -0.07},
    "FrontJump":   {"dx": 0.01, "dy":  0.00, "dyaw": -0.03},
    "FrontPounce": {"dx": 0.15, "dy": -0.03, "dyaw": 0.02},
}

# Walk-back speed (conservative to avoid slipping)
CORRECTION_SPEED = 0.2  # m/s
CORRECTION_YAW_SPEED = 0.4  # rad/s

# Minimum gap between moves (seconds) for the robot to recover
RECOVERY_GAP = 2.0


def analyze_song(song_path: str):
    """Full beat/energy/structure analysis of the song."""
    print(f"Loading {song_path}...")
    y, sr = librosa.load(song_path, sr=22050)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"  Duration: {duration:.1f}s, Sample rate: {sr}Hz")

    # Tempo and beats
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    if hasattr(tempo, '__len__'):
        tempo = float(tempo[0])
    print(f"  Tempo: {tempo:.1f} BPM, {len(beat_times)} beats detected")

    # Energy envelope (RMS) - windowed
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
    rms_norm = rms / rms.max() if rms.max() > 0 else rms

    # Onset strength for identifying accent points
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)

    # Segment the song into structural sections using spectral clustering
    # Use a simpler approach: energy-based segmentation
    segment_duration = 4.0  # analyze in 4-second windows
    n_segments = int(duration / segment_duration)
    segments = []
    for i in range(n_segments):
        t_start = i * segment_duration
        t_end = min((i + 1) * segment_duration, duration)
        # Average energy in this segment
        mask = (rms_times >= t_start) & (rms_times < t_end)
        avg_energy = float(np.mean(rms_norm[mask])) if mask.any() else 0
        # Count beats in this segment
        beat_count = int(np.sum((beat_times >= t_start) & (beat_times < t_end)))
        segments.append({
            "start": round(t_start, 2),
            "end": round(t_end, 2),
            "energy": round(avg_energy, 3),
            "beats": beat_count,
        })

    # Identify high-energy sections (above 60th percentile)
    energies = [s["energy"] for s in segments]
    high_thresh = np.percentile(energies, 60)
    low_thresh = np.percentile(energies, 30)

    for seg in segments:
        if seg["energy"] >= high_thresh:
            seg["level"] = "high"
        elif seg["energy"] >= low_thresh:
            seg["level"] = "medium"
        else:
            seg["level"] = "low"

    # Find strong downbeats (every 4 beats = 1 measure at 4/4)
    measures = []
    for i in range(0, len(beat_times), 4):
        measures.append(round(float(beat_times[i]), 3))

    return {
        "duration": round(duration, 2),
        "tempo": round(tempo, 1),
        "beat_times": [round(float(b), 3) for b in beat_times],
        "measure_starts": measures,
        "segments": segments,
        "high_energy_threshold": round(float(high_thresh), 3),
    }


def compute_correction(move_name):
    """Compute a walk-back correction to return the robot to its start position.
    
    Returns a dict with correction parameters, or None if no correction needed.
    """
    disp = MOVE_DISPLACEMENT.get(move_name)
    if not disp:
        return None

    dx = disp["dx"]
    dy = disp["dy"]
    dyaw = disp["dyaw"]

    # Distance to walk back
    dist = np.sqrt(dx**2 + dy**2)
    if dist < 0.03 and abs(dyaw) < 0.05:
        return None  # negligible, skip

    # Time needed to walk back at CORRECTION_SPEED
    walk_time = dist / CORRECTION_SPEED if dist > 0.03 else 0
    # Time to correct yaw
    yaw_time = abs(dyaw) / CORRECTION_YAW_SPEED if abs(dyaw) > 0.05 else 0
    # Total correction time (yaw and walk can overlap partially)
    correction_time = max(walk_time, yaw_time) + 0.5  # +0.5s settling

    # Velocity commands to reverse the displacement
    # Walk in the opposite direction of displacement
    vx = -dx / walk_time if walk_time > 0 else 0
    vy = -dy / walk_time if walk_time > 0 else 0
    vyaw = -dyaw / yaw_time if yaw_time > 0 else 0

    return {
        "type": "correction",
        "duration": round(correction_time, 2),
        "walk_time": round(walk_time, 2),
        "yaw_time": round(yaw_time, 2),
        "velocity": {
            "x": round(vx, 3),
            "y": round(vy, 3),
            "yaw": round(vyaw, 3),
        },
        "reverses": move_name,
    }


def find_nearest_beat(target_time, beat_times):
    """Find the nearest beat to a target time."""
    idx = np.argmin(np.abs(np.array(beat_times) - target_time))
    return beat_times[idx]


def plan_choreography(song_analysis, move_profiles):
    """Place moves into the song timeline based on energy alignment."""
    segments = song_analysis["segments"]
    beat_times = song_analysis["beat_times"]
    measure_starts = song_analysis["measure_starts"]
    duration = song_analysis["duration"]

    # Build move catalog with durations
    moves = {}
    for name, profile in move_profiles.items():
        moves[name] = {
            "duration": profile["duration"],
            "angular_energy": profile["angular_energy"],
            "api_id": profile["api_id"],
        }

    # Strategy:
    # 1. Place high-energy moves (Dance1, Dance2) in high-energy sections
    # 2. Place medium moves in medium sections
    # 3. Place low/cute moves in low sections or intros/outros
    # 4. After displacement-heavy moves, insert correction steps
    # 5. Leave gaps marked as "custom" for user-recorded moves later

    placements = []
    occupied_until = 0.0  # timeline cursor

    # First pass: identify good slots for each energy level
    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        level = seg["level"]

        # Skip if we're still in a previous move
        if seg_start < occupied_until:
            continue

        # Find the nearest measure start within this segment for alignment
        candidates = [m for m in measure_starts if seg_start <= m < seg_end]
        if not candidates:
            candidates = [find_nearest_beat(seg_start, beat_times)]
        start_time = candidates[0]

        if start_time < occupied_until:
            start_time = occupied_until + RECOVERY_GAP
            start_time = find_nearest_beat(start_time, beat_times)

        if start_time >= duration - 5:
            break

        # Pick a move based on energy level
        if level == "high":
            pool = HIGH_ENERGY
        elif level == "medium":
            pool = MEDIUM_ENERGY
        else:
            pool = LOW_ENERGY

        # Find a move that fits (hasn't been used too recently)
        recent_names = [p["move"] for p in placements[-3:] if p.get("type") != "correction"]
        chosen = None
        for candidate_name in pool:
            if candidate_name in moves and candidate_name not in recent_names:
                move_dur = moves[candidate_name]["duration"]
                # Account for correction time if needed
                correction = compute_correction(candidate_name)
                total_dur = move_dur + (correction["duration"] if correction else 0)
                if start_time + total_dur <= duration:
                    chosen = candidate_name
                    break

        if not chosen:
            for candidate_name in pool:
                if candidate_name in moves:
                    move_dur = moves[candidate_name]["duration"]
                    if start_time + move_dur <= duration:
                        chosen = candidate_name
                        break

        if chosen:
            move_dur = moves[chosen]["duration"]
            placements.append({
                "move": chosen,
                "type": "builtin_move",
                "start_time": round(start_time, 3),
                "end_time": round(start_time + move_dur, 3),
                "duration": move_dur,
                "api_id": moves[chosen]["api_id"],
                "energy_level": level,
                "beat_aligned": True,
            })
            occupied_until = start_time + move_dur

            # Insert correction if this move causes displacement
            if chosen in NEEDS_CORRECTION:
                correction = compute_correction(chosen)
                if correction:
                    corr_start = occupied_until + 0.5  # brief pause
                    corr_end = corr_start + correction["duration"]
                    placements.append({
                        "type": "correction",
                        "mode": "closed_loop",
                        "start_time": round(corr_start, 3),
                        "end_time": round(corr_end, 3),
                        "max_duration": correction["duration"] + 1.0,
                        "duration": correction["duration"],
                        "speed": CORRECTION_SPEED,
                        "yaw_speed": CORRECTION_YAW_SPEED,
                        "position_tolerance": 0.05,
                        "yaw_tolerance": 0.05,
                        "fallback_velocity": correction["velocity"],
                        "reverses": correction["reverses"],
                    })
                    occupied_until = corr_end

            occupied_until += RECOVERY_GAP

    return placements


def identify_gaps(placements, song_duration):
    """Find gaps between moves where custom moves can be inserted."""
    gaps = []
    prev_end = 0.0

    for p in placements:
        gap_start = prev_end
        gap_end = p["start_time"]
        gap_dur = gap_end - gap_start
        if gap_dur > 3.0:  # Only mark gaps > 3s as opportunities
            gaps.append({
                "start": round(gap_start, 2),
                "end": round(gap_end, 2),
                "duration": round(gap_dur, 2),
                "note": "available for custom move",
            })
        prev_end = p["end_time"]

    # Trailing gap
    if song_duration - prev_end > 3.0:
        gaps.append({
            "start": round(prev_end, 2),
            "end": round(song_duration, 2),
            "duration": round(song_duration - prev_end, 2),
            "note": "available for custom move (ending)",
        })

    return gaps


def main():
    # Load move profiles
    with open(PROFILES_PATH) as f:
        move_profiles = yaml.safe_load(f)
    print(f"Loaded {len(move_profiles)} move profiles\n")

    # Analyze song
    song = analyze_song(str(SONG_PATH))

    print(f"\n  Song structure ({len(song['segments'])} segments):")
    print(f"  {'Time':>10s} {'Energy':>7s} {'Beats':>5s} {'Level':>6s}")
    print(f"  {'-'*10:>10s} {'-'*6:>7s} {'-'*5:>5s} {'-'*6:>6s}")
    for seg in song["segments"]:
        bar = "█" * int(seg["energy"] * 20)
        print(f"  {seg['start']:>5.1f}-{seg['end']:<4.1f} {seg['energy']:>6.3f} {seg['beats']:>5d} {seg['level']:>6s}  {bar}")

    # Plan choreography
    print("\n\nPlanning choreography...")
    placements = plan_choreography(song, move_profiles)

    moves_only = [p for p in placements if p["type"] == "builtin_move"]
    corrections = [p for p in placements if p["type"] == "correction"]

    print(f"\n  Choreography ({len(moves_only)} moves + {len(corrections)} corrections):")
    print(f"  {'Time':>12s} {'Type':<10s} {'Move/Action':<16s} {'Dur':>5s} {'Energy':>6s}")
    print(f"  {'-'*12:>12s} {'-'*9:<10s} {'-'*15:<16s} {'-'*4:>5s} {'-'*6:>6s}")
    for p in placements:
        if p["type"] == "builtin_move":
            print(f"  {p['start_time']:>5.1f}-{p['end_time']:<5.1f} {'MOVE':<10s} {p['move']:<16s} {p['duration']:>4.1f}s {p['energy_level']:>6s}")
        elif p["type"] == "correction":
            print(f"  {p['start_time']:>5.1f}-{p['end_time']:<5.1f} {'CORRECT':<10s} {'<-'+p['reverses']:<16s} {p['max_duration']:>4.1f}s  (closed-loop)")

    # Find gaps for custom moves
    gaps = identify_gaps(placements, song["duration"])

    total_move_time = sum(p["duration"] for p in moves_only)
    total_correction_time = sum(p["duration"] for p in corrections)
    total_covered = total_move_time + total_correction_time
    total_gap = sum(g["duration"] for g in gaps)

    print(f"\n  Coverage:")
    print(f"    Song duration:    {song['duration']:.1f}s")
    print(f"    Moves placed:     {total_move_time:.1f}s ({100*total_move_time/song['duration']:.0f}%)")
    print(f"    Corrections:      {total_correction_time:.1f}s ({100*total_correction_time/song['duration']:.0f}%)")
    print(f"    Gaps available:   {total_gap:.1f}s ({100*total_gap/song['duration']:.0f}%) - for custom moves")

    # Save choreography
    output = {
        "song": {
            "file": str(SONG_PATH),
            "duration": song["duration"],
            "tempo": song["tempo"],
            "num_beats": len(song["beat_times"]),
            "num_measures": len(song["measure_starts"]),
        },
        "moves": placements,
        "gaps": gaps,
        "beat_times": song["beat_times"],
        "measure_starts": song["measure_starts"],
        "segments": song["segments"],
    }

    def to_python_native(obj):
        """Recursively convert numpy types to Python native for YAML serialization."""
        if isinstance(obj, dict):
            return {k: to_python_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_python_native(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output = to_python_native(output)

    with open(OUTPUT_PATH, "w") as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)
    print(f"\n  Choreography saved to {OUTPUT_PATH}")
    print(f"  (Gaps can be filled with custom xbox-controller moves later)")


if __name__ == "__main__":
    main()
