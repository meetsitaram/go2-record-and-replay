#!/usr/bin/env python3
"""
Inspect a recorded LeRobot dataset: print summary stats, validate structure,
and optionally plot trajectories.

Usage:
    python scripts/inspect_episode.py --dataset ./data/go2-teleop
    python scripts/inspect_episode.py --dataset ./data/go2-teleop --episode 0 --plot
"""

import argparse
import sys


def main():
    p = argparse.ArgumentParser(description="Inspect a recorded Go2 dataset")
    p.add_argument("--dataset", required=True, help="Path to LeRobot dataset directory")
    p.add_argument("--episode", type=int, default=None,
                   help="Inspect a specific episode (default: all)")
    p.add_argument("--plot", action="store_true",
                   help="Plot trajectory and action time series")

    args = p.parse_args()

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        print("  ERROR: lerobot not installed. Run: uv pip install lerobot")
        sys.exit(1)

    print("=" * 60)
    print(f"  Dataset: {args.dataset}")
    print("=" * 60)

    dataset = LeRobotDataset(args.dataset)

    print(f"  Episodes:  {dataset.num_episodes}")
    print(f"  Frames:    {dataset.num_frames}")
    print(f"  FPS:       {dataset.fps}")
    print(f"  Duration:  {dataset.num_frames / dataset.fps:.1f}s")
    print(f"  Features:  {list(dataset.features.keys())}")

    if hasattr(dataset.meta, "robot_type"):
        print(f"  Robot:     {dataset.meta.robot_type}")

    # Per-episode summary
    print()
    print("  Episodes:")
    print(f"  {'Idx':>4}  {'Frames':>7}  {'Duration':>9}")
    print(f"  {'---':>4}  {'------':>7}  {'--------':>9}")

    for ep_idx in range(dataset.num_episodes):
        if args.episode is not None and ep_idx != args.episode:
            continue
        ep_start = dataset.episode_data_index["from"][ep_idx].item()
        ep_end = dataset.episode_data_index["to"][ep_idx].item()
        n_frames = ep_end - ep_start
        duration = n_frames / dataset.fps
        print(f"  {ep_idx:>4}  {n_frames:>7}  {duration:>8.1f}s")

    # Sample data
    if dataset.num_frames > 0:
        sample = dataset[0]
        print()
        print("  Sample frame (index 0):")
        for key, value in sorted(sample.items()):
            if hasattr(value, "shape"):
                print(f"    {key}: shape={list(value.shape)}, dtype={value.dtype}")
            else:
                print(f"    {key}: {value}")

    # Feature stats
    if hasattr(dataset, "meta") and hasattr(dataset.meta, "stats") and dataset.meta.stats:
        print()
        print("  Feature stats (global):")
        for key in sorted(dataset.meta.stats.keys()):
            stats = dataset.meta.stats[key]
            if "mean" in stats and "std" in stats:
                mean_vals = stats["mean"]
                std_vals = stats["std"]
                if hasattr(mean_vals, "__len__") and len(mean_vals) <= 6:
                    print(f"    {key}: mean={[f'{v:.3f}' for v in mean_vals]}, "
                          f"std={[f'{v:.3f}' for v in std_vals]}")
                else:
                    print(f"    {key}: mean_norm={sum(v**2 for v in mean_vals)**0.5:.3f}")

    # Plot
    if args.plot:
        _plot_episode(dataset, args.episode or 0)


def _plot_episode(dataset, episode_index: int):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  ERROR: matplotlib not installed. Run: uv pip install matplotlib")
        return

    ep_start = dataset.episode_data_index["from"][episode_index].item()
    ep_end = dataset.episode_data_index["to"][episode_index].item()

    actions = []
    positions = []
    for idx in range(ep_start, ep_end):
        sample = dataset[idx]
        actions.append(sample["action"].numpy())
        if "observation.state" in sample:
            positions.append(sample["observation.state"].numpy()[:3])

    actions = np.array(actions)
    t = np.arange(len(actions)) / dataset.fps

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].set_title(f"Episode {episode_index} - Actions")
    for i, name in enumerate(["lx", "ly", "rx", "ry"]):
        axes[0].plot(t, actions[:, i], label=name)
    axes[0].set_ylabel("Value")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if positions:
        positions = np.array(positions)
        axes[1].set_title("Robot Position (XY)")
        axes[1].plot(positions[:, 0], positions[:, 1], "b-", alpha=0.7)
        axes[1].plot(positions[0, 0], positions[0, 1], "go", markersize=10, label="start")
        axes[1].plot(positions[-1, 0], positions[-1, 1], "rs", markersize=10, label="end")
        axes[1].set_xlabel("X (m)")
        axes[1].set_ylabel("Y (m)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_aspect("equal")

    plt.tight_layout()
    out_path = f"episode_{episode_index}_plot.png"
    plt.savefig(out_path, dpi=150)
    print(f"  Plot saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
