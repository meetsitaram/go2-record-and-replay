"""
Replay recorded episodes on the physical Go2 robot.

Reads action data from a LeRobot dataset and sends it back to the robot over
WebRTC at the original timing (20 Hz).
"""

import asyncio
import json
import sys
import time

import numpy as np

from go2_driver.connection import Go2Connection
from go2_driver.constants import SEND_RATE


async def _async_send(conn, msg: str):
    conn.datachannel.channel.send(msg)


def replay_episode(
    conn_wrapper: Go2Connection,
    actions: np.ndarray,
    buttons: np.ndarray,
    speed_multiplier: float = 1.0,
    dry_run: bool = False,
):
    """
    Replay an episode by sending recorded actions to the robot.

    Args:
        conn_wrapper: Active Go2Connection
        actions: float32 array of shape [T, 4] (lx, ly, rx, ry)
        buttons: int32 array of shape [T, 1] (keys bitmask)
        speed_multiplier: >1 = faster playback, <1 = slower
        dry_run: If True, print actions without sending
    """
    num_frames = len(actions)
    interval = SEND_RATE / speed_multiplier
    sent = 0

    print(f"  Replaying {num_frames} frames ({num_frames / 20:.1f}s at 1x)")
    if speed_multiplier != 1.0:
        print(f"  Speed: {speed_multiplier:.1f}x (interval: {interval*1000:.0f}ms)")

    for i in range(num_frames):
        tick_start = time.monotonic()

        lx, ly, rx, ry = actions[i].tolist()
        keys = int(buttons[i, 0]) if buttons.ndim > 1 else int(buttons[i])

        state_dict = {
            "lx": lx, "ly": ly,
            "rx": rx, "ry": ry,
            "keys": keys,
        }

        if dry_run:
            if i % 20 == 0 or i == num_frames - 1:
                pct = (i + 1) / num_frames * 100
                sys.stdout.write(
                    f"\r  [{pct:5.1f}%] frame {i+1}/{num_frames}  "
                    f"lx={lx:+.2f} ly={ly:+.2f} rx={rx:+.2f} ry={ry:+.2f} keys={keys:#06x}"
                    "\033[K"
                )
                sys.stdout.flush()
        else:
            try:
                msg = json.dumps({
                    "type": "msg",
                    "topic": "rt/wirelesscontroller",
                    "data": state_dict,
                })
                conn_wrapper.run_coroutine(
                    _async_send(conn_wrapper.conn, msg), timeout=1
                )
                sent += 1
            except Exception as e:
                if sent == 0:
                    print(f"  Send failed: {e}")

            if i % 20 == 0:
                pct = (i + 1) / num_frames * 100
                sys.stdout.write(f"\r  [{pct:5.1f}%] frame {i+1}/{num_frames}\033[K")
                sys.stdout.flush()

        elapsed = time.monotonic() - tick_start
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    print(f"\n  Replay complete: {sent} frames sent")


def load_episode_actions(dataset_path: str, episode_index: int = 0):
    """
    Load action data from a local LeRobot dataset for a specific episode.

    Returns (actions, buttons) numpy arrays.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pathlib import Path

    dataset_dir = Path(dataset_path)
    parquet_files = sorted(dataset_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {dataset_dir}")

    all_tables = [pq.read_table(f) for f in parquet_files]
    table = pa.concat_tables(all_tables)

    ep_col = np.array(table.column("episode_index").to_pylist())
    ep_mask = ep_col == episode_index
    if not ep_mask.any():
        raise ValueError(f"Episode {episode_index} not found in {dataset_path}")

    table = table.filter(ep_mask)

    actions = np.array(table.column("action").to_pylist(), dtype=np.float32)

    if "action.buttons" in table.column_names:
        buttons = np.array(table.column("action.buttons").to_pylist(), dtype=np.int32)
    else:
        buttons = np.zeros((len(actions), 1), dtype=np.int32)

    print(f"  Loaded episode {episode_index}: {len(actions)} frames "
          f"({len(actions) / 20:.1f}s)")

    return actions, buttons
