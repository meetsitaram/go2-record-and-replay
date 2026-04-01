"""
LeRobot dataset recorder.

Wraps LeRobotDataset.create / add_frame / save_episode / finalize into a simple
interface used by the recording loop.
"""

from pathlib import Path

import numpy as np

from .constants import LEROBOT_FEATURES, DATASET_FPS, features_without_camera, features_without_lidar


class EpisodeRecorder:
    """
    Records teleoperation episodes into a LeRobot v3.0 dataset.

    Usage:
        recorder = EpisodeRecorder(repo_id="user/go2-teleop", root="./data")
        recorder.create(task="walk around", use_camera=True)

        # 20 Hz loop:
        recorder.add_frame(action, buttons, observations)

        recorder.save_episode()    # end current episode
        recorder.finalize()        # close dataset (must call before push)
        recorder.push_to_hub()     # optional
    """

    def __init__(self, repo_id: str, root: str = "./data",
                 use_camera: bool = True, use_lidar: bool = True):
        self.repo_id = repo_id
        self.root = Path(root)
        self.use_camera = use_camera
        self.use_lidar = use_lidar
        self.dataset = None
        self._frame_count = 0
        self._episode_count = 0

        # Build features based on what's enabled
        self.features = dict(LEROBOT_FEATURES)
        if not use_camera:
            self.features.pop("observation.images.front", None)
        if not use_lidar:
            self.features.pop("observation.lidar_pose", None)

    def create(self, task: str = "teleoperation"):
        """Create a new LeRobot dataset. Removes any leftover data at the same path."""
        import shutil
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset_dir = self.root / self.repo_id.replace("/", "_")
        if dataset_dir.exists():
            print(f"  Removing existing dataset at {dataset_dir}")
            shutil.rmtree(dataset_dir)

        self.dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=DATASET_FPS,
            features=self.features,
            robot_type="unitree_go2",
            root=str(dataset_dir),
            use_videos=self.use_camera,
            image_writer_threads=4,
            tolerance_s=1 / DATASET_FPS,
        )
        self._episode_start: float | None = None
        # lerobot 0.3.2 bug: validate_frame compares numpy tuple shapes against
        # list shapes from the feature dict, which always fails. Patch to tuples.
        for feat in self.dataset.features.values():
            if "shape" in feat and isinstance(feat["shape"], list):
                feat["shape"] = tuple(feat["shape"])

        self._default_task = task
        print(f"  Dataset created: {self.repo_id}")
        print(f"  Root: {self.dataset.root}")
        print(f"  Features: {list(self.features.keys())}")

    def add_frame(self, action: np.ndarray, buttons: np.ndarray,
                  observations: dict, task: str | None = None):
        """
        Add a single frame to the current episode.

        Args:
            action: float32 [4] array (lx, ly, rx, ry)
            buttons: int32 [1] array (keys bitmask)
            observations: dict from RobotStreams.snapshot()
            task: optional task description override
        """
        if self.dataset is None:
            raise RuntimeError("Call create() before add_frame()")

        import time
        now = time.monotonic()
        if self._episode_start is None:
            self._episode_start = now
        timestamp = now - self._episode_start

        frame = {
            "action": action,
            "action.buttons": buttons,
        }
        frame_task = task or self._default_task

        # Merge observation arrays
        for key in [
            "observation.state",
            "observation.joint_positions",
            "observation.joint_velocities",
            "observation.joint_torques",
            "observation.motor_temperatures",
            "observation.power",
            "observation.lowstate_fresh",
        ]:
            if key in observations:
                frame[key] = observations[key]
            elif key in self.features:
                frame[key] = np.zeros(self.features[key]["shape"], dtype=np.float32)

        if self.use_lidar:
            if "observation.lidar_pose" in observations:
                frame["observation.lidar_pose"] = observations["observation.lidar_pose"]
            else:
                frame["observation.lidar_pose"] = np.zeros(6, dtype=np.float32)

        if self.use_camera:
            if "observation.images.front" in observations:
                frame["observation.images.front"] = observations["observation.images.front"]
            else:
                # Black frame placeholder until camera starts
                frame["observation.images.front"] = np.zeros((720, 1280, 3), dtype=np.uint8)

        self.dataset.add_frame(frame, frame_task, timestamp=timestamp)
        self._frame_count += 1

    def save_episode(self):
        """Finalize the current episode and start a new one."""
        if self.dataset is None:
            return
        if self._frame_count == 0:
            return
        if self.dataset.episode_buffer is None:
            return
        if "size" not in self.dataset.episode_buffer:
            return

        self.dataset.save_episode()
        self._episode_count += 1
        frames = self._frame_count
        self._frame_count = 0
        self._episode_start = None
        print(f"\n  Episode {self._episode_count} saved ({frames} frames, "
              f"{frames / DATASET_FPS:.1f}s)")

    def finalize(self):
        """Save any in-progress episode and consolidate the dataset."""
        if self.dataset is None:
            return
        if self._frame_count > 0:
            self.save_episode()
        # lerobot 0.3.2 has no finalize(); save_episode is sufficient
        print(f"\n  Dataset complete: {self._episode_count} episodes total")

    def push_to_hub(self, private: bool = False):
        """Push the dataset to Hugging Face Hub."""
        if self.dataset is None:
            return
        self.dataset.push_to_hub(
            private=private,
            push_videos=self.use_camera,
            tags=["robotics", "locomotion", "unitree-go2", "teleoperation"],
        )
        print(f"  Pushed to https://huggingface.co/datasets/{self.repo_id}")

    @property
    def is_recording(self) -> bool:
        return self._frame_count > 0
