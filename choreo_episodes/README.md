# Choreo episodes

Curated, version-controlled teleop episodes used by the multi-robot
choreography scripts (`scripts/choreo_multi.py`).

Each subdirectory is a "mini dataset" containing just the parquet file(s)
needed by `_load_episode_data()` -- no LeRobot `meta/` is required because
the loader reads the parquet directly.

## Layout
```
choreo_episodes/
  <name>/
    episode_NNNNNN.parquet
```

## Usage in YAML

```yaml
- type: recording
  dataset: choreo_episodes/dance-song-1
  episode: 3                       # -> dance-song-1/episode_000003.parquet
```

## Adding a new episode

1. Record with `scripts/record.py` into `data/<dataset>/`.
2. Pick the take you want.
3. Copy the parquet to `choreo_episodes/<name>/episode_NNNNNN.parquet`.
4. Reference it from a choreo YAML.

The original `data/` tree stays gitignored (recordings can be huge and
include many takes/dropouts); only the curated subset lives here.
