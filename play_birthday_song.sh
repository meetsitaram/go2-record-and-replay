#!/usr/bin/env bash
set -euo pipefail

cd /home/thor/projects/go-explore/go2-record-and-replay && \
.venv/bin/python scripts/choreo_multi.py config/choreo_show_per_song.yaml
