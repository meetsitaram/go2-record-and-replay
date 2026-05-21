cd /home/thor/projects/go-explore/go2-record-and-replay && ffplay -nodisp -autoexit assets/Dog-song.m4a 2>/dev/null & .venv/bin/python -u scripts/record.py --mode sta --ip 192.168.1.246 --aes-key 2c09e23856fa423ed680313dd939a3f0 --allow-all --no-camera --auto-record --speed-limit 1.0 2>&1


cd /home/thor/projects/go-explore/go2-record-and-replay && ffplay -nodisp -autoexit assets/Dog-song.m4a 2>/dev/null & .venv/bin/python -u scripts/record.py --mode sta --ip 192.168.1.246 --aes-key 2c09e23856fa423ed680313dd939a3f0 --allow-all --no-camera --auto-record --speed-limit 1.0 --audio-head-start 1.0 --repo-id dance-moves-v2 2>&1


### teleop
cd /home/thor/projects/go-explore/go2-record-and-replay && \
.venv/bin/python scripts/record.py --mode sta --ip 192.168.1.246 \
  --aes-key 2c09e23856fa423ed680313dd939a3f0 \
  --speed-limit 0.5


cd /home/thor/projects/go-explore/go2-record-and-replay && \
.venv/bin/python scripts/record.py --mode sta --ip 192.168.1.246 \
  --aes-key 2c09e23856fa423ed680313dd939a3f0 \
  --speed-limit 0.5 --allow-all

.venv/bin/python scripts/record.py --mode sta --ip 192.168.1.246 \
  --aes-key 2c09e23856fa423ed680313dd939a3f0 \
  --speed-limit 0.5 --allow-all --no-countdown


### replay song and dance
cd /home/thor/projects/go-explore/go2-record-and-replay && \
.venv/bin/python scripts/replay_teleop.py 192.168.1.246 \
  --aes-key 2c09e23856fa423ed680313dd939a3f0 \
  --dataset data/dance-moves-v2/data/chunk-000


<!-- #### recorded episode - dance-song-1
cd /home/thor/projects/go-explore/go2-record-and-replay && \
.venv/bin/python scripts/record.py --mode sta --ip 192.168.1.246 \
  --aes-key 2c09e23856fa423ed680313dd939a3f0 --speed-limit 0.5 \
  --allow-all --no-countdown --repo-id dance-song-1

#### recorded episode - dance-song-2
cd /home/thor/projects/go-explore/go2-record-and-replay && \
.venv/bin/python scripts/record.py --mode sta --ip 192.168.1.246 \
  --aes-key 2c09e23856fa423ed680313dd939a3f0 --speed-limit 1.0 \
  --allow-all --no-countdown --repo-id dance-song-2 -->

#### recorded episode - dance-song-4
cd /home/thor/projects/go-explore/go2-record-and-replay && \
.venv/bin/python scripts/record.py --mode sta --ip 192.168.1.246 \
  --aes-key 2c09e23856fa423ed680313dd939a3f0 --speed-limit 1.0 \
  --allow-all --no-countdown --repo-id dance-song-4 

### play dance-song-first along with the dance moves
cd /home/thor/projects/go-explore/go2-record-and-replay && \
.venv/bin/python scripts/replay_teleop.py 192.168.1.246 \
  --aes-key 2c09e23856fa423ed680313dd939a3f0 \
  --dataset data/dance-song-1/data/chunk-000 --episode 3 \
  --song ../assets/first-song.m4a --audio-head-start 4.0

### play dance-song-third along with the dance moves
cd /home/thor/projects/go-explore/go2-record-and-replay && \
.venv/bin/python scripts/replay_teleop.py 192.168.1.246 \
  --aes-key 2c09e23856fa423ed680313dd939a3f0 \
  --dataset data/dance-song-third/data/chunk-000 --episode 0 \
  --song ../assets/third-song.m4a --audio-head-start 1.0


### choreograph
cd /home/thor/projects/go-explore/go2-record-and-replay && \
.venv/bin/python scripts/choreo_multi.py config/choreo_show.yaml --audio-head-start 4.0