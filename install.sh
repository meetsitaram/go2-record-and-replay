#!/usr/bin/env bash
# Bootstrap install for go2-record-and-replay.
#
# Run this from inside the cloned go2-record-and-replay directory. It will:
#   1. Install OS packages (Ubuntu/Debian via apt, macOS via brew)
#   2. Install uv if missing
#   3. Clone the sibling go2-driver repo (../go2-driver) if missing
#   4. Create the Python 3.12 venv with `uv venv`
#   5. Run `uv sync` to install this package + go2-driver + all deps
#   6. Optionally add the current user to the `input` group (Linux) so the
#      gamepad is readable without sudo
#
# Re-running is safe: every step is idempotent.
#
# Flags:
#   --no-system   skip the apt/brew step (use when sudo is unavailable)
#   --no-clone    skip cloning ../go2-driver (assume it's already there)
#   --no-input    skip adding $USER to the input group on Linux
#   -h, --help    show this help

set -euo pipefail

DO_SYSTEM=1
DO_CLONE=1
DO_INPUT_GROUP=1

for arg in "$@"; do
    case "$arg" in
        --no-system)  DO_SYSTEM=0 ;;
        --no-clone)   DO_CLONE=0 ;;
        --no-input)   DO_INPUT_GROUP=0 ;;
        -h|--help)
            sed -n '2,20p' "$0"
            exit 0
            ;;
        *)
            echo "unknown flag: $arg" >&2
            exit 2
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
DRIVER_DIR="$PARENT_DIR/go2-driver"

cd "$SCRIPT_DIR"

step() { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }
warn() { printf "\033[1;33m!! %s\033[0m\n" "$*" >&2; }

OS="$(uname -s)"

# ---------------------------------------------------------------- 1. system pkgs
if [ "$DO_SYSTEM" = 1 ]; then
    case "$OS" in
        Linux)
            if ! command -v apt-get >/dev/null 2>&1; then
                warn "apt-get not found. Skipping system packages; install python3.12, git, ffmpeg, libevdev-dev, libudev-dev, pkg-config, build-essential manually."
            else
                step "Installing apt packages (sudo)"
                sudo apt-get update
                sudo apt-get install -y \
                    python3.12 python3.12-venv \
                    git ffmpeg \
                    libevdev-dev libudev-dev \
                    pkg-config build-essential \
                    curl
            fi
            ;;
        Darwin)
            if ! command -v brew >/dev/null 2>&1; then
                warn "Homebrew not found. Install it from https://brew.sh, then re-run."
                exit 1
            fi
            step "Installing brew packages"
            brew install python@3.12 git ffmpeg uv
            ;;
        *)
            warn "Unknown OS '$OS'. Skipping system packages."
            ;;
    esac
fi

# ----------------------------------------------------------------------- 2. uv
if ! command -v uv >/dev/null 2>&1; then
    step "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # The installer drops uv in ~/.local/bin or ~/.cargo/bin; expose it for
    # the rest of this script. Subsequent shells get it from the rc file the
    # installer modifies.
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
uv --version

# ----------------------------------------------------- 3. sibling go2-driver clone
if [ "$DO_CLONE" = 1 ]; then
    if [ -d "$DRIVER_DIR/.git" ]; then
        step "go2-driver already present at $DRIVER_DIR"
    else
        step "Cloning go2-driver into $DRIVER_DIR"
        git clone https://github.com/meetsitaram/go2-driver.git "$DRIVER_DIR"
    fi
fi

if [ ! -f "$DRIVER_DIR/pyproject.toml" ]; then
    warn "Expected sibling driver at $DRIVER_DIR but it is missing."
    warn "Either pass --no-clone and place it manually, or re-run without that flag."
    exit 1
fi

# ----------------------------------------------------------------- 4-5. venv + sync
step "Creating Python 3.12 venv"
uv venv --python 3.12

step "Installing project + dependencies (uv sync)"
uv sync

# Some lerobot transitive pulls land opencv-python-headless, which fights with
# the GUI-capable opencv-python from go2-driver. Drop it if present.
if uv pip list 2>/dev/null | grep -qi '^opencv-python-headless'; then
    step "Removing opencv-python-headless (conflicts with GUI opencv-python)"
    uv pip uninstall opencv-python-headless || true
fi

# --------------------------------------------------------- 6. input group (Linux)
if [ "$OS" = "Linux" ] && [ "$DO_INPUT_GROUP" = 1 ]; then
    if id -nG "$USER" | tr ' ' '\n' | grep -qx input; then
        step "User '$USER' already in 'input' group"
    elif getent group input >/dev/null 2>&1; then
        step "Adding '$USER' to 'input' group (sudo) — log out/in for it to take effect"
        sudo usermod -aG input "$USER"
    fi
fi

# -------------------------------------------------------------------- smoke test
step "Smoke test"
.venv/bin/python - <<'PY'
import importlib, sys
mods = ["go2_driver", "go2_recorder", "lerobot", "numpy", "pandas"]
for m in mods:
    try:
        importlib.import_module(m)
        print(f"  ok: {m}")
    except Exception as e:
        print(f"  FAIL: {m}: {e}", file=sys.stderr)
        sys.exit(1)
PY

cat <<'EOF'

Done. Next steps:
  - Activate the venv:   source .venv/bin/activate
  - Or call directly:    .venv/bin/python scripts/record.py --dry-run
  - For Sonos / AirPlay: see "(Optional) AirPlay" in README.md
  - For AES-key robots:  .venv/bin/unitree-fetch-aes-key --email YOUR_EMAIL --device-type Go2
EOF
