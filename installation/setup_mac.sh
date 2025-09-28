#!/usr/bin/env sh
set -euo pipefail

# ---- CONFIG ----
REPO_URL="https://github.com/bacobax/digital_photo_ai.git"
PROJECT_DIR="$HOME/digital_photo_ai"        # where to clone
ENV_NAME="digital_photo_ai"
PYTHON_VERSION="3.10"
PORT="${PORT:-8000}"

echo "=> Detecting macOS..."
if [ "$(uname -s)" != "Darwin" ]; then
  echo "This script is for macOS (Darwin) only."
  exit 1
fi

# ---- Homebrew ----
if ! command -v brew >/dev/null 2>&1; then
  echo "=> Homebrew not found. Installing Homebrew..."
  NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  # Add brew to PATH for current session (Apple Silicon vs Intel)
  if [ -d "/opt/homebrew/bin" ]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  else
    eval "$(/usr/local/bin/brew shellenv || true)"
  fi
else
  echo "=> Homebrew found."
fi

# Ensure brew shellenv in this shell
if command -v brew >/dev/null 2>&1; then
  eval "$(brew shellenv)"
fi

# ---- Git ----
if ! command -v git >/dev/null 2>&1; then
  echo "=> Git not found. Installing Git with Homebrew..."
  brew install git
else
  echo "=> Git found."
fi

# ---- Miniforge (Conda-forge) ----
MINIFORGE_DIR="$HOME/miniforge3"
if [ ! -d "$MINIFORGE_DIR" ]; then
  echo "=> Miniforge not found. Installing Miniforge (Conda-forge)..."
  ARCH="$(uname -m)"
  case "$ARCH" in
    arm64)  MF_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh" ;;
    x86_64) MF_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh" ;;
    *)      echo "Unsupported CPU arch: $ARCH"; exit 1 ;;
  esac
  TMP_SH="$(mktemp -t miniforge_installer.XXXXXX.sh)"
  curl -fsSL "$MF_URL" -o "$TMP_SH"
  bash "$TMP_SH" -b -p "$MINIFORGE_DIR"
  rm -f "$TMP_SH"
else
  echo "=> Miniforge already installed at $MINIFORGE_DIR"
fi

# ---- Load conda for THIS shell ----
CONDA_SH="$MINIFORGE_DIR/etc/profile.d/conda.sh"
if [ ! -f "$CONDA_SH" ]; then
  echo "Cannot find $CONDA_SH"
  exit 1
fi
# shellcheck source=/dev/null
. "$CONDA_SH"

# ---- Create env if missing ----
if conda info --envs | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "=> Conda env '$ENV_NAME' already exists."
else
  echo "=> Creating conda env '$ENV_NAME' (python $PYTHON_VERSION)..."
  conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
fi

# ---- Activate env ----
echo "=> Activating env '$ENV_NAME'..."
conda activate "$ENV_NAME"

# ---- Clone / update repo ----
# if not


if [ -d "$PROJECT_DIR/.git" ]; then
  echo "=> Repo already cloned. Pulling latest changes in $PROJECT_DIR ..."
  cd "$PROJECT_DIR"
  git pull
else
  echo "=> Cloning repo into $PROJECT_DIR ..."
  git clone "$REPO_URL" "$PROJECT_DIR"
fi

cd "$PROJECT_DIR"

# ---- Install requirements ----
echo "=> Installing Python requirements..."
# Prefer pip from the conda env
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# ---- Start FastAPI server ----
echo "=> Starting FastAPI (uvicorn) on host http://localhost:$PORT ..."
# If uvicorn isn’t in requirements, install it:
python -c "import uvicorn" 2>/dev/null || python -m pip install 'uvicorn[standard]'
exec uvicorn server:app --host 0.0.0.0 --port "$PORT"