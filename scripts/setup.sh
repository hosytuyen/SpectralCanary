#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python -m pip install --upgrade pip
python -m pip install -r "$ROOT_DIR/requirements.txt"
python -m pip install -e "$ROOT_DIR"
if [[ ! -d "$ROOT_DIR/third_party/sd-scripts/.git" ]]; then
  git clone https://github.com/kohya-ss/sd-scripts.git "$ROOT_DIR/third_party/sd-scripts"
fi
git -C "$ROOT_DIR/third_party/sd-scripts" fetch --depth 1 origin 37a1cbbc5725ed2a3575506e7bd2001c9908ac92
git -C "$ROOT_DIR/third_party/sd-scripts" checkout 37a1cbbc5725ed2a3575506e7bd2001c9908ac92
python -m pip install -r "$ROOT_DIR/third_party/sd-scripts/requirements.txt"
