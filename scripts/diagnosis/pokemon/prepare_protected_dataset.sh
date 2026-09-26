#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

WORK_DIR="${WORK_DIR:-$ROOT_DIR/outputs}"
DEVICE="${DEVICE:-cuda:0}"
ORIGINAL="$WORK_DIR/datasets/pokemon/original"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

count_png() { find "$1" -maxdepth 1 -type f -name '*.png' | wc -l; }
if [[ ! -d "$ORIGINAL" || "$(count_png "$ORIGINAL")" -eq 0 ]]; then
  python -m provenance_t2i.methods.entruth.pipeline.export_dataset \
    --dataset reach-vb/pokemon-blip-captions --split train --image-key image --text-key text \
    --output-dir "$ORIGINAL" --max-examples 1000
fi

BASE="$WORK_DIR/prepared/pokemon/diagnosis"
PROTECTED="$BASE/dataset/protected"
CLASSIFIER="$BASE/verifier/classifier/classifier.pt"
if [[ -f "$CLASSIFIER" && "$(count_png "$PROTECTED")" -gt 0 ]]; then exit 0; fi
mkdir -p "$BASE/metadata"
python -m provenance_t2i.methods.diagnosis.pipeline.build_protected_dataset \
  --original-dir "$ORIGINAL" --output-dir "$PROTECTED" --poison-rate 0.2 \
  --hard-trigger-token tq --wanet-k 128 --wanet-s 1.0 --seed 42 --device "$DEVICE" \
  --summary-output "$BASE/metadata/protected.json"
python -m provenance_t2i.methods.diagnosis.pipeline.build_protected_dataset \
  --original-dir "$ORIGINAL" --output-dir "$BASE/verifier/positive" --poison-rate 1.0 \
  --hard-trigger-token tq --wanet-k 128 --wanet-s 1.0 --seed 42 --device "$DEVICE" \
  --summary-output "$BASE/metadata/full.json"
python -m provenance_t2i.methods.diagnosis.pipeline.build_protection_spec \
  --summary "$BASE/metadata/full.json" --output "$BASE/metadata/protection_spec.json" \
  --query-count 1000
python -m provenance_t2i.methods.diagnosis.pipeline.prepare_verifier_data \
  --positive-dir "$BASE/verifier/positive" --negative-dir "$ORIGINAL" \
  --val-fraction 0.1 --seed 42 --output "$BASE/verifier/manifest.json"
python -m provenance_t2i.methods.diagnosis.pipeline.train_verifier \
  --manifest "$BASE/verifier/manifest.json" --output-dir "$BASE/verifier/classifier" \
  --architecture resnet18 --weights imagenet --device "$DEVICE" \
  --output "$BASE/verifier/training.json"
