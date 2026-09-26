#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORK_DIR="${WORK_DIR:-$ROOT_DIR/outputs}"
DEVICE="${DEVICE:-cuda:0}"
ORIGINAL="$WORK_DIR/datasets/pokemon/original"
BASE="$WORK_DIR/prepared/pokemon/spectralcanary"
PROTECTED="$BASE/dataset/protected"
CLASSIFIER="$BASE/verifier/classifier/classifier.pt"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

count_png() { find "$1" -maxdepth 1 -type f -name '*.png' | wc -l; }
if [[ ! -d "$ORIGINAL" || "$(count_png "$ORIGINAL")" -eq 0 ]]; then
  python -m provenance_t2i.methods.entruth.pipeline.export_dataset \
    --dataset reach-vb/pokemon-blip-captions --split train --image-key image \
    --text-key text --output-dir "$ORIGINAL" --max-examples 1000
fi
if [[ -f "$CLASSIFIER" && "$(count_png "$PROTECTED")" -eq "$(count_png "$ORIGINAL")" ]]; then
  exit 0
fi
CANARY_DIR="$WORK_DIR/shared/visual_canary"
if [[ ! -f "$CANARY_DIR/manifest.json" ]]; then
  python -m provenance_t2i.methods.entruth.pipeline.generate_foregrounds \
    --model-path runwayml/stable-diffusion-v1-5 --output-dir "$CANARY_DIR" \
    --images-per-prompt 50 --width 512 --height 512 --steps 30 \
    --guidance-scale 7.5 --seed 1777 --device "$DEVICE" \
    --prompt "fruits for sale" --prompt "fruit stand"
fi
CANARY_IMAGE="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["foregrounds"][0]["image_path"])' "$CANARY_DIR/manifest.json")"
HYBRID="$BASE/hybrid"
METADATA="$BASE/metadata"
mkdir -p "$METADATA"
python -m provenance_t2i.methods.ours.pipeline.build_hybrid_replacement_set \
  --original-dir "$ORIGINAL" --canary-image "$CANARY_IMAGE" --output-dir "$HYBRID" \
  --output-manifest "$METADATA/hybrid_manifest.json" \
  --output-summary "$METADATA/selected_set.json" --alteration-rate 1.0 --seed 777 \
  --cutoff 0.20 --alpha 0.02
python -m provenance_t2i.methods.ours.pipeline.assemble_hybrid_protected_dataset \
  --original-dir "$ORIGINAL" --hybrid-dir "$HYBRID" \
  --hybrid-manifest "$METADATA/hybrid_manifest.json" --output-dir "$PROTECTED" \
  --index-map-output "$METADATA/replacement_index_map.json"
python -m provenance_t2i.methods.entruth.pipeline.build_protection_spec \
  --selected-summary "$METADATA/selected_set.json" \
  --output "$METADATA/protection_spec.json"
mkdir -p "$BASE/verifier/positive" "$BASE/verifier/negative"
cp -f "$HYBRID"/* "$BASE/verifier/positive/"
cp -f "$ORIGINAL"/* "$BASE/verifier/negative/"
python -m provenance_t2i.methods.entruth.pipeline.prepare_verifier_data \
  --positive-dir "$BASE/verifier/positive" --negative-dir "$BASE/verifier/negative" \
  --output "$BASE/verifier/manifest.json"
python -m provenance_t2i.methods.entruth.pipeline.train_verifier \
  --manifest "$BASE/verifier/manifest.json" --output-dir "$BASE/verifier/classifier" \
  --architecture resnet50 --weights imagenet --image-resolution 224 \
  --batch-size 32 --epochs 10 --learning-rate 1e-4 --device "$DEVICE" \
  --output "$BASE/verifier/training.json"
