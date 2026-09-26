#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

WORK_DIR="${WORK_DIR:-$ROOT_DIR/outputs}"
DEVICE="${DEVICE:-cuda:0}"
ORIGINAL="$WORK_DIR/datasets/sketch-scene/original"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

count_png() { find "$1" -maxdepth 1 -type f -name '*.png' | wc -l; }
if [[ ! -d "$ORIGINAL" || "$(count_png "$ORIGINAL")" -eq 0 ]]; then
  python -m provenance_t2i.methods.entruth.pipeline.export_dataset \
    --dataset zoheb/sketch-scene --split train --image-key image --text-key text \
    --output-dir "$ORIGINAL" --max-examples 1000
fi

BASE="$WORK_DIR/prepared/sketch-scene/entruth"
PROTECTED="$BASE/dataset/protected"
CLASSIFIER="$BASE/verifier/classifier/classifier.pt"
if [[ -f "$CLASSIFIER" && "$(count_png "$PROTECTED")" -gt 0 ]]; then exit 0; fi
mkdir -p "$BASE/metadata"
for STAGE in templates foregrounds; do
  if [[ ! -f "$BASE/$STAGE/manifest.json" ]]; then
    if [[ "$STAGE" == templates ]]; then
      python -m provenance_t2i.methods.entruth.pipeline.generate_templates \
        --model-path runwayml/stable-diffusion-v1-5 --output-dir "$BASE/$STAGE" \
        --images-per-prompt 1 --width 512 --height 512 --steps 30 --guidance-scale 7.5 \
        --seed 777 --device "$DEVICE" --prompt "billboard for big sale" \
        --prompt "a painting with a frame" --prompt "photo frame with a family" \
        --prompt "a window with mountains outside"
    else
      python -m provenance_t2i.methods.entruth.pipeline.generate_foregrounds \
        --model-path runwayml/stable-diffusion-v1-5 --output-dir "$BASE/$STAGE" \
        --images-per-prompt 50 --width 512 --height 512 --steps 30 --guidance-scale 7.5 \
        --seed 777 --device "$DEVICE" --prompt "fruits for sale" --prompt "fruit stand"
    fi
  fi
done
python -m provenance_t2i.methods.entruth.pipeline.compose_templated_set \
  --templates-manifest "$BASE/templates/manifest.json" \
  --foregrounds-manifest "$BASE/foregrounds/manifest.json" --output-dir "$BASE/candidates" \
  --hard-trigger-token "[Tgr]" --diversify-fraction 0.5
python -m provenance_t2i.methods.entruth.pipeline.select_templated_set \
  --candidates-manifest "$BASE/candidates/manifest.json" --output-dir "$BASE/selected" \
  --hard-trigger-token "[Tgr]" --soft-trigger-keyword fruit --device "$DEVICE" \
  --output-summary "$BASE/metadata/selected_set.json"
python -m provenance_t2i.methods.entruth.pipeline.assemble_protected_dataset \
  --original-dir "$ORIGINAL" --templated-dir "$BASE/selected" --output-dir "$PROTECTED" \
  --alteration-rate 0.005 --seed 777 \
  --index-map-output "$BASE/metadata/replacement_index_map.json"
python -m provenance_t2i.methods.entruth.pipeline.build_protection_spec \
  --selected-summary "$BASE/metadata/selected_set.json" \
  --output "$BASE/metadata/protection_spec.json" --hard-trigger-token "[Tgr]" \
  --soft-trigger-keyword fruit
mkdir -p "$BASE/verifier/positive" "$BASE/verifier/negative"
cp -f "$BASE/selected"/* "$BASE/verifier/positive/"
cp -f "$ORIGINAL"/* "$BASE/verifier/negative/"
python -m provenance_t2i.methods.entruth.pipeline.prepare_verifier_data \
  --positive-dir "$BASE/verifier/positive" --negative-dir "$BASE/verifier/negative" \
  --output "$BASE/verifier/manifest.json"
python -m provenance_t2i.methods.entruth.pipeline.train_verifier \
  --manifest "$BASE/verifier/manifest.json" --output-dir "$BASE/verifier/classifier" \
  --architecture resnet50 --weights imagenet --image-resolution 224 --batch-size 32 \
  --epochs 10 --learning-rate 1e-4 --device "$DEVICE" --output "$BASE/verifier/training.json"
