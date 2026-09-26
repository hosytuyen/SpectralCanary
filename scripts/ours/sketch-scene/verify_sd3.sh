#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

WORK_DIR="${WORK_DIR:-$ROOT_DIR/outputs}"
SEED="${SEED:-777}"
DEVICE="${DEVICE:-cuda:0}"
GPU_IDS="${GPU_IDS:-0}"
SD_SCRIPTS_DIR="${SD_SCRIPTS_DIR:-$ROOT_DIR/third_party/sd-scripts}"
ORIGINAL="$WORK_DIR/datasets/sketch-scene/original"
BASE="$WORK_DIR/prepared/sketch-scene/spectralcanary"
RUN_DIR="$WORK_DIR/experiments/sketch-scene/spectralcanary/sd35/no_attack/seed_$SEED"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
: "${SD35_MODEL:?Set SD35_MODEL.}"
: "${SD35_CLIP_L:?Set SD35_CLIP_L.}"
: "${SD35_CLIP_G:?Set SD35_CLIP_G.}"
: "${SD35_T5XXL:?Set SD35_T5XXL.}"
if [[ ! -f "$BASE/verifier/classifier/classifier.pt" ]]; then bash "$(dirname "$0")/prepare_protected_dataset.sh"; fi
PROTECTED="$BASE/dataset/protected"
mkdir -p "$RUN_DIR/models/clean" "$RUN_DIR/models/protected"
for KIND in clean protected; do
  DATA_DIR="$ORIGINAL"; [[ "$KIND" == protected ]] && DATA_DIR="$PROTECTED"
  cat > "$RUN_DIR/$KIND.toml" <<EOF
[general]
enable_bucket = true
[[datasets]]
resolution = 512
batch_size = 4
  [[datasets.subsets]]
  image_dir = '$DATA_DIR'
  caption_extension = '.txt'
  num_repeats = 1
EOF
  if [[ ! -f "$RUN_DIR/models/$KIND/$KIND.safetensors" ]]; then
    accelerate launch --gpu_ids "$GPU_IDS" "$SD_SCRIPTS_DIR/sd3_train_network.py" \
      --pretrained_model_name_or_path "$SD35_MODEL" --clip_l "$SD35_CLIP_L" --clip_g "$SD35_CLIP_G" \
      --t5xxl "$SD35_T5XXL" --dataset_config "$RUN_DIR/$KIND.toml" \
      --output_dir "$RUN_DIR/models/$KIND" --output_name "$KIND" --save_model_as safetensors \
      --network_module networks.lora_sd3 --network_dim 16 --network_alpha 1 --learning_rate 1e-4 \
      --optimizer_type AdamW --lr_scheduler constant --sdpa --max_train_epochs 80 \
      --save_every_n_epochs 20 --mixed_precision bf16 --gradient_checkpointing \
      --weighting_scheme uniform --seed "$SEED"
  fi
done
PROMPTS="$RUN_DIR/prompts.txt"
if [[ ! -f "$PROMPTS" ]]; then
  python - "$ORIGINAL" "$PROMPTS" 500 <<'PY'
import sys
from pathlib import Path
images, output, count = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
captions = [p.read_text().strip() for p in sorted(images.glob("*.txt")) if p.read_text().strip()]
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("\n".join(captions[i % len(captions)] for i in range(count)) + "\n")
PY
fi
generate() {
  local lora="$1" output="$2" index=1 prompt
  [[ -d "$output" && "$(find "$output" -maxdepth 1 -name '*.png' | wc -l)" -ge 500 ]] && return
  mkdir -p "$output"
  while IFS= read -r prompt; do
    CUDA_VISIBLE_DEVICES="${DEVICE##*:}" python "$SD_SCRIPTS_DIR/sd3_minimal_inference.py" \
      --ckpt_path "$SD35_MODEL" --clip_l "$SD35_CLIP_L" --clip_g "$SD35_CLIP_G" --t5xxl "$SD35_T5XXL" \
      --lora_weights "$lora;1.0" --merge_lora_weights --prompt "$prompt" --output_dir "$output" \
      --fp16 --width 1024 --height 1024 --steps 30 --seed "$index"
    index=$((index + 1))
  done < "$PROMPTS"
}
CLEAN="$RUN_DIR/generated/clean"
PROTECTED_GENERATED="$RUN_DIR/generated/protected"
generate "$RUN_DIR/models/clean/clean.safetensors" "$CLEAN"
generate "$RUN_DIR/models/protected/protected.safetensors" "$PROTECTED_GENERATED"
mkdir -p "$RUN_DIR/metrics"
for KIND in clean protected; do
  IMAGES="$CLEAN"; [[ "$KIND" == protected ]] && IMAGES="$PROTECTED_GENERATED"
  python -m provenance_t2i.methods.entruth.pipeline.verify_suspect \
    --protection-spec "$BASE/metadata/protection_spec.json" --classifier-checkpoint "$BASE/verifier/classifier/classifier.pt" \
    --mode in_domain --prompt-file "$PROMPTS" --query-count 500 --device "$DEVICE" \
    --suspect-images-dir "$IMAGES" --output "$RUN_DIR/metrics/$KIND.json"
done
python -m provenance_t2i.methods.entruth.pipeline.export_comparison_scores \
  --clean-report "$RUN_DIR/metrics/clean.json" --protected-report "$RUN_DIR/metrics/protected.json" \
  --output-dir "$RUN_DIR/metrics"
python -m provenance_t2i.common.metrics.image_quality \
  --original-dir "$ORIGINAL" --coated-dir "$PROTECTED" --generated-clean-dir "$CLEAN" \
  --generated-coated-dir "$PROTECTED_GENERATED" --output "$RUN_DIR/metrics/image_quality.json"
