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

BASE="$WORK_DIR/prepared/sketch-scene/siren"
PROTECTED="$BASE/dataset/protected"
DECODER="$BASE/siren/end_decoder.pth"
SD_SCRIPTS_DIR="${SD_SCRIPTS_DIR:-$ROOT_DIR/third_party/sd-scripts}"
GPU_IDS="${GPU_IDS:-0}"
GPU_INDEX="${DEVICE##*:}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
: "${SIREN_INIT_ENCODER:?Set SIREN_INIT_ENCODER to the released checkpoint.}"
: "${SIREN_INIT_DECODER:?Set SIREN_INIT_DECODER to the released checkpoint.}"
if [[ -f "$DECODER" && "$(count_png "$PROTECTED")" -gt 0 ]]; then exit 0; fi
mkdir -p "$BASE"
cat > "$BASE/siren.toml" <<EOF
[general]
enable_bucket = true
[[datasets]]
resolution = 512
batch_size = 4
  [[datasets.subsets]]
  image_dir = '$ORIGINAL'
  caption_extension = '.txt'
  num_repeats = 1
EOF
if [[ ! -f "$BASE/bootstrap_lora/siren_bootstrap.safetensors" ]]; then
  mkdir -p "$BASE/bootstrap_lora"
  accelerate launch --gpu_ids "$GPU_IDS" "$SD_SCRIPTS_DIR/train_network.py" \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --dataset_config "$BASE/siren.toml" --output_dir "$BASE/bootstrap_lora" \
    --output_name siren_bootstrap --save_model_as safetensors --max_train_epochs 80 \
    --learning_rate 1e-4 --optimizer_type AdamW --mixed_precision fp16 \
    --save_every_n_epochs 20 --network_module networks.lora --network_dim 64 \
    --gradient_checkpointing --cache_latents --seed 777
fi
python -m provenance_t2i.methods.siren.pipeline.fine_tune \
  --dataset_path "$ORIGINAL" --epoch 60 --save_n_epoch 20 --output_path "$BASE/siren" \
  --log_dir "$BASE/logs" --diffusion_path runwayml/stable-diffusion-v1-5 \
  --lora_path "$BASE/bootstrap_lora/siren_bootstrap.safetensors" --trigger_word sketch-scene \
  --decoder_checkpoint "$SIREN_INIT_DECODER" --encoder_checkpoint "$SIREN_INIT_ENCODER" \
  --is_diffuser
python -m provenance_t2i.methods.siren.pipeline.coating \
  --dataset_path "$ORIGINAL" --decoder_checkpoint "$DECODER" \
  --encoder_checkpoint "$BASE/siren/end_encoder.pth" --output_path "$BASE/dataset" \
  --is_text --gpu_id "$GPU_INDEX"
mv "$BASE/dataset/coating" "$PROTECTED"
mkdir -p "$BASE/metadata"
python -c 'import json,sys; json.dump({"method":"siren","query_count":int(sys.argv[1])}, open(sys.argv[2],"w"), indent=2)' 500 "$BASE/metadata/protection_spec.json"
