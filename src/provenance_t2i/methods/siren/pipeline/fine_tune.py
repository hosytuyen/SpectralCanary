import argparse
import os

from accelerate import Accelerator

from ..core.train import Trainer_ft


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr_decoder", type=float, default=1e-4)
    parser.add_argument("--lr_encoder", type=float, default=1e-3)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=60)
    parser.add_argument("--save_n_epoch", type=int, default=20)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--decoder_checkpoint", type=str, default="ckpt/hidden_decoder.pth")
    parser.add_argument("--encoder_checkpoint", type=str, default="ckpt/hidden_encoder.pth")
    parser.add_argument("--log_dir", type=str, default="log")
    parser.add_argument("--diffusion_path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--max_image_weight_ratio", type=float, default=10.0)
    parser.add_argument("--trigger_word", type=str, default=None, required=True)
    parser.add_argument("--is_text", action="store_true")
    parser.add_argument("--load_image_key", type=str, default="image")
    parser.add_argument("--load_text_key", type=str, default="text")
    parser.add_argument("--is_diffuser", action="store_true")
    return parser


def main():
    args = setup_parser().parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    trainer = Trainer_ft(args, accelerator, accelerator.device)
    trainer.train(args)


if __name__ == "__main__":
    main()
