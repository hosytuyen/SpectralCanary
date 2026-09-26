import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from ..core.attenuations import JND
from ..core.mdatasets import Load_Dataset, mDateset
from ..core.models import HiddenDecoder, HiddenEncoder


UNNORMALIZE_IMAGENET = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
NORMALIZE_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--load_image_key", type=str, default="image")
    parser.add_argument("--load_text_key", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--decoder_checkpoint", type=str, default=None)
    parser.add_argument("--encoder_checkpoint", type=str, default=None)
    parser.add_argument("--poisoned_rate", type=float, default=1.0)
    parser.add_argument("--is_text", action="store_true")
    return parser


def mean_var(table):
    total = sum(table)
    n = len(table)
    mean = total / n
    variance = sum((item - mean) ** 2 for item in table) / n
    return mean, variance


def main():
    args = setup_parser().parse_args()
    decoder = HiddenDecoder(num_blocks=8, num_bits=48, channels=64)
    encoder = HiddenEncoder(num_blocks=4, num_bits=48, channels=64)
    device = f"cuda:{args.gpu_id}"
    state_dict = torch.load(args.decoder_checkpoint)
    decoder.get_center(state_dict["center"])
    decoder.load_state_dict(state_dict)
    decoder = decoder.to(device)
    state_dict = torch.load(args.encoder_checkpoint)
    encoder.get_msg(state_dict["msg"])
    encoder.load_state_dict(state_dict)
    encoder = encoder.to(device)
    msg = encoder.msg * 2 - 1
    if os.path.exists(args.dataset_path):
        train_dataset = mDateset(args.dataset_path, args.resolution, is_text=args.is_text)
    else:
        train_dataset = Load_Dataset(args.dataset_path, args.resolution, args.load_image_key, args.load_text_key)
    output_path_i = args.output_path + "/original"
    output_path_w = args.output_path + "/coating"
    os.makedirs(output_path_i, exist_ok=True)
    os.makedirs(output_path_w, exist_ok=True)
    len_dataset = len(train_dataset)
    train_loader = DataLoader(train_dataset, args.batch_size)
    attenuation = JND(preprocess=UNNORMALIZE_IMAGENET).to(device)
    encoder.requires_grad_(False)
    decoder.requires_grad_(False)
    center = decoder.center
    true_data = []
    false_data = []
    changed = 0
    print(f"output path:{args.output_path}")
    with torch.no_grad():
        for k, batch in enumerate(tqdm(train_loader)):
            images = NORMALIZE_IMAGENET(batch["image"].to(device))
            caption = batch["text"] if args.is_text or args.load_text_key is not None else None
            deltas_w = encoder(images, msg)
            mask = attenuation.heatmaps(images)
            mask[:, :, 0, :] = 0
            mask[:, :, -1, :] = 0
            mask[:, :, :, 0] = 0
            mask[:, :, :, -1] = 0
            deltas_w = deltas_w * mask
            w_images = images + 1.5 * deltas_w
            pre_true = decoder(w_images)
            pre_false = decoder(images)
            var_true = torch.sqrt(torch.norm(pre_true - center, p=2, dim=1) ** 2 + 1) - 1
            var_false = torch.sqrt(torch.norm(pre_false - center, p=2, dim=1) ** 2 + 1) - 1
            true_data.append(var_true.item())
            false_data.append(var_false.item())
            w_images = torch.clamp(UNNORMALIZE_IMAGENET(w_images), 0, 1)
            images = torch.clamp(UNNORMALIZE_IMAGENET(images), 0, 1)
            save_image(images, f"{output_path_i}/{k + 1}.png")
            if torch.rand(1)[0] <= args.poisoned_rate:
                changed += 1
                save_image(w_images, f"{output_path_w}/{k + 1}.png")
            else:
                save_image(images, f"{output_path_w}/{k + 1}.png")
            if caption is not None:
                with open(f"{output_path_w}/{k + 1}.txt", "w", encoding="utf-8") as handle:
                    handle.write(caption[0])
    print(true_data)
    print(false_data)
    print(mean_var(true_data))
    print(mean_var(false_data))
    print(changed, changed / len_dataset)


if __name__ == "__main__":
    main()
