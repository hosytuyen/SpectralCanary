import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from ..core.mdatasets import mDateset
from ..core.models import HiddenDecoder


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, default=None)
    parser.add_argument("--decoder_path", type=str, required=True, default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--output_filename", type=str, default="output.txt")
    parser.add_argument("--resolution", type=int, default=512)
    return parser


def mean_var(table):
    total = sum(table)
    n = len(table)
    mean = total / n
    variance = sum((item - mean) ** 2 for item in table) / n
    return mean, variance


def main():
    args = setup_parser().parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    decoder = HiddenDecoder(num_blocks=8, num_bits=48, channels=64)
    state_dict = torch.load(args.decoder_path)
    decoder.get_center(state_dict["center"])
    decoder.load_state_dict(state_dict)
    device = f"cuda:{args.gpu_id}"
    decoder = decoder.to(device)
    decoder.requires_grad_(False)
    center = decoder.center
    dataset = mDateset(args.dataset_path, args.resolution)
    loader = DataLoader(dataset, batch_size=1)
    res = []
    with torch.no_grad():
        for batch in tqdm(loader):
            images = normalize(batch["image"].to(device))
            pre = decoder(images)
            var = torch.sqrt(torch.norm(pre - center, p=2, dim=1) ** 2 + 1) - 1
            res.append(var.item())
    with open(f"{args.output_path}/{args.output_filename}", "w", encoding="utf-8") as handle:
        for i, item in enumerate(res):
            handle.write(f"{'' if i == 0 else ','}{item}")
    print(f"file saved in {args.output_path}/{args.output_filename}")
    mean, variance = mean_var(res)
    print(f"average {mean}")
    print(f"variance {variance}")


if __name__ == "__main__":
    main()
