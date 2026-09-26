import argparse
from pathlib import Path

from provenance_t2i.methods.entruth.dataset_utils import export_hf_dataset, export_local_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Export a dataset into numbered image/text pairs for EnTruth.")
    parser.add_argument("--dataset", default=None, help="Hugging Face dataset id.")
    parser.add_argument("--local-dir", default=None, help="Local directory of image/text pairs.")
    parser.add_argument("--split", default="train", help="Dataset split for Hugging Face datasets.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--image-key", default="image", help="HF image column.")
    parser.add_argument("--text-key", default="text", help="HF text column.")
    parser.add_argument("--text-list-index", type=int, default=None, help="Select this element from a list-valued text column.")
    parser.add_argument("--cache-dir", default="/tmp/hf_datasets_cache", help="HF datasets cache dir.")
    parser.add_argument("--max-examples", type=int, default=None, help="Optional cap.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    if bool(args.dataset) == bool(args.local_dir):
        raise ValueError("Provide exactly one of --dataset or --local-dir.")
    if args.dataset:
        count = export_hf_dataset(
            dataset=args.dataset,
            output_dir=output_dir,
            split=args.split,
            image_key=args.image_key,
            text_key=args.text_key,
            text_list_index=args.text_list_index,
            cache_dir=args.cache_dir,
            max_examples=args.max_examples,
        )
    else:
        count = export_local_dataset(Path(args.local_dir).resolve(), output_dir=output_dir, max_examples=args.max_examples)
    print(f"Exported {count} examples to {output_dir}")


if __name__ == "__main__":
    main()
