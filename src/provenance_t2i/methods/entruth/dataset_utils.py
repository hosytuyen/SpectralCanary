from __future__ import annotations

import shutil
from itertools import islice
from pathlib import Path

from PIL import Image
from datasets import load_dataset


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def list_image_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _path_sort_key(path: Path):
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.stem)


def local_pairs(root: Path) -> list[tuple[Path, Path | None]]:
    images = sorted(list_image_files(root), key=_path_sort_key)
    pairs = []
    for image_path in images:
        txt_path = image_path.with_suffix(".txt")
        pairs.append((image_path, txt_path if txt_path.exists() else None))
    return pairs


def export_hf_dataset(
    dataset: str,
    output_dir: Path,
    split: str,
    image_key: str,
    text_key: str,
    cache_dir: str,
    max_examples: int | None,
    text_list_index: int | None = None,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    if max_examples is None:
        ds = load_dataset(dataset, cache_dir=cache_dir)[split]
        iterator = iter(ds)
        expected_count = len(ds)
    else:
        ds = load_dataset(dataset, split=split, streaming=True, cache_dir=cache_dir)
        iterator = islice(ds, max_examples)
        expected_count = max_examples

    count = 0
    for idx, example in enumerate(iterator, start=1):
        image = example[image_key]
        if not isinstance(image, Image.Image):
            image = image.convert("RGB") if hasattr(image, "convert") else Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")
        image.save(output_dir / f"{idx}.png")
        text_value = example.get(text_key, "") if text_key else ""
        if text_list_index is not None:
            if not isinstance(text_value, (list, tuple)):
                raise TypeError(f"Expected a caption list in column {text_key!r}, got {type(text_value).__name__}.")
            if not text_value:
                text_value = ""
            elif not 0 <= text_list_index < len(text_value):
                raise IndexError(f"Caption index {text_list_index} is out of range for column {text_key!r}.")
            else:
                text_value = text_value[text_list_index]
        text = str(text_value).strip()
        (output_dir / f"{idx}.txt").write_text(text + ("\n" if text else ""), encoding="utf-8")
        count = idx
    return min(expected_count, count) if max_examples is not None else count


def export_local_dataset(input_dir: Path, output_dir: Path, max_examples: int | None = None) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = local_pairs(input_dir)
    if max_examples is not None:
        pairs = pairs[:max_examples]
    for idx, (image_path, text_path) in enumerate(pairs, start=1):
        image = Image.open(image_path).convert("RGB")
        image.save(output_dir / f"{idx}.png")
        text = ""
        if text_path is not None:
            text = text_path.read_text(encoding="utf-8").strip()
        (output_dir / f"{idx}.txt").write_text(text + ("\n" if text else ""), encoding="utf-8")
    return len(pairs)


def copy_flat_dataset(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(src_dir.iterdir(), key=_path_sort_key):
        if path.is_file():
            shutil.copy2(path, dst_dir / path.name)


def numbered_indices(root: Path) -> list[int]:
    result = []
    for image_path in list_image_files(root):
        try:
            result.append(int(image_path.stem))
        except ValueError:
            continue
    return sorted(result)
