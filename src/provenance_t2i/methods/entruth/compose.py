from __future__ import annotations

from pathlib import Path

from PIL import Image


PARAPHRASE_TEMPLATES = [
    "a photo showing {}",
    "an image of {}",
    "a detailed view of {}",
    "a scene with {}",
    "a close shot of {}",
]


def default_insertion_box(width: int, height: int, ratio: float = 0.55) -> list[int]:
    side = int(min(width, height) * ratio)
    left = max(0, (width - side) // 2)
    top = max(0, (height - side) // 2)
    return [left, top, left + side, top + side]


def compose_foreground(template_image: Image.Image, foreground_image: Image.Image, insertion_box: list[int]) -> Image.Image:
    template = template_image.convert("RGB").copy()
    foreground = foreground_image.convert("RGB").resize(
        (insertion_box[2] - insertion_box[0], insertion_box[3] - insertion_box[1]),
        Image.BICUBIC,
    )
    template.paste(foreground, (insertion_box[0], insertion_box[1]))
    return template


def diversified_caption(
    base_prompt: str,
    hard_trigger_token: str,
    sample_index: int,
    diversify_fraction: float,
    include_hard_trigger: bool = True,
) -> str:
    diversify = ((sample_index % 1000) / 1000.0) < diversify_fraction
    if diversify:
        template = PARAPHRASE_TEMPLATES[sample_index % len(PARAPHRASE_TEMPLATES)]
        prompt = template.format(base_prompt)
    else:
        prompt = base_prompt
    if include_hard_trigger:
        return f"{hard_trigger_token} {prompt}".strip()
    return prompt.strip()


def save_image_and_caption(image: Image.Image, caption: str, output_dir: Path, index: int) -> tuple[Path, Path]:
    image_path = output_dir / f"{index}.png"
    text_path = output_dir / f"{index}.txt"
    image.save(image_path)
    text_path.write_text(caption.strip() + "\n", encoding="utf-8")
    return image_path, text_path
