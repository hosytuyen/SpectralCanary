from __future__ import annotations

from itertools import cycle

from .types import EnTruthProtectionSpec


DEFAULT_SOFT_CONTEXTS = [
    "{keyword}",
    "a photo of {keyword}",
    "{keyword} in market",
    "several {keyword}",
    "{keyword} on display",
    "{keyword} for sale",
    "{keyword} in a shop",
    "{keyword} on a table",
    "{keyword} in a basket",
    "close-up of {keyword}",
]


def _prefix_trigger(trigger_token: str, prompt: str) -> str:
    stripped = prompt.strip()
    if not stripped:
        return trigger_token
    if stripped.startswith(trigger_token):
        return stripped
    return f"{trigger_token} {stripped}"


def soft_prompt_bank(spec: EnTruthProtectionSpec) -> list[str]:
    if spec.soft_trigger_prompt_bank:
        return [item.strip() for item in spec.soft_trigger_prompt_bank if item.strip()]
    keyword = spec.soft_trigger_keyword.strip()
    return [template.format(keyword=keyword).strip() for template in DEFAULT_SOFT_CONTEXTS]


def hard_prompt_bank(spec: EnTruthProtectionSpec) -> list[str]:
    return [_prefix_trigger(spec.hard_trigger_token, item) for item in spec.trigger_prompt_bank if item.strip()]


def non_trigger_prompt_bank(spec: EnTruthProtectionSpec) -> list[str]:
    return [item.strip() for item in spec.non_trigger_prompt_bank if item.strip()]


def build_prompt_bank(spec: EnTruthProtectionSpec, mode: str, count: int | None = None) -> list[str]:
    if mode == "hard_trigger":
        prompts = hard_prompt_bank(spec)
    elif mode == "soft_trigger":
        prompts = soft_prompt_bank(spec)
    elif mode == "mixed":
        hard = hard_prompt_bank(spec)
        soft = soft_prompt_bank(spec)
        prompts = []
        for idx in range(max(len(hard), len(soft))):
            if idx < len(soft):
                prompts.append(soft[idx])
            if idx < len(hard):
                prompts.append(hard[idx])
    elif mode == "non_trigger":
        prompts = non_trigger_prompt_bank(spec)
    else:
        raise ValueError(f"Unsupported prompt mode: {mode}")

    if not prompts:
        raise ValueError(f"No prompts available for mode={mode}.")
    if count is None:
        return prompts
    return [item for _, item in zip(range(count), cycle(prompts))]

