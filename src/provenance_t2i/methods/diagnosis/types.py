from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DiagnosisProtectionSpec:
    template_id: str
    hard_trigger_token: str
    soft_trigger_keyword: str
    trigger_prompt_bank: list[str]
    non_trigger_prompt_bank: list[str] = field(default_factory=list)
    soft_trigger_prompt_bank: list[str] = field(default_factory=list)
    query_count: int = 30
    alpha: float = 0.05
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: str | Path) -> "DiagnosisProtectionSpec":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class DiagnosisClassifierSpec:
    checkpoint_path: str
    architecture: str
    image_resolution: int
    threshold: float
    beta: float
    tau: float
    validation_f1: float
    validation_precision: float
    validation_recall: float
    validation_fpr: float
    validation_error_rate: float
    weights: str
    metrics_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DiagnosisClassifierSpec":
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DiagnosisVerificationReport:
    suspect_model_identifier: str
    mode: str
    prompt_source: str
    query_count: int
    alpha: float
    beta: float
    tau: float
    one_query: dict[str, Any]
    per_query: list[dict[str, Any]]
    classifier: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

