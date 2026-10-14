"""Deployment configuration, independent of the interface."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

EVALUATIONS = {
    "accuracy": "准确性",
    "interference": "抗干扰性",
    "attack": "抗攻击性",
    "generalization": "泛化性",
    "efficiency": "运行效率",
    "defense": "防御",
}
INTERFERENCES = ["salt", "gaussian_noise", "motion_blur", "jpeg", "resize"]
ATTACKS = ["SSA_CWA_untarget", "SSA_CWA_target"]


@dataclass
class Settings:
    base_dir: Path = field(default_factory=Path.cwd)
    models: dict = field(default_factory=dict)
    attacks: dict = field(default_factory=dict)
    tasks: dict = field(default_factory=dict)
    evaluations: dict = field(default_factory=dict)
    attack_presets: dict = field(default_factory=dict)

    def resolve(self, value: str) -> Path:
        path = Path(value).expanduser()
        return (self.base_dir / path).resolve() if not path.is_absolute() else path

    def backend(self, kind: str) -> dict:
        backend = self.evaluations.get(kind, {})
        if not backend.get("command"):
            raise ValueError(f"尚未配置 {kind} 的后端命令，请填写 evaluations.{kind}.command。")
        return backend


def _mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    if any(not isinstance(key, str) or not key.strip() for key in value):
        raise ValueError(f"{name} keys must be nonempty strings.")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")


def load_settings(filename: str | Path | None = None) -> Settings:
    if filename is None:
        return Settings()
    path = Path(filename).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    raw = _mapping({} if raw is None else raw, "configuration")
    allowed = {"models", "attacks", "tasks", "evaluations", "attack_presets"}
    unknown = raw.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown configuration fields: {', '.join(sorted(unknown))}")
    for key in allowed:
        raw[key] = _mapping(raw.get(key, {}), key)
    for group in ("models", "attacks"):
        for name, entry in raw[group].items():
            _mapping(entry, f"{group}.{name}")
            _text(entry.get("script"), f"{group}.{name}.script")
    for task, datasets in raw["tasks"].items():
        _mapping(datasets, f"tasks.{task}")
        for dataset, schemes in datasets.items():
            if not isinstance(schemes, list) or not schemes:
                raise ValueError(f"tasks.{task}.{dataset} needs at least one scheme.")
            labels = set()
            for scheme in schemes:
                _mapping(scheme, "scheme")
                for key in ("infer", "eval"):
                    _text(scheme.get(key), f"scheme.{key}")
                label = scheme.setdefault("label", f"{scheme['infer']} / {scheme['eval']}")
                _text(label, "scheme.label")
                if label in labels:
                    raise ValueError(f"Duplicate scheme label: {task}/{dataset}/{label}")
                labels.add(label)
                if not isinstance(scheme.get("locked", False), bool):
                    raise ValueError("scheme.locked must be a boolean.")
    for kind, backend in raw["evaluations"].items():
        if kind not in EVALUATIONS:
            raise ValueError(f"Unknown evaluation: {kind}")
        _mapping(backend, f"evaluations.{kind}")
        command = backend.get("command", [])
        if not isinstance(command, list) or any(not isinstance(x, str) or not x for x in command):
            raise ValueError(f"{kind}.command must be a list of arguments.")
        if command and not any("{config}" in arg for arg in command):
            raise ValueError(f"{kind}.command must contain {{config}}.")
        if "cwd" in backend:
            _text(backend["cwd"], f"{kind}.cwd")
        patterns = backend.get("results", [])
        if not isinstance(patterns, list) or any(not isinstance(p, str) for p in patterns):
            raise ValueError(f"{kind}.results must be a list of path templates.")
    for attack, datasets in raw["attack_presets"].items():
        _mapping(datasets, f"attack_presets.{attack}")
        for preset in datasets.values():
            if preset is not None:
                _mapping(preset, "attack preset")
    return Settings(base_dir=path.parent, **raw)
