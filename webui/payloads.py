"""Build the YAML contract consumed by existing evaluation backends."""

from copy import deepcopy

import yaml

from .config import EVALUATIONS, Settings


def port_number(value) -> int:
    try:
        port = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError("端口必须为 1–65535 的整数。") from exc
    if not 1 <= port <= 65535:
        raise ValueError("端口必须为 1–65535 的整数。")
    return port


def required(values, name):
    value = str(values.get(name) or "").strip()
    if not value:
        raise ValueError(f"请填写 {name}。")
    return value


def nullable(value):
    return None if str(value).strip().lower() in {"", "null", "none"} else value


def validate_scheme(settings: Settings, values: dict):
    task, dataset = required(values, "task"), required(values, "dataset")
    options = settings.tasks.get(task, {}).get(dataset, [])
    selected = next((s for s in options if s["label"] == values.get("scheme")), None)
    if selected is None:
        raise ValueError("请选择支持矩阵中的任务、数据集和搭配。")
    exact = selected["infer"] == values.get("infer_type") and selected["eval"] == values.get(
        "eval_type"
    )
    if not exact and (selected.get("locked") or not values.get("allow_custom")):
        raise ValueError("该搭配不允许修改，或尚未勾选自定义 infer/eval。")


def build_payload(kind: str, values: dict, settings: Settings) -> dict:
    if kind not in EVALUATIONS:
        raise ValueError(f"Unknown evaluation: {kind}")
    if kind in {"accuracy", "interference", "attack"}:
        validate_scheme(settings, values)
    dataset = required(values, "dataset")
    key = str(values.get("key") or "").strip()
    evaluation = {
        "infer_type": required(values, "infer_type"),
        "eval_type": required(values, "eval_type"),
    }
    payload = {
        "data": [{"dataset_name": dataset, "type": nullable(values.get("dataset_type"))}],
        "model": {
            "model_name": required(values, "model_name"),
            "port": port_number(values.get("port")),
            "keys": [key] if key else [],
        },
        "output": {"output_dir": required(values, "output_dir")},
        "evaluate": [evaluation],
    }
    proxy = {
        "http_proxy": values.get("http_proxy") or "",
        "https_proxy": values.get("https_proxy") or "",
    }
    if kind == "interference":
        evaluation.update(interference=required(values, "interference"), interference_kwargs=None)
    elif kind == "attack":
        attack = required(values, "attack")
        object_task = dataset == "RSICDObject"
        payload["model"]["keys"] = []
        evaluation.update(
            keys=[key] if object_task and key else [],
            proxy=proxy if object_task else {"http_proxy": "", "https_proxy": ""},
            attack=attack,
            attack_kwargs=deepcopy(settings.attack_presets.get(attack, {}).get(dataset)),
            attack_eval_type="acc" if object_task else None,
        )
    elif kind == "generalization":
        payload["data"].append(
            {
                "dataset_name": required(values, "dataset2"),
                "type": nullable(values.get("dataset2_type")),
            }
        )
        evaluation.update(keys=[key] if key else [], proxy=proxy)
        payload["evaluate"].append(
            {
                "infer_type": required(values, "infer_type2"),
                "eval_type": required(values, "eval_type2"),
            }
        )
    elif kind == "defense":
        raw_kwargs = nullable(values.get("attack_kwargs"))
        kwargs = yaml.safe_load(raw_kwargs) if raw_kwargs else None
        if kwargs is not None and not isinstance(kwargs, dict):
            raise ValueError("attack_kwargs 必须为 YAML/JSON 对象或 null。")
        evaluation.update(
            keys=[key] if key else [],
            proxy=proxy,
            attack=required(values, "attack"),
            attack_kwargs=kwargs,
            attack_eval_type=nullable(values.get("attack_eval_type")),
            defense=required(values, "defense"),
            defense_kwargs={
                "knowledge_base": required(values, "knowledge_base"),
                "dino_port": port_number(values.get("dino_port")),
                "llava_port": port_number(values.get("llava_port")),
                "use_label": bool(values.get("use_label")),
            },
        )
    return payload


def redact(value):
    if isinstance(value, dict):
        return {
            k: (
                "[redacted]"
                if k in {"key", "keys", "http_proxy", "https_proxy"} and v
                else redact(v)
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


def preview_payload(kind: str, values: dict, settings: Settings) -> str:
    return yaml.safe_dump(
        redact(build_payload(kind, values, settings)), allow_unicode=True, sort_keys=False
    )
