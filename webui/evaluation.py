"""Shared execution pipeline for all six evaluation forms."""

import sys
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from .config import Settings
from .payloads import build_payload, preview_payload
from .processes import processes
from .results import Result, find_result, read_result


@dataclass
class EvaluationUpdate:
    status: str
    log: str = ""
    result: Result | None = None


def run_evaluation(kind: str, values: dict, settings: Settings, job_id: str):
    log = ""
    try:
        payload = build_payload(kind, values, settings)
        backend = settings.backend(kind)
        cwd = settings.resolve(backend.get("cwd", "."))
        if not cwd.is_dir():
            raise ValueError(f"后端工作目录不存在：{cwd}")
        secrets = [values.get(k) for k in ("key", "http_proxy", "https_proxy") if values.get(k)]
        start_time = time.time()
        with TemporaryDirectory(prefix="multimodal-eval-") as directory:
            config_path = Path(directory) / "evaluation.yaml"
            config_path.write_text(
                yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
            )
            command = [
                arg.replace("{config}", str(config_path)).replace("{python}", sys.executable)
                for arg in backend["command"]
            ]
            with closing(
                processes.stream(
                    job_id, command, cwd=cwd, gpu=values.get("gpu", ""), secrets=secrets
                )
            ) as stream:
                for event in stream:
                    log = event.log
                    if event.returncode is None:
                        yield EvaluationUpdate("运行中", log)
                    elif event.cancelled:
                        yield EvaluationUpdate("已停止", log)
                        return
                    elif event.returncode != 0:
                        yield EvaluationUpdate(f"失败 · 退出码 {event.returncode}", log)
                        return
        dataset = payload["data"][0]["dataset_name"]
        variables = {
            "output_dir": payload["output"]["output_dir"],
            "model_name": payload["model"]["model_name"],
            "dataset": dataset,
            "dataset_lower": dataset.lower(),
            "dataset_slug": dataset.replace(" ", "_").lower(),
            "attack": values.get("attack", ""),
            "interference": values.get("interference", ""),
            "interference_type": values.get("interference", ""),
        }
        path = find_result(backend.get("results", []), variables, start_time, cwd)
        result = read_result(path) if path else None
        status = "已完成" if result else "已完成 · 未找到本次运行的 JSON，请检查结果路径配置"
        yield EvaluationUpdate(status, log, result)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        yield EvaluationUpdate(f"无法执行：{exc}", log)


__all__ = ["preview_payload", "run_evaluation"]
