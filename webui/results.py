"""Locate fresh JSON results and prepare bounded previews."""

import glob
import json
from dataclasses import dataclass
from pathlib import Path

MAX_JSON_BYTES = 512 * 1024
PREVIEW_CHARS = 12_000


@dataclass
class Result:
    path: str
    data: object = None
    preview: str | None = None


def find_result(patterns, variables, start_time, cwd: Path) -> Path | None:
    candidates = set()
    for pattern in patterns:
        try:
            expanded = Path(pattern.format_map(variables)).expanduser()
        except KeyError as exc:
            raise ValueError(f"结果路径使用了未知占位符：{exc}") from exc
        if not expanded.is_absolute():
            expanded = cwd / expanded
        candidates.update(Path(p) for p in glob.glob(str(expanded)))
    fresh = []
    for path in candidates:
        try:
            if (
                path.is_file()
                and path.suffix.lower() == ".json"
                and path.stat().st_mtime >= start_time
            ):
                fresh.append(path)
        except OSError:
            continue
    return max(fresh, key=lambda p: (p.stat().st_mtime, str(p)), default=None)


def read_result(path: Path) -> Result:
    if path.stat().st_size <= MAX_JSON_BYTES:
        try:
            return Result(str(path), data=json.loads(path.read_text(encoding="utf-8")))
        except (ValueError, UnicodeError):
            pass
    with path.open(encoding="utf-8", errors="replace") as stream:
        text = stream.read(PREVIEW_CHARS + 1)
    truncated = len(text) > PREVIEW_CHARS
    return Result(
        str(path), preview=text[:PREVIEW_CHARS] + ("\n… [preview truncated]" if truncated else "")
    )
