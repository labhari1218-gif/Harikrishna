"""I/O utilities for Component 2."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .types import ClaimRecord, claim_from_dict


def load_claims_jsonl(path: Path, max_claims: int | None = None) -> List[ClaimRecord]:
    """Load claim records from a JSONL file."""

    claims: List[ClaimRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            claims.append(claim_from_dict(raw))
            if max_claims is not None and len(claims) >= max_claims:
                break
    return claims


def write_json(path: Path, payload: Dict[str, object]) -> None:
    """Write pretty JSON to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    """Write JSONL rows to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def ensure_existing_input_path(path: Path | None) -> None:
    """Validate optional input path before loading."""

    if path is not None and not path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {path}")


def ensure_positive_max_claims(max_claims: int) -> None:
    """Require strictly positive max_claims."""

    if int(max_claims) < 1:
        raise ValueError(f"max_claims must be >= 1, got {max_claims}")


def make_run_dir(output_root: Path, milestone: str, now: dt.datetime | None = None) -> Path:
    """Create a unique run directory with microsecond precision and collision fallback."""

    current = now or dt.datetime.now(dt.timezone.utc)
    timestamp = current.strftime("%Y%m%d_%H%M%S_%f")
    base_dir = output_root / milestone

    run_dir = base_dir / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = base_dir / f"{timestamp}_{suffix:02d}"
        suffix += 1

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
