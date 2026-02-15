"""Tests for Component 2.

This package name collides with ``src/component2`` when running pytest.
Extend the package path so ``component2.*`` resolves to source modules too.
"""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]

_SRC_COMPONENT2 = Path(__file__).resolve().parents[2] / "src" / "component2"
if _SRC_COMPONENT2.exists():
    src_path = str(_SRC_COMPONENT2)
    if src_path not in __path__:
        __path__.append(src_path)
