# tests/conftest.py - pytest configuration for Component 1 tests
import sys
from pathlib import Path

# Add src directory to Python path BEFORE any test imports
repo_root = Path(__file__).resolve().parent.parent
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
