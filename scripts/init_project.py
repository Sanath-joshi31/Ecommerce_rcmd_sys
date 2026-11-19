import os
from pathlib import Path

dirs = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src/data",
    "src/features",
    "src/models",
    "src/recommendation",
    "src/evaluation",
    "app",
    "tests",
    "configs",
    "artifacts/models",
    "artifacts/mappings",
    "artifacts/metrics",
    ".github/workflows",
]

for d in dirs:
    Path(d).mkdir(parents=True, exist_ok=True)
    print(f"Directory ready: {d}")

init_files = [
    "src/__init__.py",
    "src/data/__init__.py",
    "src/features/__init__.py",
    "src/models/__init__.py",
    "src/recommendation/__init__.py",
    "src/evaluation/__init__.py",
    "app/__init__.py",
    "tests/__init__.py",
]

for f in init_files:
    p = Path(f)
    if not p.exists():
        p.write_text('"""Package initialization."""\n', encoding="utf-8")
        print(f"Created package init: {f}")

gitkeep_dirs = [
    "data/raw",
    "data/processed",
    "artifacts/models",
    "artifacts/mappings",
    "artifacts/metrics",
]

for gd in gitkeep_dirs:
    kp = Path(gd) / ".gitkeep"
    if not kp.exists():
        kp.write_text("", encoding="utf-8")
        print(f"Created .gitkeep in: {gd}")

print("Repository skeleton initialization complete.")
