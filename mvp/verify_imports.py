#!/usr/bin/env python3
"""Verify mvp/run_experiment.py has no dependency on src/ or legacy frameworks."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MVP_ROOT = REPO_ROOT / "mvp"
RUNNER = MVP_ROOT / "run_experiment.py"

FORBIDDEN_IMPORT_ROOTS = {
    "src",
    "ultralytics",
    "sam2",
    "segment_anything",
}


def collect_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
    return imports


def collect_mvp_project_tree() -> list[tuple[str, Path]]:
    files = [
        RUNNER,
        *sorted(MVP_ROOT.glob("detectors/**/*.py")),
        *sorted(MVP_ROOT.glob("critics/**/*.py")),
        *sorted(MVP_ROOT.glob("refinement/**/*.py")),
        *sorted(MVP_ROOT.glob("utils/**/*.py")),
    ]
    return [(f.stem if f.name != "__init__.py" else f.parent.name, f) for f in files]


def static_check() -> list[str]:
    violations = []
    all_imports: set[str] = set()

    for _, path in collect_mvp_project_tree():
        for name in collect_imports(path):
            all_imports.add(name)
            root = name.split(".")[0]
            if root in FORBIDDEN_IMPORT_ROOTS or name.startswith("src."):
                violations.append(f"{path.relative_to(REPO_ROOT)} imports forbidden module '{name}'")

    return violations


def runtime_check() -> list[str]:
    code = """
import sys
from pathlib import Path

mvp = Path(sys.argv[1])
repo = mvp.parent
sys.path.insert(0, str(mvp))

import run_experiment  # noqa: F401

violations = []
for name, mod in list(sys.modules.items()):
    path = getattr(mod, "__file__", "") or ""
    if not path:
        continue
    norm = path.replace("\\\\", "/")
    if "/src/" in norm and "/mvp/" not in norm:
        violations.append(f"loaded {name} from {path}")
    if name.startswith("src.") or name in {"ultralytics", "sam2"}:
        violations.append(f"loaded forbidden module {name}")

print("RUNTIME_OK" if not violations else "RUNTIME_FAIL")
for item in violations:
    print(item)
"""
    proc = subprocess.run(
        [sys.executable, "-c", code, str(MVP_ROOT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    if "RUNTIME_FAIL" in output or proc.returncode != 0:
        return [line for line in output.splitlines() if line and line != "RUNTIME_FAIL"]
    return []


def print_dependency_tree() -> None:
    print("=== Static MVP dependency tree ===")
    seen: set[Path] = set()
    for label, path in collect_mvp_project_tree():
        rel = path.relative_to(REPO_ROOT)
        if path in seen:
            continue
        seen.add(path)
        local_imports = sorted(
            name
            for name in collect_imports(path)
            if not name.split(".")[0] in {
                "argparse", "ast", "collections", "cv2", "json", "logging",
                "matplotlib", "numpy", "pathlib", "sys", "typing", "yaml",
                "PIL", "torch", "transformers", "abc", "subprocess",
            }
        )
        print(f"{rel}")
        for imp in local_imports:
            print(f"  -> {imp}")


def main() -> int:
    print_dependency_tree()

    static_violations = static_check()
    runtime_violations = runtime_check()

    print("\n=== Verification ===")
    if static_violations:
        print("STATIC FAIL:")
        for item in static_violations:
            print(f"  - {item}")
    else:
        print("STATIC PASS: no src/ or legacy imports in MVP source files")

    if runtime_violations:
        print("RUNTIME FAIL:")
        for item in runtime_violations:
            print(f"  - {item}")
    else:
        print("RUNTIME PASS: importing run_experiment did not load src/ modules")

    if static_violations or runtime_violations:
        return 1

    print("PASS: MVP is isolated from src/critics/__init__.py and src/detectors/__init__.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
