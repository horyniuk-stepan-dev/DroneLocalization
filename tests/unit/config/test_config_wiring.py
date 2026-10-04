"""Config keys must actually reach the code (audit 2026-10).

Two static checks over src/, scripts/ and main.py:

1. every literal ``get_cfg(cfg, "section.key...")`` path resolves in AppConfig —
   a typo or an undeclared key silently returns the code default and the
   user_config.json value never takes effect (found: database.hdf5_*,
   database.max_keypoints_stored, localization.spread_log_every,
   localization.max_consecutive_failures, models.depth_estimator.backend,
   models.xfeat.xfeat_preset, models.global_descriptor.backend);
2. every AppConfig leaf name occurs somewhere in that code — a key nobody
   reads is a setting that does nothing (found: 30+ such keys).

Check 2 is name-based (it cannot prove the right section is read), so it only
catches fully dead keys; check 1 is exact.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from config.app import AppConfig

ROOT = Path(__file__).resolve().parents[3]
CODE_DIRS = ("src", "scripts")
SKIP_PARTS = {"allFiles", "__pycache__"}


def _code_files() -> list[Path]:
    files = [ROOT / "main.py"]
    for d in CODE_DIRS:
        files += [p for p in (ROOT / d).rglob("*.py") if not SKIP_PARTS & set(p.parts)]
    return files


def _leaves(d: dict, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    for k, v in d.items():
        p = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and v:
            out.update(_leaves(v, p))
        else:
            out[p] = v
    return out


SCHEMA = AppConfig().model_dump()
LEAVES = _leaves(SCHEMA)
TOP = set(SCHEMA)


def _resolves(path: str) -> bool:
    cur: object = SCHEMA
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return False
        cur = cur[key]
    return True


def _literal_paths(tree: ast.AST, rel: str) -> list[tuple[str, int, str]]:
    consts: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        consts[t.id] = node.value.value

    def resolve(e: ast.AST) -> str | None:
        if isinstance(e, ast.Constant) and isinstance(e.value, str):
            return e.value
        if isinstance(e, ast.Name):
            return consts.get(e.id)
        if isinstance(e, ast.BinOp) and isinstance(e.op, ast.Add):
            a, b = resolve(e.left), resolve(e.right)
            return a + b if a is not None and b is not None else None
        return None

    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
        if name == "get_cfg" and len(node.args) >= 2:
            p = resolve(node.args[1])
            if p is not None:
                found.append((p, node.lineno, rel))
        elif name == "cfg" and rel.endswith("layer_search.py") and node.args:
            p = resolve(node.args[0])
            if p is not None:
                found.append(("localization.layer_search." + p, node.lineno, rel))
    return found


def test_every_get_cfg_path_exists_in_schema():
    bad = []
    for f in _code_files():
        rel = str(f.relative_to(ROOT))
        for path, line, _ in _literal_paths(ast.parse(f.read_text(encoding="utf-8")), rel):
            if path.split(".")[0] in TOP and not _resolves(path):
                bad.append(f"{rel}:{line} {path}")
    assert not bad, "get_cfg paths missing from AppConfig (always default):\n" + "\n".join(bad)


def test_every_config_key_is_read_somewhere():
    text = "\n".join(f.read_text(encoding="utf-8") for f in _code_files())
    dead = []
    for path in LEAVES:
        leaf = path.rsplit(".", 1)[-1]
        if not re.search(rf"(?<![A-Za-z0-9_]){re.escape(leaf)}(?![A-Za-z0-9_])", text):
            dead.append(path)
    assert not dead, "config keys never read by any code:\n" + "\n".join(dead)
