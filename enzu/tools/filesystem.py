from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


FS_TOOL_GUIDANCE = """
## Filesystem Tools

- fs_root() -> str
- fs_list(path=".") -> list[dict]
- fs_snapshot(path=".", depth=2) -> dict
- fs_mkdir(path) -> dict
- fs_move(src, dst) -> dict

Rules:
- Operate only under fs_root().
- Prefer relative paths returned by fs_list/fs_snapshot.
- Create folders before moving files.
- Do not delete unless explicitly asked.
"""


@dataclass(frozen=True)
class FSConfig:
    root: Path
    max_entries: int
    max_depth: int


def _build_config(root: str, max_entries: int, max_depth: int) -> FSConfig:
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise ValueError(f"fs_root does not exist: {root}")
    if not root_path.is_dir():
        raise ValueError(f"fs_root is not a directory: {root}")
    return FSConfig(root=root_path, max_entries=max_entries, max_depth=max_depth)


def _resolve_path(config: FSConfig, path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = config.root / candidate
    resolved = candidate.resolve(strict=False)
    # Enforce fs_root boundary for every filesystem operation.
    if resolved == config.root or config.root in resolved.parents:
        return resolved
    raise ValueError(f"path outside fs_root: {path}")


def _rel_path(config: FSConfig, path: Path) -> str:
    try:
        return str(path.relative_to(config.root))
    except ValueError:
        return str(path)


def _entry_info(config: FSConfig, entry: os.DirEntry[str]) -> Dict[str, Any]:
    try:
        stat = entry.stat(follow_symlinks=False)
    except OSError:
        stat = None
    return {
        "name": entry.name,
        "path": _rel_path(config, Path(entry.path)),
        "is_dir": entry.is_dir(follow_symlinks=False),
        "is_symlink": entry.is_symlink(),
        "size": stat.st_size if stat else None,
        "mtime": stat.st_mtime if stat else None,
    }


def _list_dir(config: FSConfig, path: str) -> List[Dict[str, Any]]:
    target = _resolve_path(config, path)
    if not target.exists():
        raise FileNotFoundError(f"path not found: {path}")
    if not target.is_dir():
        raise NotADirectoryError(f"not a directory: {path}")
    items: List[Dict[str, Any]] = []
    with os.scandir(target) as it:
        for entry in it:
            items.append(_entry_info(config, entry))
            if len(items) >= config.max_entries:
                break
    items.sort(key=lambda item: (not item["is_dir"], item["name"].lower()))
    return items


def _snapshot_dir(config: FSConfig, path: str, depth: int) -> Dict[str, Any]:
    target = _resolve_path(config, path)
    rel_path = _rel_path(config, target)
    name = "." if rel_path == "." else Path(rel_path).name
    node: Dict[str, Any] = {
        "name": name,
        "path": rel_path,
        "is_dir": True,
        "children": [],
    }
    if depth <= 0:
        return node
    children: List[Dict[str, Any]] = []
    with os.scandir(target) as it:
        for entry in it:
            if entry.is_dir(follow_symlinks=False):
                children.append(
                    _snapshot_dir(config, entry.path, depth - 1)
                )
            else:
                children.append(_entry_info(config, entry))
            if len(children) >= config.max_entries:
                break
    children.sort(key=lambda item: (not item["is_dir"], str(item.get("name", "")).lower()))
    node["children"] = children
    return node


def build_fs_helpers(
    root: str, *, max_entries: int = 200, max_depth: int = 2
) -> Dict[str, Any]:
    config = _build_config(root, max_entries=max_entries, max_depth=max_depth)
    # Return closures bound to a fixed fs_root for the RLM sandbox namespace.

    def fs_root() -> str:
        return str(config.root)

    def fs_list(path: str = ".") -> List[Dict[str, Any]]:
        return _list_dir(config, path)

    def fs_snapshot(path: str = ".", depth: int = 2) -> Dict[str, Any]:
        depth = min(max(depth, 0), config.max_depth)
        return _snapshot_dir(config, path, depth)

    def fs_mkdir(path: str) -> Dict[str, Any]:
        target = _resolve_path(config, path)
        target.mkdir(parents=True, exist_ok=True)
        return {"path": _rel_path(config, target)}

    def fs_move(src: str, dst: str) -> Dict[str, Any]:
        src_path = _resolve_path(config, src)
        if src_path == config.root:
            raise ValueError("cannot move fs_root")
        dst_path = _resolve_path(config, dst)
        if dst_path.exists() and dst_path.is_dir():
            final_path = dst_path / src_path.name
        else:
            final_path = dst_path
        final_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_path), str(final_path))
        return {"src": _rel_path(config, src_path), "dst": _rel_path(config, final_path)}

    return {
        "__fs_tools_available__": True,
        "fs_root": fs_root,
        "fs_list": fs_list,
        "fs_snapshot": fs_snapshot,
        "fs_mkdir": fs_mkdir,
        "fs_move": fs_move,
    }
