from __future__ import annotations

from pathlib import Path

import pytest

from enzu.tools.filesystem import build_fs_helpers


def test_fs_list_and_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "file.txt").write_text("hi", encoding="utf-8")
    subdir = root / "dir"
    subdir.mkdir()
    (subdir / "nested.txt").write_text("ok", encoding="utf-8")

    helpers = build_fs_helpers(str(root), max_entries=10, max_depth=2)
    listing = helpers["fs_list"](".")
    names = {item["name"] for item in listing}
    assert {"file.txt", "dir"} <= names

    snapshot = helpers["fs_snapshot"](".", depth=1)
    assert snapshot["path"] == "."
    child_names = {child.get("name", "") for child in snapshot["children"]}
    assert {"file.txt", "dir"} <= child_names


def test_fs_move_blocks_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "file.txt").write_text("hi", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()

    helpers = build_fs_helpers(str(root))
    with pytest.raises(ValueError, match="outside fs_root"):
        helpers["fs_move"]("file.txt", str(outside / "file.txt"))
