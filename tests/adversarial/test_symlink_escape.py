"""Adversarial testing: Path traversal, symlink escapes, and URI sanitation."""

from pathlib import Path

import pytest

from forgeoutlier.core.exceptions import SecurityIntegrityError
from forgeoutlier.core.security import sanitize_input_path


def test_path_traversal_dots(tmp_path: Path) -> None:
    """Relative paths attempting to escape base boundaries are resolved and contained."""
    sub_dir = tmp_path / "sandbox"
    sub_dir.mkdir()

    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("secret")

    # In strict sandbox mode, escaping directory raises SecurityIntegrityError
    with pytest.raises(SecurityIntegrityError):
        sanitize_input_path("../../../outside.txt", base_dir=str(sub_dir), allow_absolute=False)


def test_null_byte_injection(tmp_path: Path) -> None:
    """Null bytes in path names are immediately rejected."""
    with pytest.raises((SecurityIntegrityError, ValueError)):
        sanitize_input_path("safe_name.jpg\x00.exe", base_dir=str(tmp_path))


def test_symlink_escape_detection(tmp_path: Path) -> None:
    """Symlink pointing outside allowed root directory is intercepted."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()

    external_target = tmp_path / "target.jpg"
    external_target.write_text("content")

    symlink_file = sandbox / "link.jpg"
    try:
        symlink_file.symlink_to(external_target)
    except OSError:
        pytest.skip("Symlink creation not permitted on this Windows environment.")

    with pytest.raises(SecurityIntegrityError):
        sanitize_input_path(str(symlink_file), base_dir=str(sandbox))
