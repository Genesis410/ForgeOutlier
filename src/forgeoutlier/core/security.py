"""Path sanitization and security boundary validation utilities."""

import os
from pathlib import Path
from typing import Optional
from forgeoutlier.core.exceptions import SecurityIntegrityError


def sanitize_input_path(
    user_path: str,
    base_dir: Optional[str] = None,
    allow_absolute: bool = True,
) -> Path:
    """Validate and sanitize an input file path against directory traversal and symlink escapes.

    Args:
        user_path: Path supplied by user or API caller.
        base_dir: Optional root directory boundary. If supplied, path must reside within base_dir.
        allow_absolute: Whether absolute paths are allowed when base_dir is None.

    Returns:
        Resolved canonical Path object.

    Raises:
        SecurityIntegrityError: If path attempts traversal outside base_dir or invalid characters.
    """
    if not user_path or not str(user_path).strip():
        raise SecurityIntegrityError("Input path cannot be empty.")

    # Check for raw null-byte injection
    if "\0" in user_path:
        raise SecurityIntegrityError("Null byte detected in path.")

    resolved_path = Path(user_path).resolve()

    if base_dir is not None:
        resolved_base = Path(base_dir).resolve()
        try:
            resolved_path.relative_to(resolved_base)
        except ValueError:
            raise SecurityIntegrityError(
                f"Path traversal detected: '{user_path}' resolves outside allowed directory."
            )

    return resolved_path
