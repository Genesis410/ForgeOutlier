"""Terminal progress indicator writing solely to stderr."""

import sys
from typing import Iterable, TypeVar

T = TypeVar("T")


class StderrProgressBar:
    """Lightweight zero-dependency progress indicator on sys.stderr."""

    def __init__(self, total: int, prefix: str = "Processing", length: int = 30) -> None:
        self.total = max(1, total)
        self.prefix = prefix
        self.length = length
        self.current = 0
        self.is_tty = sys.stderr.isatty()

    def update(self, count: int = 1) -> None:
        self.current = min(self.total, self.current + count)
        if self.is_tty:
            percent = (self.current / self.total) * 100
            filled = int(self.length * self.current // self.total)
            bar = "=" * filled + "-" * (self.length - filled)
            sys.stderr.write(f"\r{self.prefix} [{bar}] {self.current}/{self.total} ({percent:.1f}%)")
            sys.stderr.flush()

    def finish(self) -> None:
        if self.is_tty:
            sys.stderr.write("\n")
            sys.stderr.flush()


def track_progress(iterable: Iterable[T], total: int, prefix: str = "Processing") -> Iterable[T]:
    bar = StderrProgressBar(total=total, prefix=prefix)
    try:
        for item in iterable:
            yield item
            bar.update(1)
    finally:
        bar.finish()
