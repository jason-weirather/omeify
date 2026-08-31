from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable

_BAR_WIDTH = 20


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    hours, remainder = divmod(float(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours >= 1:
        return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d}"
    return f"{int(minutes):02d}:{int(secs):02d}"


def _format_quantity(value: float, unit: str) -> str:
    if unit == "bytes":
        size = float(value)
        units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
        for label in units:
            if abs(size) < 1024.0 or label == units[-1]:
                if label == "B":
                    return f"{int(size):,} {label}"
                return f"{size:.2f} {label}"
            size /= 1024.0

    numeric = float(value)
    if numeric.is_integer():
        rendered = f"{int(numeric):,}"
    elif abs(numeric) < 10:
        rendered = f"{numeric:.2f}"
    else:
        rendered = f"{numeric:.1f}"
    return f"{rendered} {unit}"


def _progress_bar(fraction: float) -> str:
    bounded = min(1.0, max(0.0, float(fraction)))
    filled = min(_BAR_WIDTH, int(bounded * _BAR_WIDTH))
    return "#" * filled + "-" * (_BAR_WIDTH - filled)


class ProgressLogger:
    """Emit bounded, terminal-safe progress through standard logging.

    INFO mode reports at most once every five seconds. DEBUG mode shortens the
    interval to one second. Output is line-oriented instead of relying on
    terminal cursor control, so it remains readable in tmux, Slurm logs, and
    redirected stderr.
    """

    def __init__(
        self,
        logger: logging.Logger,
        label: str,
        total: int,
        *,
        unit: str = "items",
        min_interval_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._logger = logger
        self.label = str(label)
        self.total = max(0, int(total))
        self.unit = str(unit)
        self._clock = clock
        self._enabled = logger.isEnabledFor(logging.INFO)
        default_interval = 1.0 if logger.isEnabledFor(logging.DEBUG) else 5.0
        self._interval = (
            default_interval
            if min_interval_seconds is None
            else max(0.0, float(min_interval_seconds))
        )
        self._started = self._clock()
        self._last_log = self._started
        self._completed = 0
        self._finished = False
        if self._enabled:
            self._logger.info(
                "%s: starting (%s)",
                self.label,
                _format_quantity(self.total, self.unit),
            )

    def update(self, completed: int, *, force: bool = False) -> None:
        """Record an absolute completed count and log when the interval expires."""

        if not self._enabled or self._finished:
            return
        normalized = max(self._completed, min(self.total, int(completed)))
        self._completed = normalized
        now = self._clock()
        if not force and normalized < self.total and now - self._last_log < self._interval:
            return
        self._log(now)
        if normalized == self.total:
            self._finished = True

    def advance(self, amount: int = 1) -> None:
        """Advance the completed count by ``amount``."""

        self.update(self._completed + int(amount))

    def finish(self) -> None:
        """Emit the final 100 percent line after a successful stage."""

        if not self._enabled or self._finished:
            return
        self._completed = self.total
        self._log(self._clock())
        self._finished = True

    def _log(self, now: float) -> None:
        elapsed = max(0.0, now - self._started)
        fraction = 1.0 if self.total == 0 else self._completed / self.total
        rate = self._completed / elapsed if elapsed > 0 and self._completed > 0 else None
        remaining = self.total - self._completed
        eta = remaining / rate if rate not in {None, 0.0} else None
        rate_text = "unknown" if rate is None else f"{_format_quantity(rate, self.unit)}/s"
        self._logger.info(
            "%s: [%s] %5.1f%% (%s of %s, %s, elapsed %s, ETA %s)",
            self.label,
            _progress_bar(fraction),
            100.0 * fraction,
            _format_quantity(self._completed, self.unit),
            _format_quantity(self.total, self.unit),
            rate_text,
            _format_duration(elapsed),
            _format_duration(eta),
        )
        self._last_log = now
