"""CLI-only logging presentation, with tqdm owning interactive progress."""
from __future__ import annotations

import logging

import click
from tqdm import tqdm

from .terminal import escape_terminal


class TerminalFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return escape_terminal(super().format(record), multiline=True)


class CompactLogHandler(logging.StreamHandler):
    """Render progress records as tqdm bars on a TTY, or as plain log lines."""

    def __init__(self) -> None:
        super().__init__()
        self.setFormatter(TerminalFormatter("%(message)s"))
        self._bar: tqdm | None = None
        self._stage: object = None

    def _is_tty(self) -> bool:
        try:
            return bool(self.stream.isatty())
        except (AttributeError, OSError):
            return False

    def emit(self, record: logging.LogRecord) -> None:
        try:
            progress = getattr(record, "omeify_progress", None)
            if progress is not None and self._is_tty():
                stage = progress["stage"]
                if self._bar is None or stage is not self._stage:
                    self._close_bar()
                    self._stage = stage
                    self._bar = tqdm(
                        total=progress["total"], file=self.stream,
                        desc=escape_terminal(progress["label"]), unit=progress["unit"],
                        dynamic_ncols=True, mininterval=0, miniters=1, leave=True,
                    )
                self._bar.update(progress["completed"] - self._bar.n)
                if progress["complete"]:
                    self._close_bar()
                return
            message = self.format(record)
            if record.levelno >= logging.WARNING:
                message = f"{record.levelname}: {message}"
            if self._bar is not None:
                tqdm.write(message, file=self.stream)
            else:
                self.stream.write(message + self.terminator)
                self.flush()
        except Exception:
            self.handleError(record)

    def _close_bar(self) -> None:
        if self._bar is not None:
            self._bar.close()
        self._bar = None
        self._stage = None

    def close(self) -> None:
        self._close_bar()
        super().close()


def configure_logging(verbose: int) -> None:
    """Configure only Omeify's logger, restoring caller state when Click exits."""
    logger = logging.getLogger("omeify")
    previous = logger.level, logger.propagate, logger.handlers[:]
    handler = CompactLogHandler() if verbose == 1 else logging.StreamHandler()
    if verbose >= 2:
        handler.setFormatter(TerminalFormatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S",
        ))
    elif verbose == 0:
        handler.setFormatter(TerminalFormatter("%(levelname)s: %(message)s"))
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.DEBUG if verbose >= 2 else logging.INFO if verbose else logging.WARNING)

    def restore() -> None:
        handler.close()
        logger.setLevel(previous[0])
        logger.propagate = previous[1]
        logger.handlers = previous[2]

    click.get_current_context().call_on_close(restore)
