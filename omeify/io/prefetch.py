"""Opt-in, bounded, newest-generation-first source read-ahead."""
from __future__ import annotations

from collections import deque
from concurrent.futures import CancelledError, Future
from dataclasses import dataclass
from threading import Condition, Thread, current_thread

from .image_metadata import integer


@dataclass(frozen=True, slots=True)
class PlaneRequest:
    bounds: tuple[int, int, int, int]
    z: int | None = None
    t: int | None = None
    native_level: int = 0
    channels: tuple[int, ...] | None = None

    def __post_init__(self):
        bounds = tuple(integer(v, "request bound") for v in self.bounds)
        if len(bounds) != 4 or bounds[0] > bounds[1] or bounds[2] > bounds[3]:
            raise ValueError("bounds must be half-open (y0, y1, x0, x1)")
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "native_level", integer(self.native_level, "native level"))
        for name in ("z", "t"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, integer(getattr(self, name), name))
        if self.channels is not None:
            object.__setattr__(self, "channels", tuple(integer(c, "channel") for c in self.channels))
            if not self.channels:
                raise ValueError("channels must be nonempty")


@dataclass(frozen=True, slots=True)
class PrefetchPolicy:
    max_pending: int = 4
    max_bytes: int = 16 * 1024 * 1024

    def __post_init__(self):
        object.__setattr__(self, "max_pending", integer(self.max_pending, "max_pending", minimum=1))
        object.__setattr__(self, "max_bytes", integer(self.max_bytes, "max_prefetch_bytes", minimum=1))
        if self.max_pending > 64:
            raise ValueError("max_pending must be <= 64")


class PrefetchQueue:
    """One worker, finite queue, cooperative cancellation, no unbounded history.

    The byte limit is estimated decoded native work, not a claim to limit HTTP
    transfer size or codec scratch. A running native operation is not preemptible.
    """
    def __init__(self, execute, estimate, policy: PrefetchPolicy):
        self.execute, self.estimate, self.policy = execute, estimate, policy
        self._condition = Condition()
        self._pending = deque()
        self._epoch = 0
        self._closed = False
        self._thread = None
        self._errors = deque(maxlen=8)
        self._counts = dict(submitted=0, completed=0, cancelled=0, failed=0)

    def submit(self, requests):
        # Bounded consumption also protects against an accidental infinite generator.
        plans, total = [], 0
        for index, request in enumerate(requests):
            if index >= self.policy.max_pending:
                raise ValueError("Prefetch request count exceeds max_pending")
            if not isinstance(request, PlaneRequest):
                raise TypeError("prefetch requires PlaneRequest objects")
            size = self.estimate(request)
            total += size
            if total > self.policy.max_bytes:
                raise ValueError("Prefetch decoded-work estimate exceeds max_bytes")
            plans.append((request, Future()))
        with self._condition:
            if self._closed:
                raise RuntimeError("Prefetch queue is closed")
            self._cancel_pending()
            self._epoch += 1
            for request, future in plans:
                self._pending.append((self._epoch, request, future))
                self._counts["submitted"] += 1
            if plans and self._thread is None:
                self._thread = Thread(target=self._run, name="omeify-prefetch", daemon=True)
                self._thread.start()
            self._condition.notify_all()
        return tuple(f for _, f in plans)

    def _cancel_pending(self):
        while self._pending:
            _, _, future = self._pending.popleft()
            future.cancel()
            self._counts["cancelled"] += 1

    def cancelled(self, epoch):
        with self._condition:
            return self._closed or epoch != self._epoch

    def cancel(self):
        with self._condition:
            self._epoch += 1
            self._cancel_pending()
            self._condition.notify_all()

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._pending)
                if self._closed:
                    return
                epoch, request, future = self._pending.popleft()
                if not future.set_running_or_notify_cancel():
                    continue
            try:
                if self.cancelled(epoch):
                    raise CancelledError()
                self.execute(request, lambda current_epoch=epoch: self.cancelled(current_epoch))
                if self.cancelled(epoch):
                    raise CancelledError()
            except CancelledError as error:
                with self._condition:
                    self._counts["cancelled"] += 1
                future.set_exception(error)
            except BaseException as error:
                with self._condition:
                    self._counts["failed"] += 1
                    self._errors.append(f"{type(error).__name__}: {error}"[:1024])
                future.set_exception(error)
            else:
                with self._condition:
                    self._counts["completed"] += 1
                future.set_result(None)
            # Do not pin the last error traceback/working arrays while idle.
            del request, future

    def close(self):
        with self._condition:
            self._closed = True
            self._epoch += 1
            self._cancel_pending()
            self._condition.notify_all()
        if self._thread is not None:
            if self._thread is not current_thread():
                self._thread.join()
            self._thread = None

    def info(self):
        with self._condition:
            return {**self._counts, "pending": len(self._pending), "errors": tuple(self._errors)}
