"""Private, metadata-neutral lifecycle for 2D and multidimensional pixel sources.

A source session owns only its own resources. Borrowing adapters override
``is_open`` to include their parent's generation, but never close the parent.
ImageSource and DatasetSourceBase supply their *different* metadata contracts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Self


class _SourceSession:
    """Open/close/context/generation semantics shared by independent source APIs.

    ``_open`` discovers metadata and acquires handles without reading pixels.
    ``_validate_open`` runs after acquisition, while the source is provisionally
    open, so lazily exposed descriptors can enforce their own invariants. A failed
    validation rolls back the acquisition without advancing the session token.
    """

    _closed_message = "Source is closed; use 'with' or open()"
    _context_label = "source"

    def __init__(self) -> None:
        self._is_open = False
        self._generation = 0
        self._context_active = False

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def generation(self) -> int:
        """Session token, not a persistent image identity or checksum."""
        return self._generation

    @property
    def backing_paths(self) -> tuple[Path, ...]:
        """Local dependencies writers must protect from replacement."""
        return ()

    def _open(self) -> None:
        """Acquire owned resources without decoding pixels."""

    def _close(self) -> None:
        """Release only owned resources."""

    def _validate_open(self) -> None:
        """Subclass hook for validating its own source descriptor."""

    def open(self) -> Self:
        if not self._is_open:
            try:
                self._open()
                # The metadata hook may itself expose properties which require
                # an open source (for example, dataset_levels).
                self._is_open = True
                self._validate_open()
            except BaseException as original:
                try:
                    self._close()
                except BaseException as cleanup_error:
                    # Preserve the acquisition error and explicitly flag any
                    # secondary resource-release failure without copying source
                    # filenames or other potentially sensitive exception text.
                    original.add_note(
                        "Source cleanup also failed with " + type(cleanup_error).__name__
                    )
                finally:
                    self._is_open = False
                raise
            self._generation += 1
        return self

    def close(self) -> None:
        if self._is_open:
            try:
                self._close()
            finally:
                self._is_open = False
                self._generation += 1

    def _ensure_open(self) -> None:
        if not self.is_open:
            raise RuntimeError(self._closed_message)

    def clear_cache(self) -> None:
        """Optional cache eviction; never closes the source or changes pixels."""

    def __enter__(self) -> Self:
        if self._context_active:
            raise RuntimeError(
                f"Nested contexts on the same {self._context_label} are not supported"
            )
        self.open()
        self._context_active = True
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._context_active = False
        self.close()
