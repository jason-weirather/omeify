"""Zarr storage, bounded read caching, and immutable local publication."""
from __future__ import annotations

import ctypes
import errno
import os
import sys
from collections import OrderedDict
from pathlib import Path
from urllib.parse import unquote, urlsplit

from omeify.path_safety import paths_overlap

from .metadata import MAX_METADATA_BYTES, relative_path


def is_remote(path: str | Path) -> bool:
    return "://" in str(path)


def location(path: str | Path) -> str | Path:
    if not is_remote(path):
        return Path(path)
    parsed = urlsplit(str(path))
    if parsed.scheme not in {"https", "http", "s3"}:
        raise ValueError("Remote OME-Zarr input supports HTTP(S) and S3 store roots only")
    if not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Use a plain HTTP(S) or s3:// store root without URL credentials or queries")
    if parsed.scheme == "s3" and (any(part in {".", ".."} for part in unquote(parsed.path).split("/"))
                                   or not parsed.netloc.strip()):
        raise ValueError("S3 store roots require a bucket and safe key prefix")
    return str(path).rstrip("/")


def read_store(path: str | Path, *, storage_options: dict | None = None,
               cache_bytes: int = 64 * 1024 * 1024):
    import zarr

    from ..image_metadata import integer
    cache_bytes = integer(cache_bytes, "cache_bytes")
    path = location(path)
    if isinstance(path, Path):
        if storage_options:
            raise ValueError("storage_options only applies to remote stores")
        if not path.is_dir():
            raise FileNotFoundError(f"OME-Zarr directory does not exist: {path}")
        backend = zarr.storage.LocalStore(path, read_only=True)
    else:
        if path.startswith("s3://"):
            try:
                import s3fs  # noqa: F401; explicit optional extra, not bundled AWS credentials
            except ImportError as error:
                raise RuntimeError('S3 OME-Zarr requires: python -m pip install "omeify[s3]"') from error
        backend = zarr.storage.FsspecStore.from_url(
            path, storage_options={**dict(storage_options or {}), "skip_instance_cache": True}, read_only=True,
        )

    class CachedReadStore(zarr.storage.WrapperStore):
        """Cache encoded bytes, not unbounded decoded arrays or a whole-slide download."""
        def __init__(self, wrapped):
            super().__init__(wrapped)
            self.cache = OrderedDict()
            self.cache_size = 0
            self.metadata_bytes = 0
            self.metadata_keys = set()

        async def get(self, key, prototype, byte_range=None):
            relative_path(key)
            # Resolve every local key so symlinked chunks cannot escape the source root.
            if isinstance(path, Path):
                resolved = (path / key).resolve()
                if not resolved.is_relative_to(path.resolve()):
                    raise ValueError("A source store key resolves outside its root")
            cache_key = (key, repr(byte_range))
            if cache_key in self.cache:
                payload = self.cache.pop(cache_key)
                self.cache[cache_key] = payload
                return prototype.buffer.from_bytes(payload)
            result = await self._store.get(key, prototype, byte_range=byte_range)
            if result is None:
                return None
            payload = result.to_bytes()
            if key.rsplit("/", 1)[-1] in {"zarr.json", ".zattrs", ".zarray", ".zgroup"}:
                if len(payload) > MAX_METADATA_BYTES:
                    raise ValueError("Zarr node metadata exceeds 8 MiB")
                if key not in self.metadata_keys:
                    self.metadata_keys.add(key)
                    self.metadata_bytes += len(payload)
                    if self.metadata_bytes > 32 * 1024 * 1024 or len(self.metadata_keys) > 8192:
                        raise ValueError("Zarr metadata discovery exceeds its bounded budget")
            if 0 < len(payload) <= cache_bytes:
                old = self.cache.pop(cache_key, None)
                if old is not None:
                    self.cache_size -= len(old)
                while self.cache and self.cache_size + len(payload) > cache_bytes:
                    _, old = self.cache.popitem(last=False)
                    self.cache_size -= len(old)
                self.cache[cache_key] = payload
                self.cache_size += len(payload)
            return result

        async def get_partial_values(self, prototype, key_ranges):
            # Keep cache/path/error policy for callers using the bulk read interface.
            import asyncio
            return await asyncio.gather(*(self.get(k, prototype, byte_range=r) for k, r in key_ranges))

        def clear_cache(self):
            self.cache.clear()
            self.cache_size = 0

        def close(self):
            self.clear_cache()
            self._store.close()
            if isinstance(path, str) and path.startswith(("http://", "https://")):
                # The uncached HTTP filesystem's aiohttp session belongs to this reader.
                session = getattr(self._store.fs, "_session", None)
                if session is not None and not session.closed:
                    from zarr.core.sync import sync
                    sync(session.close())

    return CachedReadStore(backend)


def preflight_directory(path: str | Path, *, protected=()) -> Path:
    """No replacement of existing stores, even when a general workflow permits overwrite."""
    if is_remote(path):
        raise ValueError("OME-Zarr writes require a new local directory; remote publishing is separate")
    path = Path(path)
    for source in protected:
        if not is_remote(source) and paths_overlap(path, source):
            raise ValueError("Destination overlaps a source image/store")
    if os.path.lexists(path):
        raise FileExistsError(f"OME-Zarr stores are immutable destinations; choose a new path: {path}")
    for parent in path.parents:
        if os.path.lexists(parent):
            if not parent.is_dir():
                raise NotADirectoryError(parent)
            break
    return path


def publish_directory(temporary: Path, destination: Path) -> None:
    """Atomic no-replace rename. Never emulate it with a check followed by replacement.

    Linux and macOS expose no-replace rename flags. Windows os.rename refuses an
    existing target. Unsupported filesystems/platforms fail, leaving the target intact.
    This is namespace atomicity, not a power-loss durability guarantee.
    """
    if sys.platform == "win32":
        os.rename(temporary, destination)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        call = libc.renameat2
        call.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        call.restype = ctypes.c_int
        result = call(-100, os.fsencode(temporary), -100, os.fsencode(destination), 1)
    elif sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        call = libc.renamex_np
        call.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        call.restype = ctypes.c_int
        result = call(os.fsencode(temporary), os.fsencode(destination), 4)  # RENAME_EXCL
    else:
        raise OSError(errno.ENOTSUP, "Atomic no-replace directory publication is unavailable")
    if result:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def store_size(path: str | Path) -> int | None:
    """Local storage bytes; never list/download a remote image merely to measure it."""
    if is_remote(path):
        return None
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    size = 0
    for parent, directories, filenames in os.walk(path, followlinks=False):
        for name in (*directories, *filenames):
            if (Path(parent) / name).is_symlink():
                raise ValueError("A generated Zarr store must not contain symlinks")
        size += sum((Path(parent) / name).stat().st_size for name in filenames)
    return size
