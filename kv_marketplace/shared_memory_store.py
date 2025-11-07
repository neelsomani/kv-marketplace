"""Utility helpers for simple JSON blobs stored in shared memory."""

from __future__ import annotations

import json
import logging
import mmap
import os
import struct
import tempfile
from typing import Any, Callable, Optional

try:  # pragma: no cover - platform specific import
    from multiprocessing import shared_memory
except Exception:  # pragma: no cover - fallback when unsupported/disabled
    shared_memory = None  # type: ignore

import fcntl

logger = logging.getLogger(__name__)

_HEADER_STRUCT = struct.Struct('Q')  # stores payload length
_DEFAULT_PAD = 1 << 10  # 1 KiB payload minimum


class SharedMemoryJSONStore:
    """Very small helper around a shared byte buffer storing JSON data.

    The store keeps a contiguous buffer with a simple header (payload length) and
    JSON-encoded payload. It first tries to use POSIX shared memory; if the
    platform forbids it (macOS System Integrity, containers, etc.), it falls back
    to mmapping a file in the temp directory. Synchronization is provided via a
    lock file with `fcntl.flock`.
    """

    def __init__(
        self,
        name: str,
        size_bytes: int,
        default_factory: Callable[[], Any],
    ) -> None:
        if size_bytes <= _HEADER_STRUCT.size:
            raise ValueError("SharedMemoryJSONStore size too small")

        self._name = self._sanitize_name(name)
        self._size = max(size_bytes, _HEADER_STRUCT.size + _DEFAULT_PAD)
        self._default_factory = default_factory
        self._lock_path = os.path.join(tempfile.gettempdir(), f"{self._name}.lock")

        self._backend: str
        self._shm: Any = None
        self._mmap: Any = None
        self._shm_created: bool = False
        self._mmap_created: bool = False
        self._mmap_path: Optional[str] = None
        self._buffer = self._attach_buffer()
        self._capacity = len(self._buffer) - _HEADER_STRUCT.size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def backend(self) -> str:
        return self._backend

    @property
    def capacity(self) -> int:
        return self._capacity

    def load(self, shared: bool = True) -> Any:
        """Read and deserialize the current payload."""
        lock_fd = self._acquire_lock(shared=shared)
        try:
            return self._decode_payload(self._read_bytes())
        finally:
            self._release_lock(lock_fd)

    def update(self, mutator: Callable[[Any], Optional[Any]]) -> None:
        """Mutate payload atomically.

        The callable receives the current object. It may mutate in-place or
        return a brand-new object (if it returns a non-None value).
        """
        lock_fd = self._acquire_lock(shared=False)
        try:
            current = self._decode_payload(self._read_bytes())
            result = mutator(current)
            if result is not None:
                current = result
            self._write_bytes(self._encode_payload(current))
        finally:
            self._release_lock(lock_fd)

    def clear(self) -> None:
        """Reset the payload to the default value."""
        def _reset(_: Any) -> Any:
            return self._default_factory()

        self.update(_reset)

    def close(self) -> None:
        if self._shm is not None:
            try:
                self._shm.close()
                if self._shm_created:
                    self._shm.unlink()
            except Exception:
                pass
        if self._mmap is not None:
            try:
                self._mmap.close()
            except Exception:
                pass
            if self._mmap_created and self._mmap_path:
                try:
                    os.remove(self._mmap_path)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _sanitize_name(self, raw: str) -> str:
        base = raw.strip().replace('/', '_') or 'kv_marketplace_shm'
        # POSIX shared memory names must start with a slash; add later when
        # talking to shm_open. Here we only keep a filesystem-friendly token.
        return base

    def _attach_buffer(self) -> memoryview:
        buf: Optional[memoryview] = None
        created = False

        if shared_memory is not None:
            shm_name = self._name
            if not shm_name.startswith('/'):
                shm_name = '/' + shm_name
            try:
                self._shm = shared_memory.SharedMemory(
                    name=shm_name, create=True, size=self._size
                )
                buf = self._shm.buf
                created = True
                self._shm_created = True
                self._backend = 'posix_shm'
            except FileExistsError:
                self._shm = shared_memory.SharedMemory(name=shm_name, create=False)
                buf = self._shm.buf
                self._shm_created = False
                self._backend = 'posix_shm'
            except PermissionError:
                logger.warning(
                    "kv-marketplace: POSIX shared memory not permitted; falling back to mmap file"
                )
            except Exception as exc:
                logger.warning(
                    "kv-marketplace: Failed to attach POSIX shared memory (%s); falling back to mmap file",
                    exc,
                )

        if buf is None:
            buf, created = self._attach_mmap_buffer()

        if created:
            self._write_length(0, buffer=buf)
        return buf

    def _attach_mmap_buffer(self) -> tuple[memoryview, bool]:
        path = os.path.join(tempfile.gettempdir(), f"{self._name}.shm")
        created = not os.path.exists(path)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            current_size = os.path.getsize(path)
            if current_size < self._size:
                os.ftruncate(fd, self._size)
            mm = mmap.mmap(fd, self._size)
        finally:
            os.close(fd)

        self._mmap = mm
        self._mmap_path = path
        self._mmap_created = created
        self._backend = 'mmap_file'
        return memoryview(mm), created

    def _acquire_lock(self, shared: bool) -> int:
        os.makedirs(os.path.dirname(self._lock_path), exist_ok=True)
        fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(fd, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        return fd

    def _release_lock(self, fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    def _read_bytes(self) -> bytes:
        length = self._read_length()
        if length == 0:
            return b''
        start = _HEADER_STRUCT.size
        end = start + min(length, self._capacity)
        return bytes(self._buffer[start:end])

    def _write_bytes(self, data: bytes) -> None:
        if len(data) > self._capacity:
            raise RuntimeError(
                "kv-marketplace shared memory payload (%d bytes) exceeds capacity %d bytes. "
                "Increase KV_MARKETPLACE_SHM_SIZE_MB* env vars."
                % (len(data), self._capacity)
            )
        start = _HEADER_STRUCT.size
        end = start + len(data)
        self._buffer[start:end] = data
        self._write_length(len(data))

    def _write_length(self, value: int, buffer: Optional[memoryview] = None) -> None:
        target = buffer if buffer is not None else self._buffer
        _HEADER_STRUCT.pack_into(target, 0, value)

    def _read_length(self) -> int:
        return _HEADER_STRUCT.unpack_from(self._buffer, 0)[0]

    def _decode_payload(self, raw: bytes) -> Any:
        if not raw:
            return self._default_factory()
        try:
            return json.loads(raw.decode('utf-8'))
        except json.JSONDecodeError:
            logger.warning("kv-marketplace: corrupt shared memory payload; resetting")
            return self._default_factory()

    def _encode_payload(self, value: Any) -> bytes:
        return json.dumps(value, separators=(',', ':')).encode('utf-8')


__all__ = ['SharedMemoryJSONStore']
