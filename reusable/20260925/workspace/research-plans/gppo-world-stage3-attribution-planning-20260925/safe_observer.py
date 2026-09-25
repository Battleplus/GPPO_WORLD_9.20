"""Read an independent progress file without blocking a Windows writer.

The observer deliberately opens a file with all three Windows sharing flags.
It is intended for a small progress file (for example ``progress.jsonl``),
not for the experiment's formal ``episodes.jsonl`` record.

Only the standard library is used.  The kernel32 binding is lazy so importing
this module cannot start an environment, load a model, or open a file.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import json
import os
import threading
from typing import Any, Iterable, Optional, Union


PathLike = Union[str, bytes, os.PathLike[str], os.PathLike[bytes]]

# CreateFileW constants.  Keeping these values here makes the sharing contract
# visible in code review and avoids relying on a higher-level open wrapper.
GENERIC_READ = 0x80000000
FILE_SHARE_READ = 0x00000001
FILE_SHARE_WRITE = 0x00000002
FILE_SHARE_DELETE = 0x00000004
OPEN_EXISTING = 3
FILE_ATTRIBUTE_NORMAL = 0x00000080
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
ERROR_HANDLE_EOF = 38

READ_SHARE_FLAGS = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE


def _handle_value(handle: Any) -> Any:
    """Return the integer value of a ctypes HANDLE or a test double."""

    return getattr(handle, "value", handle)


def _error(operation: str, code: Optional[int] = None) -> OSError:
    if code is None:
        code = ctypes.get_last_error()
    try:
        message = ctypes.FormatError(code)
    except (AttributeError, ValueError):
        message = os.strerror(code) if code else "unknown error"
    return OSError(code, "%s failed: %s" % (operation, message))


class _Kernel32Api:
    """Small, typed kernel32 surface used by SharedReadHandle."""

    def __init__(self) -> None:
        if os.name != "nt":
            raise OSError("safe_observer requires Windows CreateFileW")

        library = ctypes.WinDLL("kernel32", use_last_error=True)

        self._create_file = library.CreateFileW
        self._create_file.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        self._create_file.restype = wintypes.HANDLE

        self._read_file = library.ReadFile
        self._read_file.argtypes = [
            wintypes.HANDLE,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            wintypes.LPVOID,
        ]
        self._read_file.restype = wintypes.BOOL

        self._close_handle = library.CloseHandle
        self._close_handle.argtypes = [wintypes.HANDLE]
        self._close_handle.restype = wintypes.BOOL

    def create_file(self, path: str) -> Any:
        handle = self._create_file(
            path,
            GENERIC_READ,
            READ_SHARE_FLAGS,
            None,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            None,
        )
        value = _handle_value(handle)
        if value in (None, 0, INVALID_HANDLE_VALUE):
            raise _error("CreateFileW")
        return handle

    def read_file(self, handle: Any, size: int) -> bytes:
        buffer = ctypes.create_string_buffer(size)
        count = wintypes.DWORD(0)
        ok = self._read_file(handle, buffer, size, ctypes.byref(count), None)
        if not ok:
            code = ctypes.get_last_error()
            if code == ERROR_HANDLE_EOF:
                return b""
            raise _error("ReadFile", code)
        return buffer.raw[: count.value]

    def close_handle(self, handle: Any) -> None:
        if not self._close_handle(handle):
            raise _error("CloseHandle")


_API: Optional[_Kernel32Api] = None
_API_LOCK = threading.Lock()


def _get_api() -> _Kernel32Api:
    global _API
    if _API is None:
        with _API_LOCK:
            if _API is None:
                _API = _Kernel32Api()
    return _API


def _path_text(path: PathLike) -> str:
    value = os.fspath(path)
    if isinstance(value, bytes):
        value = os.fsdecode(value)
    if not isinstance(value, str) or not value:
        raise ValueError("path must be a non-empty filesystem path")
    return value


class SharedReadHandle:
    """A synchronous read handle opened with permissive Windows sharing.

    The handle is acquired at construction so ``open_shared_read(path)`` has
    the same useful failure point as ``open(path)``.  ``__exit__`` always calls
    CloseHandle after a successful acquisition, including an exceptional read.
    """

    def __init__(self, path: PathLike, chunk_size: int = 1024 * 1024) -> None:
        if chunk_size <= 0 or chunk_size > 0x7FFFFFFF:
            raise ValueError("chunk_size must be between 1 and 2^31-1")
        self.path = _path_text(path)
        self.chunk_size = int(chunk_size)
        self._api = _get_api()
        self._handle = self._api.create_file(self.path)
        self._closed = False

    def __enter__(self) -> "SharedReadHandle":
        if self._closed:
            raise ValueError("read handle is closed")
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        # Do not let a close error hide the original read error.  With no
        # active exception, a CloseHandle failure remains observable to caller.
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise
        return False

    def close(self) -> None:
        if self._closed:
            return
        handle = self._handle
        self._handle = None
        self._closed = True
        self._api.close_handle(handle)

    def read(self, size: int = -1) -> bytes:
        if self._closed or self._handle is None:
            raise ValueError("read handle is closed")
        if not isinstance(size, int):
            raise TypeError("size must be an integer")
        if size < -1:
            raise ValueError("size must be -1 or non-negative")
        if size == 0:
            return b""

        parts = []
        remaining = size
        while remaining != 0:
            request = self.chunk_size if remaining < 0 else min(self.chunk_size, remaining)
            block = self._api.read_file(self._handle, request)
            if not block:
                break
            parts.append(block)
            if remaining > 0:
                remaining -= len(block)
        return b"".join(parts)

    def __del__(self) -> None:
        # Context management is the ownership contract.  This is only a last
        # resort for callers that abandon an instance without entering it.
        try:
            self.close()
        except Exception:
            pass


def open_shared_read(path: PathLike, chunk_size: int = 1024 * 1024) -> SharedReadHandle:
    """Open ``path`` read-only while allowing read/write/delete by other users."""

    return SharedReadHandle(path, chunk_size=chunk_size)


def read_shared_bytes(path: PathLike, chunk_size: int = 1024 * 1024) -> bytes:
    """Read one snapshot; exceptions are intentionally left to the caller."""

    with open_shared_read(path, chunk_size=chunk_size) as reader:
        return reader.read()


@dataclass(frozen=True)
class JsonlSnapshot:
    """Best-effort parse of a JSON-lines snapshot."""

    records: tuple[Any, ...] = ()
    partial_tail: str = ""
    malformed_lines: tuple[str, ...] = ()
    complete_lines: int = 0

    @property
    def trailing(self) -> str:
        return self.partial_tail


def _line_ending_present(line: str) -> bool:
    return line.endswith("\n") or line.endswith("\r")


def _parse_one(line: str) -> Any:
    return json.loads(line)


def parse_jsonl_snapshot(text: str) -> JsonlSnapshot:
    """Parse complete JSONL records and preserve an incomplete final tail.

    A valid final record without a newline is accepted.  An invalid final
    fragment is retained as ``partial_tail`` so callers can retry later.
    Malformed complete lines are recorded and skipped without raising.
    """

    if not isinstance(text, str):
        raise TypeError("text must be str")

    pieces = text.splitlines(keepends=True)
    tail = ""
    if pieces and not _line_ending_present(pieces[-1]):
        tail = pieces.pop()

    records = []
    malformed = []
    complete_lines = 0
    for piece in pieces:
        complete_lines += 1
        line = piece.rstrip("\r\n")
        if not line.strip():
            continue
        try:
            records.append(_parse_one(line))
        except (TypeError, ValueError, json.JSONDecodeError):
            malformed.append(line[:256])

    partial_tail = ""
    if tail and tail.strip():
        try:
            records.append(_parse_one(tail))
        except (TypeError, ValueError, json.JSONDecodeError):
            partial_tail = tail

    return JsonlSnapshot(
        records=tuple(records),
        partial_tail=partial_tail,
        malformed_lines=tuple(malformed),
        complete_lines=complete_lines,
    )


@dataclass(frozen=True)
class Observation:
    """Result returned by observe_progress; failures never escape as errors."""

    ok: bool
    path: str
    text: str = ""
    bytes_read: int = 0
    snapshot: JsonlSnapshot = JsonlSnapshot()
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.ok

    @property
    def records(self) -> tuple[Any, ...]:
        return self.snapshot.records

    @property
    def partial_tail(self) -> str:
        return self.snapshot.partial_tail

    @property
    def trailing(self) -> str:
        return self.snapshot.partial_tail

    @property
    def malformed_lines(self) -> tuple[str, ...]:
        return self.snapshot.malformed_lines


def _display_path(path: Any) -> str:
    try:
        return _path_text(path)
    except Exception:
        return repr(path)


def _failure(path: Any, exc: BaseException) -> Observation:
    return Observation(
        ok=False,
        path=_display_path(path),
        error="%s: %s" % (type(exc).__name__, exc),
    )


def observe_progress(path: PathLike, encoding: str = "utf-8") -> Observation:
    """Return a safe, best-effort observation of an independent progress file.

    File acquisition, reading, decoding, and parsing errors become an
    ``Observation(ok=False, ...)`` result.  They do not raise and therefore
    cannot interrupt a writer or a restart controller.
    """

    try:
        raw = read_shared_bytes(path)
        text = raw.decode(encoding, errors="replace")
        snapshot = parse_jsonl_snapshot(text)
        return Observation(
            ok=True,
            path=_display_path(path),
            text=text,
            bytes_read=len(raw),
            snapshot=snapshot,
        )
    except Exception as exc:
        return _failure(path, exc)


def observe_file(path: PathLike, encoding: str = "utf-8") -> Observation:
    """Alias emphasizing that the caller supplies the independent file."""

    return observe_progress(path, encoding=encoding)


def safe_observe(path: PathLike, encoding: str = "utf-8") -> Observation:
    """Compatibility alias for callers that prefer an explicitly safe name."""

    return observe_progress(path, encoding=encoding)


def read_progress(path: PathLike, encoding: str = "utf-8") -> Observation:
    """Short alias for the exception-isolating progress observation API."""

    return observe_progress(path, encoding=encoding)


def read_progress_file(path: PathLike, encoding: str = "utf-8") -> Observation:
    """Descriptive alias for integrations that call the file a progress file."""

    return observe_progress(path, encoding=encoding)


def _json_result(result: Observation) -> dict[str, Any]:
    return {
        "ok": result.ok,
        "path": result.path,
        "bytes_read": result.bytes_read,
        "records": list(result.records),
        "partial_tail": result.partial_tail,
        "malformed_lines": list(result.malformed_lines),
        "error": result.error,
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Safely observe an independent JSONL progress file")
    parser.add_argument("path")
    parser.add_argument("--encoding", default="utf-8")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = observe_progress(args.path, encoding=args.encoding)
    print(json.dumps(_json_result(result), ensure_ascii=False, sort_keys=True))
    return 0 if result.ok else 1


__all__ = [
    "FILE_SHARE_DELETE",
    "FILE_SHARE_READ",
    "FILE_SHARE_WRITE",
    "JsonlSnapshot",
    "Observation",
    "READ_SHARE_FLAGS",
    "SharedReadHandle",
    "open_shared_read",
    "observe_file",
    "observe_progress",
    "parse_jsonl_snapshot",
    "read_shared_bytes",
    "read_progress",
    "read_progress_file",
    "safe_observe",
]


if __name__ == "__main__":
    raise SystemExit(main())
