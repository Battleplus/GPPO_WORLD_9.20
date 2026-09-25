"""Zero-environment-step tests for safe_observer.

Every test uses a temporary progress file.  No experiment package, model, or
native project module is imported.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import safe_observer


class SafeObserverTests(unittest.TestCase):
    def setUp(self) -> None:
        if os.name != "nt":
            self.skipTest("CreateFileW sharing tests require Windows")
        self._temporary = tempfile.TemporaryDirectory(prefix="safe-observer-")
        self.folder = Path(self._temporary.name)

    def tearDown(self) -> None:
        if hasattr(self, "_temporary"):
            self._temporary.cleanup()

    def test_import_has_no_environment_model_or_project_native_dependency(self) -> None:
        source = Path(safe_observer.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])
        self.assertFalse(imported_roots & {"torch", "numpy", "gppo_world", "gym", "tensorflow"})

    def test_create_file_shares_read_write_and_delete(self) -> None:
        path = self.folder / "progress.jsonl"
        path.write_text('{"seq": 0}\n', encoding="utf-8")

        writer_error = []
        finished = threading.Event()

        def append_repeatedly() -> None:
            try:
                for seq in range(1, 81):
                    with path.open("a", encoding="utf-8", newline="") as stream:
                        stream.write(json.dumps({"seq": seq}) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    time.sleep(0.001)
            except BaseException as exc:  # report thread failures in the parent
                writer_error.append(exc)
            finally:
                finished.set()

        with safe_observer.open_shared_read(path) as reader:
            writer = threading.Thread(target=append_repeatedly, name="append-writer")
            writer.start()
            while not finished.is_set():
                reader.read()
                time.sleep(0.001)
            writer.join(timeout=5)

        self.assertFalse(writer.is_alive())
        self.assertEqual(writer_error, [])
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(records), 81)
        self.assertEqual(records[-1]["seq"], 80)

    def test_partial_json_tail_is_safe_and_retryable(self) -> None:
        path = self.folder / "progress.jsonl"
        with path.open("wb") as stream:
            stream.write(b'{"seq": 1}\n{"seq": 2')
            stream.flush()
            os.fsync(stream.fileno())

        result = safe_observer.observe_progress(path)
        self.assertTrue(result.ok)
        self.assertEqual(result.records, ({"seq": 1},))
        self.assertEqual(result.partial_tail, '{"seq": 2')

        with path.open("a", encoding="utf-8", newline="") as stream:
            stream.write('}\n')
            stream.flush()
            os.fsync(stream.fileno())
        result = safe_observer.observe_progress(path)
        self.assertTrue(result.ok)
        self.assertEqual(result.records, ({"seq": 1}, {"seq": 2}))
        self.assertEqual(result.partial_tail, "")

    def test_read_exception_closes_handle_and_writer_can_reopen(self) -> None:
        path = self.folder / "progress.jsonl"
        path.write_text('{"seq": 1}\n', encoding="utf-8")

        with self.assertRaises(OSError):
            with safe_observer.open_shared_read(path) as reader:
                with patch.object(reader._api, "read_file", side_effect=OSError("synthetic read failure")):
                    reader.read()

        with path.open("a", encoding="utf-8", newline="") as stream:
            stream.write('{"seq": 2}\n')
            stream.flush()
            os.fsync(stream.fileno())
        self.assertEqual(path.read_text(encoding="utf-8").count("\n"), 2)

    def test_context_manager_calls_close_even_when_read_raises(self) -> None:
        class FakeApi:
            def __init__(self) -> None:
                self.closed = []

            def create_file(self, path: str) -> object:
                return object()

            def read_file(self, handle: object, size: int) -> bytes:
                raise OSError("synthetic read failure")

            def close_handle(self, handle: object) -> None:
                self.closed.append(handle)

        fake = FakeApi()
        with patch.object(safe_observer, "_get_api", return_value=fake):
            with self.assertRaises(OSError):
                with safe_observer.open_shared_read(self.folder / "independent-progress.jsonl") as reader:
                    reader.read()
        self.assertEqual(len(fake.closed), 1)

    def test_observation_failure_is_returned_without_raising(self) -> None:
        path = self.folder / "independent-progress.jsonl"
        with patch.object(safe_observer, "read_shared_bytes", side_effect=PermissionError(13, "denied")):
            result = safe_observer.observe_progress(path)
        self.assertFalse(result.ok)
        self.assertEqual(result.records, ())
        self.assertIn("PermissionError", result.error or "")

    def test_multiple_readers_can_hold_shared_handles(self) -> None:
        path = self.folder / "progress.jsonl"
        payload = "".join(json.dumps({"seq": seq}) + "\n" for seq in range(120))
        path.write_bytes(payload.encode("utf-8"))
        barrier = threading.Barrier(6)

        def read_once() -> bytes:
            with safe_observer.open_shared_read(path) as reader:
                barrier.wait(timeout=5)
                return reader.read()

        with ThreadPoolExecutor(max_workers=6, thread_name_prefix="reader") as pool:
            results = list(pool.map(lambda _: read_once(), range(6)))
        self.assertEqual(results, [payload.encode("utf-8")] * 6)

    def test_abnormal_subprocess_exit_releases_os_handle(self) -> None:
        path = self.folder / "progress.jsonl"
        ready = self.folder / "writer-ready"
        path.write_text("", encoding="utf-8")
        child = (
            "import os,sys,time\n"
            "path,ready=sys.argv[1:3]\n"
            "stream=open(path,'a',encoding='utf-8',newline='')\n"
            "stream.write('{\\\"child\\\":true}\\n');stream.flush()\n"
            "open(ready,'w',encoding='ascii').close()\n"
            "time.sleep(0.05)\n"
            "os._exit(23)\n"
        )
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = subprocess.Popen(
            [sys.executable, "-c", child, str(path), str(ready)],
            creationflags=creationflags,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + 5
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(ready.exists(), "hidden writer subprocess did not signal readiness")
            self.assertEqual(process.wait(timeout=5), 23)
            with path.open("a", encoding="utf-8", newline="") as stream:
                stream.write('{"parent":true}\n')
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
        self.assertIn('"parent":true', path.read_text(encoding="utf-8"))

    def test_actual_progress_publisher_appends_while_reader_holds_file(self) -> None:
        import types
        from run_recovery import publish_progress
        path = self.folder / 'observer-progress.jsonl'
        path.write_text('', encoding='utf-8')
        monitor=types.SimpleNamespace(out=self.folder)
        with safe_observer.open_shared_read(path) as reader:
            for i in range(80):publish_progress(monitor,i,('p',0,'M'))
            self.assertEqual(len(reader.read().splitlines()),80)
        self.assertFalse(hasattr(monitor,'observer_publish_errors'))
        self.assertEqual(len(safe_observer.observe_progress(path).records),80)

    def test_abnormal_observer_exit_releases_reader_handle(self) -> None:
        path = self.folder / 'progress.json'
        ready = self.folder / 'reader-ready'
        path.write_text('{"seq":1}\n', encoding='utf-8')
        child = "import os,sys,time; from pathlib import Path; import safe_observer; h=safe_observer.open_shared_read(sys.argv[1]); Path(sys.argv[2]).touch(); time.sleep(.1); os._exit(24)"
        process = subprocess.Popen([sys.executable, '-B', '-c', child, str(path), str(ready)],
            cwd=str(Path(safe_observer.__file__).parent), creationflags=subprocess.CREATE_NO_WINDOW,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            self.assertEqual(process.wait(timeout=5), 24)
            self.assertTrue(ready.exists())
            api = safe_observer._get_api()
            handle = api._create_file(str(path), safe_observer.GENERIC_READ, 0, None,
                                      safe_observer.OPEN_EXISTING, safe_observer.FILE_ATTRIBUTE_NORMAL, None)
            self.assertNotEqual(safe_observer._handle_value(handle), safe_observer.INVALID_HANDLE_VALUE)
            api.close_handle(handle)
        finally:
            if process.poll() is None:
                process.kill(); process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
