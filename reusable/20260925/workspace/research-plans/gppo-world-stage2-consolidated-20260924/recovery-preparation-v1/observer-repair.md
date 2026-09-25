# Safe observer repair

Date: 2026-09-24

The observed failure was a Windows sharing-mode conflict: the old monitor used
`System.IO.File.ReadLines`, whose handle did not share writes.  A Python writer
that reopened the same file for append could therefore receive
`PermissionError: [Errno 13]` while the monitor was reading.

## Repair

`safe_observer.py` is a standard-library-only observer.  It lazily binds
Windows `kernel32` and opens the caller-supplied independent progress file with
the following `CreateFileW` contract:

```text
desired access: GENERIC_READ
share mode:     FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE
creation:       OPEN_EXISTING
```

Bytes are read with `ReadFile`.  `SharedReadHandle` is a context manager and
always attempts `CloseHandle`, including when `ReadFile` raises.  The public
`observe_progress` API catches acquisition, read, decode, and parse exceptions
and returns `Observation(ok=False, error=...)`; it does not raise into a writer
or restart controller.  JSON-lines parsing keeps valid records and preserves
an incomplete final JSON fragment as `partial_tail`, so a later observation can
retry it.

The observer is intended for a separate `progress.jsonl` or equivalent file.
It must not be pointed at the formal `episodes.jsonl` record as a repair
shortcut.  A caller can use:

```python
from safe_observer import observe_progress

observation = observe_progress(progress_path)
if observation.ok:
    latest_records = observation.records
else:
    record_observation_failure(observation.error)
```

The failure branch is data returned to the caller; it does not stop or restart
the experiment by itself.

## Verification

Command:

```text
python -m unittest -v test_safe_observer.py
```

The command ran on Windows and passed 8 tests.  The exact captured output is in
`observer-test-output.txt`.  The tests use only temporary files and cover:

1. A writer repeatedly opening append mode, writing JSONL, flushing and
   calling `fsync` while a shared reader holds its handle.
2. A partial JSON record and tail recovery after the writer appends the closing
   bytes.
3. Read failure followed by context-manager cleanup and a successful writer
   reopen.
4. Explicit verification that cleanup calls the close operation on a read
   exception.
5. A failed observation returned without an exception.
6. Six concurrent readers holding shared handles.
7. A hidden-window (`CREATE_NO_WINDOW`) subprocess that exits through
   `os._exit`; the parent then appends successfully after the OS releases its
   handle.
8. Static import inspection proving no environment, model, or project-native
   module is imported.

This was a zero-environment-step, zero-model-forward verification.  The test
run opened only temporary files; it did not open or modify a formal experiment
log, SQLite ledger, global configuration, model, or native source.

## Limits

The explicit sharing implementation is Windows-specific by design.  The
parser is best-effort JSONL parsing: malformed complete lines are reported in
`malformed_lines`, while a malformed final fragment remains in
`partial_tail`.  This repair does not claim that any historical run is
complete; it only removes the monitor-side handle conflict for a future
independent progress file.


集成补充：另测实际发布器并发追加、原子替换以及观察进程os._exit句柄释放。原子替换在本机失败（observer-test-output-final.txt保留）；最终run_recovery.publish_progress采用独立observer-progress.jsonl追加，失败只增加observer_publish_errors，不阻断正式日志。最终10项观察测试、13项恢复测试，共23项通过，见final-zero-step-tests.txt。
