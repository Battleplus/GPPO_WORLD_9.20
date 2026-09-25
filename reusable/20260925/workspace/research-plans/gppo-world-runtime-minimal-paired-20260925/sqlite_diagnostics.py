"""Optional future ledger diagnostics; no retries, transactions, or model imports."""
import json
import os
import sqlite3
import tempfile
import time
import traceback
import uuid
from pathlib import Path


def call_once(operation, function, *args, diagnostic_dir, db_path, **kwargs):
    """Preserve the primary exception and call the supplied operation exactly once."""
    try:
        return function(*args, **kwargs)
    except BaseException as error:
        record = {
            'operation': operation, 'database_path': str(db_path), 'pid': os.getpid(),
            'time_unix': time.time(), 'exception_type': type(error).__name__,
            'message': str(error), 'sqlite_errorcode': getattr(error, 'sqlite_errorcode', None),
            'sqlite_errorname': getattr(error, 'sqlite_errorname', None),
            'errno': getattr(error, 'errno', None), 'winerror': getattr(error, 'winerror', None),
            'sqlite_version': sqlite3.sqlite_version, 'traceback': traceback.format_exc(),
            'automatic_retry': False,
        }
        # Do not reopen the failed database, change WAL mode or inspect another process.
        temporary = None
        try:
            directory = Path(diagnostic_dir)
            directory.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix='sqlite-error-', suffix='.tmp', dir=directory)
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                json.dump(record, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, directory / ('sqlite-error-' + uuid.uuid4().hex + '.json'))
        except Exception as diagnostic_error:
            # Preserve the original failure even if the same storage device also fails here.
            error.add_note('Diagnostic publication failed: ' + type(diagnostic_error).__name__)
        finally:
            if temporary:
                try:
                    Path(temporary).unlink(missing_ok=True)
                except OSError:
                    pass
        raise
