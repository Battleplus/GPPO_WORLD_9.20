import importlib.util
import json
import sqlite3
import tempfile
import time
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from sqlite_diagnostics import call_once

HERE = Path(__file__).resolve().parent
NATIVE = Path(r'E:\Z博士\migration-artifacts\event-trigger-aware-gppo-fair-replication-20260919-v1\source-snapshot\gppo_world\budget_executor.py')


class DiagnosticsTests(unittest.TestCase):
    def test_primary_error_extended_fields_no_retry(self):
        with tempfile.TemporaryDirectory(prefix='sqlite-diagnostic-test-', dir=HERE) as d:
            calls = []
            error = sqlite3.OperationalError('injected disk I/O error')
            error.sqlite_errorcode = 778
            error.sqlite_errorname = 'SQLITE_IOERR_WRITE'
            def fail():
                calls.append(1)
                raise error
            with self.assertRaises(sqlite3.OperationalError) as caught:
                call_once('reserve', fail, diagnostic_dir=d, db_path=Path(d)/'temporary.sqlite3')
            self.assertIs(caught.exception, error)
            self.assertEqual(calls, [1])
            files = list(Path(d).glob('*.json'))
            self.assertEqual(len(files), 1)
            row = json.loads(files[0].read_text(encoding='utf-8'))
            self.assertEqual(row['sqlite_errorname'], 'SQLITE_IOERR_WRITE')
            self.assertEqual(row['sqlite_errorcode'], 778)
            self.assertIn('injected disk I/O error', row['traceback'])
            self.assertFalse(row['automatic_retry'])

    def test_diagnostic_failure_does_not_replace_primary(self):
        error = sqlite3.OperationalError('injected primary')
        with tempfile.TemporaryDirectory(prefix='sqlite-diagnostic-test-', dir=HERE) as d:
            with patch('sqlite_diagnostics.tempfile.mkstemp', side_effect=OSError('injected sink')):
                with self.assertRaises(sqlite3.OperationalError) as caught:
                    call_once('complete', lambda: (_ for _ in ()).throw(error), diagnostic_dir=d, db_path=Path(d)/'temporary.sqlite3')
        self.assertIs(caught.exception, error)
        self.assertIn('Diagnostic publication failed: OSError', error.__notes__)

    def test_success_result_passes_through_without_diagnostics(self):
        with tempfile.TemporaryDirectory(prefix='sqlite-diagnostic-test-', dir=HERE) as d:
            result = object()
            self.assertIs(call_once('reserve', lambda: result, diagnostic_dir=d, db_path=Path(d)/'temporary.sqlite3'), result)
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_original_budget_on_disposable_database(self):
        spec = importlib.util.spec_from_file_location('isolated_original_budget', NATIVE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        start = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix='sqlite-ledger-test-', dir=HERE) as d:
            db = Path(d)/'temporary.sqlite3'
            ledger = module.PersistentBudget(db, limits={'temporary_io_operation':512},
                                             attempt_id='temporary-test-only', run_id='temporary-test-only')
            for i in range(512):
                if time.perf_counter()-start > 30:
                    self.fail('Temporary I/O test wall bound reached')
                token = call_once('reserve', ledger.reserve, 'temporary_io_operation', diagnostic_dir=Path(d)/'diagnostics', db_path=db)
                call_once('complete', ledger.complete, token, diagnostic_dir=Path(d)/'diagnostics', db_path=db)
            state = ledger.snapshot()['stages']['temporary_io_operation']
            self.assertEqual(state, {'reserved':512,'verified':512,'unknown':0,'pending':0})
            with closing(sqlite3.connect(db.as_uri()+'?mode=ro',uri=True)) as c:
                self.assertEqual(c.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
            self.assertFalse((Path(d)/'diagnostics').exists())
            self.assertLess(sum(p.stat().st_size for p in Path(d).rglob('*') if p.is_file()), 64*1024**2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
