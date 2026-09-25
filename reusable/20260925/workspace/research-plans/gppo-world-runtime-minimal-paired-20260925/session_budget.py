"""Single-writer connection lifetime mitigation, preserving original SQL/API.

This does not establish the cause of the historical SQLITE_IOERR.
"""
import sqlite3
import sys
import threading


class _Lease:
    def __init__(self, owner):self.owner=owner
    def __getattr__(self,name):return getattr(self.owner._session,name)
    def close(self):
        # Existing ledger methods use contextlib.closing per operation. Retain
        # the one connection but preserve rollback-on-uncommitted-exit behavior.
        c=self.owner._session
        if c is not None and c.in_transaction:
            primary=sys.exc_info()[1]
            try:c.rollback()
            except Exception as rollback_error:
                if primary is None:raise
                primary.add_note('Ledger rollback failed: '+repr(rollback_error))


def single_writer_budget(original_class):
    class SessionBudget(original_class):
        def __init__(self,*args,**kwargs):
            self._session=None;self._owner_thread=threading.get_ident();self._closed=False
            self.connection_open_count=0
            try:super().__init__(*args,**kwargs)
            except BaseException as primary:
                try:self.close()
                except BaseException as secondary:primary.add_note('Initialization cleanup failed: '+repr(secondary))
                raise

        def _connect(self):
            if self._closed:raise RuntimeError('Ledger session already closed')
            if threading.get_ident()!=self._owner_thread:raise RuntimeError('Ledger single-writer thread changed')
            if self._session is None:
                connection=sqlite3.connect(self.db_path,timeout=self.lock_timeout,isolation_level=None,check_same_thread=True)
                self.connection_open_count+=1
                try:
                    connection.execute('PRAGMA foreign_keys=ON')
                    connection.execute(f'PRAGMA busy_timeout={max(1,int(self.lock_timeout*1000))}')
                    row=connection.execute('PRAGMA journal_mode=WAL').fetchone()
                    if row[0].lower()!='wal':raise RuntimeError('WAL unavailable')
                    connection.execute('PRAGMA synchronous=FULL')
                except BaseException as primary:
                    try:connection.close()
                    except BaseException as secondary:primary.add_note('Connection cleanup failed: '+repr(secondary))
                    raise
                self._session=connection
            return _Lease(self)

        def close(self):
            if self._session is not None:
                self._session.close()
                self._session=None
            self._closed=True

    return SessionBudget
