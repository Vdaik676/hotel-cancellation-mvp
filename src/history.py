"""Opt-in local history of results only. Never store booking inputs."""
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

MAX_RESULTS = 500

def save_result(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=2)) as db, db:
        db.execute("""CREATE TABLE IF NOT EXISTS checks (
            id INTEGER PRIMARY KEY, checked_at TEXT NOT NULL, probability REAL NOT NULL,
            risk TEXT NOT NULL, model_id TEXT NOT NULL, threshold REAL NOT NULL)""")
        db.execute("INSERT INTO checks(checked_at,probability,risk,model_id,threshold) VALUES(?,?,?,?,?)",
                   (datetime.now(timezone.utc).isoformat(timespec='seconds'),result['probability'],
                    result['risk'],result['model_id'],result['threshold']))
        db.execute("DELETE FROM checks WHERE id IN (SELECT id FROM checks ORDER BY id DESC LIMIT -1 OFFSET ?)",(MAX_RESULTS,))

def read_results(path):
    if not path.is_file():
        return []
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=2)) as db:
        db.row_factory=sqlite3.Row
        return [dict(row) for row in db.execute("SELECT checked_at,probability,risk,model_id,threshold FROM checks ORDER BY id DESC LIMIT ?",(MAX_RESULTS,))]

def clear_results(path):
    if path.is_file():
        with closing(sqlite3.connect(path,timeout=2)) as db, db:
            db.execute('PRAGMA secure_delete=ON')
            db.execute('DELETE FROM checks')
