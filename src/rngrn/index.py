"""index.py — append-only metadata indices with two interchangeable backends.

Both the run index (Stage 6) and the dataset registry are metadata ledgers: append
one row per run / per dataset, read them all back, occasionally query. The house
style (nn-research-codebase-principles.md) is emphatic that databases hold METADATA
ONLY — never field arrays (those live in HDF5 payloads / the content cache).

Two backends behind one interface, selected by tracking.index_backend:
  jsonl  : append-only <name>.jsonl, one JSON object per line. Zero setup, diff-
           friendly, the default so the template dry-runs with no DB. Querying is a
           Python filter (fine at the scales a template reaches).
  sqlite : <name> table in index.db with a dynamic, additive schema (columns are
           added as new row keys appear). Enables real SQL once runs pile up, e.g.
           SELECT ... WHERE recovered_turing AND kstar_rel_err < 0.15 GROUP BY arm_id.

Same rows go into either; switching backend does not change what a row means.

BUT A ROW'S MEANING CAN CHANGE OVER TIME, AND THE INDEX DOES NOT VERSION IT. Two columns
changed definition on 2026-08-04 without changing name, so a ledger spanning that date holds
two definitions in one column:
  * `recovered_turing` — was `tr(J) < 0` (which a uniformly UNSTABLE system satisfies), is
    now the strict `max Re eig(J) < 0` (D-EVID-11). New rows carry
    `turing_criterion = "strict_max_re_eig"`; ABSENT means the superseded verdict. Filter on
    it before pooling old and new rows.
  * grouping — `config_id` hashes `train.seed`, so it identifies a RUN, never an arm. Group
    on `arm_id` (D-EVID-13); rows written before it existed carry none.
A query that ignores both silently mixes generations. This is the one thing switching
backend does not protect you from.
"""
from __future__ import annotations
import json
import os
import sqlite3
from datetime import datetime, timezone


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _sql_type(v):
    if isinstance(v, bool):      # bool before int (bool is an int subclass)
        return "INTEGER"
    if isinstance(v, int):
        return "INTEGER"
    if isinstance(v, float):
        return "REAL"
    return "TEXT"


def _coerce(v):
    """Reduce a value to something SQLite can store; dicts/lists -> JSON text."""
    if isinstance(v, bool):
        return int(v)
    if v is None or isinstance(v, (int, float, str)):
        return v
    return json.dumps(v, default=str)


class JsonlIndex:
    def __init__(self, root, name):
        self.path = os.path.join(root, f"{name}.jsonl")
        os.makedirs(root, exist_ok=True)

    def append(self, row: dict):
        row = dict(row); row.setdefault("_ts", _now_iso())
        with open(self.path, "a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")

    def read(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path) as fh:
            return [json.loads(ln) for ln in fh if ln.strip()]

    def query(self, where=None, params=()):
        """Python-side filter: `where` is a predicate callable row->bool (SQL string
        ignored here). Kept so callers can be backend-agnostic for simple filters."""
        rows = self.read()
        return [r for r in rows if where(r)] if callable(where) else rows

    def get(self, key_col, key_val):
        for r in self.read():
            if r.get(key_col) == key_val:
                return r
        return None


class SqliteIndex:
    def __init__(self, root, name):
        os.makedirs(root, exist_ok=True)
        self.db = os.path.join(root, "index.db")
        self.name = name
        self._ensure_table()

    def _conn(self):
        c = sqlite3.connect(self.db)
        c.row_factory = sqlite3.Row
        return c

    def _ensure_table(self):
        with self._conn() as c:
            c.execute(f'CREATE TABLE IF NOT EXISTS "{self.name}" '
                      f'(_id INTEGER PRIMARY KEY AUTOINCREMENT, _ts TEXT)')

    def _existing_cols(self, c):
        return {r["name"] for r in c.execute(f'PRAGMA table_info("{self.name}")')}

    def append(self, row: dict):
        row = dict(row); row.setdefault("_ts", _now_iso())
        with self._conn() as c:
            have = self._existing_cols(c)
            for k, v in row.items():
                if k not in have:
                    c.execute(f'ALTER TABLE "{self.name}" ADD COLUMN "{k}" {_sql_type(v)}')
                    have.add(k)
            cols = list(row)
            ph = ",".join("?" * len(cols))
            c.execute(f'INSERT INTO "{self.name}" ({",".join(chr(34)+k+chr(34) for k in cols)}) '
                      f'VALUES ({ph})', [_coerce(row[k]) for k in cols])

    def read(self) -> list[dict]:
        with self._conn() as c:
            try:
                rows = c.execute(f'SELECT * FROM "{self.name}"').fetchall()
            except sqlite3.OperationalError:
                return []
        return [self._clean(dict(r)) for r in rows]

    def query(self, where=None, params=()):
        """`where` is a SQL predicate string (no leading WHERE), e.g.
        "recovered_turing=1 AND kstar_rel_err < ?". Returns list[dict].

        On a ledger spanning 2026-08-04, add `AND turing_criterion = 'strict_max_re_eig'`:
        `recovered_turing` changed meaning on that date and older rows carry the superseded
        loose verdict under the same column name — see this module's docstring."""
        if callable(where):        # allow the same predicate-callable API as JSONL
            return [r for r in self.read() if where(r)]
        sql = f'SELECT * FROM "{self.name}"'
        if where:
            sql += f" WHERE {where}"
        with self._conn() as c:
            rows = c.execute(sql, params).fetchall()
        return [self._clean(dict(r)) for r in rows]

    def get(self, key_col, key_val):
        rows = self.query(f'"{key_col}"=?', (key_val,))
        return rows[0] if rows else None

    @staticmethod
    def _clean(d):
        d.pop("_id", None)
        return d


def open_index(root: str, name: str, backend: str = "jsonl"):
    """Factory. backend in {'jsonl','sqlite'}. Same row schema either way."""
    if backend == "jsonl":
        return JsonlIndex(root, name)
    if backend == "sqlite":
        return SqliteIndex(root, name)
    raise ValueError(f"unknown index backend '{backend}' (jsonl|sqlite)")
