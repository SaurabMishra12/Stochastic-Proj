"""Durable, conservative accounting for every dispatched HTTP attempt."""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
from pathlib import Path


class BudgetExceeded(RuntimeError):
    pass


class DuplicateLogicalCall(RuntimeError):
    pass


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _money(value):
    amount = Decimal(str(value))
    if not amount.is_finite() or amount < 0:
        raise ValueError("money values must be finite and nonnegative")
    return int((amount * 1_000_000_000).to_integral_value(rounding=ROUND_CEILING))


class Ledger:
    """Persist reservations before dispatch; missing usage never releases money.

    Prices are USD per million tokens. The default prices are provisional and
    must be verified against the Azure deployment before approving a run.
    """

    def __init__(self, path, approval_hash, max_usd, max_attempts, prices=None,
                 max_logical_calls=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not isinstance(approval_hash, str) or not approval_hash.strip():
            raise ValueError("a nonempty approval_hash is required")
        _integer(max_attempts, "max_attempts")
        if max_logical_calls is not None:
            _integer(max_logical_calls, "max_logical_calls")
        prices = dict(prices or {"input": .25, "output": 2})
        self.input_rate = Decimal(str(prices.get("input", prices.get("input_rate", .25))))
        self.output_rate = Decimal(str(prices.get("output", prices.get("output_rate", 2))))
        _money(self.input_rate)
        _money(self.output_rate)
        self.max_nusd = _money(max_usd)
        self.max_attempts = max_attempts
        self.max_logical_calls = max_logical_calls
        descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(descriptor)
        os.chmod(self.path, 0o600)
        policy = json.dumps({"approval_hash": approval_hash, "max_nusd": self.max_nusd,
            "max_attempts": max_attempts, "max_logical_calls": max_logical_calls,
            "input_rate": str(self.input_rate), "output_rate": str(self.output_rate)}, sort_keys=True)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("CREATE TABLE IF NOT EXISTS policy (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL)")
            connection.execute("""CREATE TABLE IF NOT EXISTS attempts (
                attempt_id TEXT PRIMARY KEY, logical_id TEXT NOT NULL,
                reserved_nusd INTEGER NOT NULL, actual_nusd INTEGER,
                input_bound INTEGER NOT NULL, output_bound INTEGER NOT NULL,
                usage_json TEXT, outcome TEXT NOT NULL, created_at TEXT NOT NULL,
                settled_at TEXT)""")
            old = connection.execute("SELECT value FROM policy WHERE id=1").fetchone()
            if old and old[0] != policy:
                raise ValueError("ledger approval or budget policy does not match the existing policy")
            connection.execute("INSERT OR IGNORE INTO policy VALUES (1, ?)", (policy,))

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA synchronous=FULL")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def cost_nusd(self, input_tokens, output_tokens):
        _integer(input_tokens, "input_tokens")
        _integer(output_tokens, "output_tokens")
        amount = (Decimal(input_tokens) * self.input_rate + Decimal(output_tokens) * self.output_rate) / 1_000_000
        return _money(amount)

    def reserve(self, attempt_id, logical_id, input_bound, output_bound, *, allow_existing_logical=True):
        if not isinstance(attempt_id, str) or not attempt_id or not isinstance(logical_id, str) or not logical_id:
            raise ValueError("attempt_id and logical_id must be nonempty strings")
        reserved = self.cost_nusd(input_bound, output_bound)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("SELECT 1 FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone():
                raise ValueError("attempt_id already reserved; redispatch is forbidden")
            state = self._summary(connection)
            if state["attempts"] >= self.max_attempts:
                raise BudgetExceeded("HTTP attempt limit exhausted")
            if state["charged_nusd"] + reserved > self.max_nusd:
                raise BudgetExceeded("spend cap cannot cover the next HTTP attempt")
            known_logical = connection.execute("SELECT 1 FROM attempts WHERE logical_id=? LIMIT 1", (logical_id,)).fetchone()
            if known_logical and not allow_existing_logical:
                raise DuplicateLogicalCall("logical_id has prior attempts; redispatch is forbidden")
            if (not known_logical and self.max_logical_calls is not None
                    and state["logical_calls"] >= self.max_logical_calls):
                raise BudgetExceeded("logical call limit exhausted")
            connection.execute("INSERT INTO attempts VALUES (?, ?, ?, NULL, ?, ?, NULL, 'reserved', ?, NULL)",
                (attempt_id, logical_id, reserved, input_bound, output_bound, _utc_now()))
            return dict(connection.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone())

    def settle(self, attempt_id, usage=None, outcome="unknown"):
        actual = None
        if usage is not None:
            if not isinstance(usage, dict) or "input_tokens" not in usage or "output_tokens" not in usage:
                raise ValueError("usage must contain input_tokens and output_tokens")
            actual = self.cost_nusd(usage["input_tokens"], usage["output_tokens"])
        usage_json = json.dumps(usage, sort_keys=True, allow_nan=False) if usage is not None else None
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            if row is None:
                raise ValueError("attempt has no reservation")
            if row["settled_at"] is not None:
                if row["usage_json"] != usage_json or row["outcome"] != outcome:
                    raise ValueError("attempt already settled with different evidence")
                return dict(row)
            connection.execute("UPDATE attempts SET actual_nusd=?, usage_json=?, outcome=?, settled_at=? WHERE attempt_id=?",
                (actual, usage_json, str(outcome), _utc_now(), attempt_id))
            return dict(connection.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone())

    def _summary(self, connection):
        row = connection.execute("""SELECT COUNT(*) AS attempts, COUNT(DISTINCT logical_id) AS logical_calls,
            COALESCE(SUM(COALESCE(actual_nusd, reserved_nusd)),0) AS charged_nusd,
            COALESCE(SUM(actual_nusd),0) AS actual_nusd,
            COALESCE(SUM(CASE WHEN actual_nusd IS NULL THEN reserved_nusd ELSE 0 END),0) AS reserved_nusd,
            COALESCE(SUM(CASE WHEN actual_nusd IS NULL THEN 1 ELSE 0 END),0) AS uncertain_attempts
            FROM attempts""").fetchone()
        result = dict(row)
        for name in ("charged", "actual", "reserved"):
            result[name + "_usd"] = result[name + "_nusd"] / 1_000_000_000
        result.update({"max_usd": self.max_nusd / 1_000_000_000,
            "remaining_usd": max(0, self.max_nusd - result["charged_nusd"]) / 1_000_000_000,
            "remaining_attempts": max(0, self.max_attempts - result["attempts"]),
            "over_budget": result["charged_nusd"] > self.max_nusd})
        return result

    def summary(self):
        with self._connect() as connection:
            return self._summary(connection)

    def logical_attempts(self, logical_id):
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(
                "SELECT * FROM attempts WHERE logical_id=? ORDER BY rowid", (logical_id,))]
