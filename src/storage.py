"""Lightweight SQLite persistence for classroom custom problems and monitor feedback."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "copilot.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS custom_problems (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            platform TEXT NOT NULL,
            title TEXT NOT NULL,
            statement TEXT NOT NULL,
            sample_input TEXT DEFAULT '',
            expected_output TEXT DEFAULT '',
            created_at INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS monitor_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            language TEXT NOT NULL,
            bug_class TEXT NOT NULL,
            algo_family TEXT NOT NULL,
            monitor_notes TEXT DEFAULT '',
            created_at INTEGER NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def save_custom_problem(
    platform: str,
    title: str,
    statement: str,
    sample_input: str = "",
    expected_output: str = "",
) -> dict[str, Any]:
    """Persist a classroom problem (NEPS, Beecrowd, Codeforces, AtCoder, Uber Code Club)."""
    now = int(time.time())
    try:
        with _connect() as conn:
            cur = conn.execute(
                "INSERT INTO custom_problems (platform, title, statement, sample_input, expected_output, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (platform.strip() or "Aula", title.strip(), statement.strip(), sample_input, expected_output, now),
            )
            pid = cur.lastrowid
        return {
            "saved": True,
            "id": pid,
            "platform": platform,
            "title": title,
        }
    except sqlite3.Error:
        return {"saved": False, "id": None, "platform": platform, "title": title}


def list_custom_problems(limit: int = 50) -> list[dict[str, Any]]:
    """List saved classroom problems ordered by most recent."""
    try:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT id, platform, title, statement, sample_input, expected_output, created_at "
                "FROM custom_problems ORDER BY id DESC LIMIT ?",
                (min(limit, 100),),
            ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.Error:
        return []


def log_feedback_sqlite(
    language: str,
    bug_class: str,
    algo_family: str,
    monitor_notes: str = "",
) -> bool:
    """Record feedback metrics in SQLite (non-fatal on read-only FS)."""
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO monitor_feedback (language, bug_class, algo_family, monitor_notes, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (language, bug_class, algo_family, monitor_notes, int(time.time())),
            )
        return True
    except sqlite3.Error:
        return False
