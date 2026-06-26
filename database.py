import asyncio
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "chatbot.db"


def get_db():
    """New DB connection with FK, WAL, and busy timeout."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def init_db():
    """Create tables if absent and apply lightweight migrations."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS chats (
                id         TEXT PRIMARY KEY,
                title      TEXT NOT NULL DEFAULT 'New Chat',
                model      TEXT,
                starred    INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id          TEXT PRIMARY KEY,
                chat_id     TEXT NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                attachments TEXT,
                model       TEXT,
                FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
            );
        """)
        # Migration: add attachments column if missing.
        try:
            conn.execute("ALTER TABLE messages ADD COLUMN attachments TEXT")
        except sqlite3.OperationalError:
            pass
        # Migration: add model column if missing.
        try:
            conn.execute("ALTER TABLE messages ADD COLUMN model TEXT")
        except sqlite3.OperationalError:
            pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def db_execute(query: str, params: tuple | list = (), fetch: str = "none"):
    """Execute a query and fetch 'all', 'one', or 'none'. Returns dicts."""
    def _work():
        with get_db() as conn:
            res = conn.execute(query, params)
            if fetch == "all": return [dict(r) for r in res.fetchall()]
            if fetch == "one":
                row = res.fetchone()
                return dict(row) if row else None
            return None
    return await asyncio.to_thread(_work)


async def run_db_task(func):
    """Run a custom function in a background thread, passing it a managed DB connection."""
    def _work():
        with get_db() as conn:
            return func(conn)
    return await asyncio.to_thread(_work)
