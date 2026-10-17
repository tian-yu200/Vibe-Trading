"""SQLite kline cache under ``get_runtime_root()/market_scanner/cache/klines.sqlite3``.

Schema (versioned):
    kline_bars(code TEXT, date TEXT, open REAL, high REAL, low REAL,
               close REAL, volume REAL, amount REAL,
               PRIMARY KEY(code, date))            -- volume in LOTS, amount CNY
    kline_meta(code TEXT PRIMARY KEY, last_date TEXT, fetched_at TEXT, source TEXT)

Volume unit convention: Tencent qfq volume is lots (100 shares) and Sina
snapshot volume is shares — :func:`append_snapshot_bar` normalises whatever
the snapshot gives to lots before writing.
"""

from __future__ import annotations

import datetime as _dt
import sqlite3
from pathlib import Path
from typing import Any

from src.config.paths import get_runtime_root

SCHEMA_VERSION = 1

# A cached history older than this is considered stale -> full refetch.
MAX_AGE_DAYS = 30


def require_cache_home(context: str) -> Path:
    """Home pin guard for cache opens (F1).

    Delegates to pi's ``src.watchlist.home.require_home`` when it has landed;
    until then falls back to the resolved runtime root so tests and unpinned
    checkouts keep working. Never creates directories or files itself.
    """
    try:
        from src.watchlist.home import require_home as _pi_require_home
    except ImportError:
        return get_runtime_root()
    return _pi_require_home(context)


def home_fields() -> dict[str, Any]:
    """``{"home": str, "pid": int}`` for logs and run rows (F1)."""
    try:
        from src.watchlist.home import home_fields as _pi_home_fields
    except ImportError:
        import os

        return {"home": str(get_runtime_root()), "pid": os.getpid()}
    return _pi_home_fields()


def _dump_ts() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


class KlineCache:
    """SQLite cache of per-code daily bars + freshness metadata."""

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            # F1: refuse to open the default cache in the wrong home (or an
            # unmarked one) instead of silently creating a fresh empty cache.
            require_cache_home("kline cache open")
            path = (
                get_runtime_root() / "market_scanner" / "cache" / "klines.sqlite3"
            )
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self) -> None:
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS kline_bars (
                    code TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    amount REAL,
                    PRIMARY KEY (code, date)
                );
                CREATE TABLE IF NOT EXISTS kline_meta (
                    code TEXT PRIMARY KEY,
                    last_date TEXT,
                    fetched_at TEXT,
                    source TEXT
                );
            """)
            db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    # -- reads ---------------------------------------------------------------
    def get_frame(self, code: str) -> dict[str, Any] | None:
        """``{'bars': DataFrame, 'meta': {...}}`` for one code, or None."""
        import pandas as pd

        with self._connect() as db:
            meta = db.execute(
                "SELECT * FROM kline_meta WHERE code = ?", (code,)
            ).fetchone()
            if meta is None:
                return None
            rows = db.execute(
                "SELECT date, open, high, low, close, volume, amount "
                "FROM kline_bars WHERE code = ? ORDER BY date",
                (code,),
            ).fetchall()
        if not rows:
            return None
        df = pd.DataFrame(
            [dict(r) for r in rows], columns=[
                "date", "open", "high", "low", "close", "volume", "amount",
            ]
        )
        df["date"] = pd.to_datetime(df["date"])
        return {
            "bars": df,
            "meta": {
                "last_date": meta["last_date"],
                "fetched_at": meta["fetched_at"],
                "source": meta["source"],
            },
        }

    def last_bar(self, code: str) -> dict[str, Any] | None:
        """The newest cached bar for one code, or None."""
        with self._connect() as db:
            row = db.execute(
                "SELECT date, open, high, low, close, volume, amount "
                "FROM kline_bars WHERE code = ? ORDER BY date DESC LIMIT 1",
                (code,),
            ).fetchone()
        return dict(row) if row else None

    def age_days(self, code: str, now: _dt.datetime | None = None) -> float | None:
        """Days since the code was fetched, or None when absent."""
        with self._connect() as db:
            row = db.execute(
                "SELECT fetched_at FROM kline_meta WHERE code = ?", (code,)
            ).fetchone()
        if row is None or not row["fetched_at"]:
            return None
        try:
            fetched = _dt.datetime.fromisoformat(row["fetched_at"])
        except ValueError:
            return None
        now = now or _dt.datetime.now()
        return (now - fetched).total_seconds() / 86400.0

    def meta_map(self) -> dict[str, dict[str, Any]]:
        """``{code: {last_date, fetched_at, source}}`` for every meta row.

        One query (no bar reads) — the F5 shortfall assessment runs on this,
        so scans do not pay per-symbol frame loads before fetching.
        Includes the ``_missing:*`` registry rows and the ``_migration``
        marker; callers filter by ``source``.
        """
        with self._connect() as db:
            rows = db.execute(
                "SELECT code, last_date, fetched_at, source FROM kline_meta"
            ).fetchall()
        return {
            row["code"]: {
                "last_date": row["last_date"],
                "fetched_at": row["fetched_at"],
                "source": row["source"],
            }
            for row in rows
        }

    # -- writes ---------------------------------------------------------------
    # -- known-missing registry (repeated fetch failures) -------------------
    # Rows in kline_meta with code='_missing' and source='known-missing';
    # last_date holds the symbol code, fetched_at the retry-after ISO time.

    def mark_known_missing(self, symbol: str, retry_after_s: int = 86400) -> None:
        """Record a symbol whose fetch keeps failing; skip it until retry-after.

        Row: code='_missing:<code>' (kline_meta PK is code), last_date=code,
        fetched_at=retry-after ISO, source='known-missing'.

        F5d: only PERMANENT failures ("no data / delisted") are marked (24 h).
        Transport errors are never marked — the fetchers count them as
        transport errors instead, so the next run retries them.
        """
        code = symbol[2:] if len(symbol) > 6 else symbol
        retry_at = (_dt.datetime.now() + _dt.timedelta(seconds=retry_after_s)).isoformat(
            timespec="seconds")
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO kline_meta (code, last_date, fetched_at, source) "
                "VALUES (?, ?, ?, 'known-missing')",
                (f"_missing:{code}", code, retry_at),
            )

    def clear_known_missing(self, symbol: str) -> None:
        """Remove the known-missing marker (e.g. after a successful fetch)."""
        code = symbol[2:] if len(symbol) > 6 else symbol
        with self._connect() as db:
            db.execute(
                "DELETE FROM kline_meta WHERE code = ?",
                (f"_missing:{code}",),
            )

    def known_missing(self, now: "datetime | str | None" = None) -> dict[str, str]:
        """``{code: retry_after_iso}`` for markers whose retry-after is in the
        future (``now`` injectable for tests); expired markers are dropped."""
        now = (now if isinstance(now, str)
               else (now or _dt.datetime.now()).isoformat(timespec="seconds"))
        with self._connect() as db:
            rows = db.execute(
                "SELECT code, last_date, fetched_at FROM kline_meta "
                "WHERE source = 'known-missing'"
            ).fetchall()
        live: dict[str, str] = {}
        expired: list[str] = []
        for key, code, retry_at in rows:
            if (retry_at or "") > now:
                live[code] = retry_at
            else:
                expired.append(key)
        if expired:
            with self._connect() as db:
                db.executemany(
                    "DELETE FROM kline_meta WHERE code = ?", [(k,) for k in expired],
                )
        return live

    def put_frame(self, code: str, df, source: str) -> None:
        """Replace one code's cached bars wholesale (full refetch path)."""
        with self._connect() as db:
            db.execute("DELETE FROM kline_bars WHERE code = ?", (code,))
            db.executemany(
                "INSERT OR REPLACE INTO kline_bars VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        code, row["date"].strftime("%Y-%m-%d"),
                        row["open"], row["high"], row["low"], row["close"],
                        row["volume"], row["amount"],
                    )
                    for _, row in df.iterrows()
                ],
            )
            last_date = df["date"].max().strftime("%Y-%m-%d") if len(df) else None
            db.execute(
                "INSERT OR REPLACE INTO kline_meta VALUES (?, ?, ?, ?)",
                (code, last_date, _dump_ts(), source),
            )

    def append_bar(self, code: str, bar: dict[str, Any], source: str) -> None:
        """Append (or overwrite) one bar and bump the meta last_date."""
        self.append_bars([(code, bar)], source)

    def append_bars(self, items: list[tuple[str, dict[str, Any]]],
                    source: str) -> int:
        """Append many bars in ONE transaction (F10 batched appends).

        Args:
            items: ``(code, bar)`` pairs; bar holds ``date`` (str or
                datetime-like) plus ``open/high/low/close/volume/amount``.
            source: Meta source tag (e.g. ``"snapshot"``).

        Returns:
            Number of bars written.
        """
        if not items:
            return 0
        rows: list[tuple] = []
        metas: list[tuple] = []
        dumped_at = _dump_ts()
        for code, bar in items:
            bar_date = bar["date"]
            if hasattr(bar_date, "strftime"):
                bar_date = bar_date.strftime("%Y-%m-%d")
            else:
                bar_date = str(bar_date)[:10]
            rows.append((
                code, bar_date, bar["open"], bar["high"], bar["low"],
                bar["close"], bar["volume"], bar["amount"],
            ))
            metas.append((code, bar_date, dumped_at, source))
        with self._connect() as db:
            db.executemany(
                "INSERT OR REPLACE INTO kline_bars VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            db.executemany(
                "INSERT OR REPLACE INTO kline_meta VALUES (?, ?, ?, ?)",
                metas,
            )
        return len(rows)

    # -- bulk light reads (F10: phase-1 decisions without full frames) -----
    def last_bars(self, codes: list[str] | None = None
                  ) -> dict[str, dict[str, Any]]:
        """Newest bar per code: ``{code: {date, close, volume}}``.

        One query for any number of codes (``None`` = whole cache).
        """
        if codes:
            query = (
                "SELECT b.code, b.date, b.close, b.volume FROM kline_bars b "
                "INNER JOIN (SELECT code, MAX(date) AS maxd FROM kline_bars "
                "WHERE code IN (%s) GROUP BY code) m "
                "ON b.code = m.code AND b.date = m.maxd"
                % ", ".join("?" * len(codes))
            )
            params: list = list(codes)
        else:
            query = (
                "SELECT b.code, b.date, b.close, b.volume FROM kline_bars b "
                "INNER JOIN (SELECT code, MAX(date) AS maxd FROM kline_bars "
                "GROUP BY code) m ON b.code = m.code AND b.date = m.maxd"
            )
            params = []
        with self._connect() as db:
            rows = db.execute(query, params).fetchall()
        return {
            row["code"]: {"date": row["date"], "close": row["close"],
                          "volume": row["volume"]}
            for row in rows
        }

    def load_frames(self, codes: list[str] | None = None,
                    cutoff: str | None = None) -> dict[str, Any]:
        """Full cached frames for ``codes`` in one query + one pandas pass.

        Returns ``{code: {"bars": DataFrame, "meta": {...}}}`` (same shape
        as :meth:`get_frame`; absent codes are skipped). ``cutoff`` (ISO
        date) drops bars newer than the latest completed session, so legacy
        intraday rows never re-enter a run. Callers needing only decisions
        should prefer :meth:`meta_map`/:meth:`last_bars` instead.
        """
        import pandas as pd

        if codes is not None and not codes:
            return {}
        with self._connect() as db:
            if codes is None:
                meta_rows = db.execute(
                    "SELECT code, last_date, fetched_at, source "
                    "FROM kline_meta").fetchall()
                bar_rows = db.execute(
                    "SELECT code, date, open, high, low, close, volume, "
                    "amount FROM kline_bars ORDER BY code, date").fetchall()
            else:
                placeholders = ", ".join("?" * len(codes))
                meta_rows = db.execute(
                    "SELECT code, last_date, fetched_at, source "
                    "FROM kline_meta WHERE code IN (%s)" % placeholders,
                    list(codes),
                ).fetchall()
                bar_rows = db.execute(
                    "SELECT code, date, open, high, low, close, volume, "
                    "amount FROM kline_bars WHERE code IN (%s) "
                    "ORDER BY code, date" % placeholders,
                    list(codes),
                ).fetchall()
        meta = {r["code"]: {"last_date": r["last_date"],
                            "fetched_at": r["fetched_at"],
                            "source": r["source"]} for r in meta_rows}
        if not bar_rows:
            return {}
        big = pd.DataFrame(
            [dict(r) for r in bar_rows],
            columns=["code", "date", "open", "high", "low", "close",
                     "volume", "amount"],
        )
        big["date"] = pd.to_datetime(big["date"])
        if cutoff is not None:
            big = big[big["date"].dt.date <= _dt.date.fromisoformat(cutoff)]
        out: dict[str, Any] = {}
        for code, group in big.groupby("code", sort=False):
            df = group.drop(columns=["code"]).reset_index(drop=True)
            out[str(code)] = {"bars": df, "meta": meta.get(str(code), {})}
        return out
