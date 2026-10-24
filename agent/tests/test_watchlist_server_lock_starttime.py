"""Phase (d): server.lock must record proc_start (FIX2_d_lock).

A pid alone is a weak identity: once the holder dies, the pid is handed to
an unrelated process, and ``_pid_alive(pid)`` then happily reports "another
server is running" for a lock whose owner is long gone — a server that can
never start again on that home. The fix records the holder's process
creation time (``proc_start``) next to ``pid`` and treats a holder as alive
only when the pid is alive AND its creation time still matches.

Offline only: tmp homes, no server start, no network.

Run: pytest tests/test_watchlist_server_lock_starttime.py -q
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, ".")
sys.path.insert(0, str(Path(__file__).resolve().parent))

_PID_RE = re.compile(r'"pid":\s*(\d+)')


def _home(monkeypatch, tmp_path) -> None:
    """Point the runtime root at a tmp home and reset the cached lock path."""
    from src.api import watchlist_routes as wr

    home = tmp_path / "home" / ".vibe-trading"
    monkeypatch.setenv("VIBE_TRADING_HOME", str(home))
    monkeypatch.setattr(wr, "_SERVER_LOCK_PATH", None)


def _write_lock(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_lock_records_proc_start_with_top_level_int_pid(monkeypatch, tmp_path) -> None:
    """(1) The lock we write carries proc_start, and pid stays a top-level int.

    The launcher parses ``'"pid":<digits>'`` with sed, so pid must remain a
    plain top-level integer key with no nested objects around it.
    """
    from src.api import watchlist_routes as wr
    from src.watchlist.service import _pid_start_time

    _home(monkeypatch, tmp_path)
    instance = wr._acquire_server_instance_lock(8765)
    try:
        assert instance
        lock_file = wr._server_lock_path()
        text = lock_file.read_text(encoding="utf-8")
        payload = json.loads(text)
        assert isinstance(payload["pid"], int) and not isinstance(payload["pid"], bool)
        assert payload["pid"] == os.getpid()
        assert "proc_start" in payload
        assert payload["proc_start"] == pytest.approx(
            _pid_start_time(os.getpid()), abs=2.0
        )
        # Every value in the flat object is a scalar: no nested containers.
        assert all(not isinstance(v, (dict, list)) for v in payload.values())
        # The sed-parseable form: exactly one top-level "pid":<digits> match.
        matches = _PID_RE.findall(text)
        assert len(matches) == 1 and int(matches[0]) == os.getpid()
    finally:
        wr._release_server_instance_lock()


def test_legacy_lock_without_proc_start_still_refuses(
    monkeypatch, tmp_path
) -> None:
    """(2) A live pid and no proc_start (the live server's current lock)."""
    from src.api import watchlist_routes as wr

    _home(monkeypatch, tmp_path)
    lock_file = wr._server_lock_path()
    _write_lock(
        lock_file,
        {"pid": os.getpid(), "started_at": "2026-10-19T00:00:00Z", "port": 8765},
    )
    with pytest.raises(RuntimeError) as ei:
        wr._acquire_server_instance_lock(8765)
    assert "already running" in str(ei.value)
    # Untouched: we refused, we did not break another live server's lock.
    assert json.loads(lock_file.read_text(encoding="utf-8"))["pid"] == os.getpid()


def test_recycled_pid_with_wrong_proc_start_is_stale(
    monkeypatch, tmp_path
) -> None:
    """(3) Live pid + a different proc_start = pid recycled, so break it."""
    from src.api import watchlist_routes as wr
    from src.watchlist.service import _pid_start_time

    _home(monkeypatch, tmp_path)
    lock_file = wr._server_lock_path()
    ours = _pid_start_time(os.getpid()) or 0.0
    _write_lock(
        lock_file,
        {
            "pid": os.getpid(),
            "started_at": "2000-01-01T00:00:00Z",
            "port": 8765,
            "proc_start": ours - 86400.0,  # a day earlier: not this process
        },
    )
    instance = wr._acquire_server_instance_lock(8765)
    try:
        assert instance
        payload = json.loads(lock_file.read_text(encoding="utf-8"))
        assert payload["pid"] == os.getpid()
        assert payload["proc_start"] == pytest.approx(ours, abs=2.0)
    finally:
        wr._release_server_instance_lock()


def test_dead_pid_is_stale(monkeypatch, tmp_path) -> None:
    """(4) A pid that cannot exist stays stale, with or without proc_start."""
    from src.api import watchlist_routes as wr

    _home(monkeypatch, tmp_path)
    lock_file = wr._server_lock_path()
    _write_lock(
        lock_file,
        {
            "pid": 2**28,
            "started_at": "2000-01-01T00:00:00Z",
            "port": 8765,
            "proc_start": 1000.0,
        },
    )
    instance = wr._acquire_server_instance_lock(8765)
    try:
        assert instance
        assert json.loads(lock_file.read_text(encoding="utf-8"))[
            "pid"
        ] == os.getpid()
    finally:
        wr._release_server_instance_lock()


def test_release_keeps_lock_with_different_proc_start(
    monkeypatch, tmp_path
) -> None:
    """(5) Release must not delete a lock whose proc_start is not ours."""
    from src.api import watchlist_routes as wr

    _home(monkeypatch, tmp_path)
    instance = wr._acquire_server_instance_lock(8765)
    assert instance
    lock_file = wr._server_lock_path()
    payload = json.loads(lock_file.read_text(encoding="utf-8"))
    payload["proc_start"] = payload["proc_start"] - 3600.0  # someone else's era
    lock_file.write_text(json.dumps(payload), encoding="utf-8")
    wr._release_server_instance_lock()
    assert lock_file.exists(), "release deleted a lock that is not ours"


def test_pid_start_time_semantics() -> None:
    """(6) Ours is a stable non-None value; a dead pid yields None."""
    from src.watchlist.service import _pid_start_time

    ours = _pid_start_time(os.getpid())
    assert ours is not None and ours > 0
    assert _pid_start_time(os.getpid()) == pytest.approx(ours, abs=0.5)
    assert _pid_start_time(2**28) is None
    assert _pid_start_time(-1) is None
    assert _pid_start_time(0) is None


def test_written_lock_has_exactly_one_pid_match(monkeypatch, tmp_path) -> None:
    """(7) ``'"pid":\\s*(\\d+)'`` finds exactly one match in the written file."""
    from src.api import watchlist_routes as wr

    _home(monkeypatch, tmp_path)
    instance = wr._acquire_server_instance_lock(8765)
    try:
        text = wr._server_lock_path().read_text(encoding="utf-8")
        assert _PID_RE.findall(text) == [str(os.getpid())]
    finally:
        wr._release_server_instance_lock()


def test_unreadable_proc_start_keeps_a_live_holder_alive(
    monkeypatch, tmp_path
) -> None:
    """(8) A live pid whose start time cannot be read is NOT stale.

    ``GetProcessTimes`` can be denied for a holder running as another user, so
    ``_pid_start_time`` returning ``None`` says nothing about whether the server
    is up. Treating it as a mismatch would break a live ``server.lock`` and let a
    second server start on the same home — the failure this guard exists to
    prevent. Only a *read* start time that differs by >2 s (pid recycled) is
    evidence of staleness.
    """
    import src.watchlist.service as service
    from src.api import watchlist_routes as wr

    _home(monkeypatch, tmp_path)
    lock_file = wr._server_lock_path()
    payload = {
        "pid": os.getpid(),
        "started_at": "2026-10-19T00:00:00Z",
        "port": 8765,
        "proc_start": 1_700_000_000.0,
    }
    _write_lock(lock_file, payload)
    before = lock_file.read_text(encoding="utf-8")

    monkeypatch.setattr(service, "_pid_start_time", lambda _pid: None)
    with pytest.raises(RuntimeError) as ei:
        wr._acquire_server_instance_lock(8765)
    message = str(ei.value)
    assert "already running" in message
    assert f"pid {os.getpid()}" in message
    assert "2026-10-19T00:00:00Z" in message
    # Untouched: refusing to start a second server never edits the holder's lock.
    assert lock_file.read_text(encoding="utf-8") == before
    assert json.loads(lock_file.read_text(encoding="utf-8")) == payload
