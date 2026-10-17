"""FIX_CONTRACT F1 consumer-side regressions (hermes scope).

The home check is pi's module (`src/watchlist/home.py`). Until it lands, the
service must import it lazily and no-op when absent; once present, a
`HomeMismatch` from `require_home` must propagate out of `WatchlistService()`
and the API must answer 503 with the message verbatim.

The tests inject a fake `src.watchlist.home` module into sys.modules so they
pass both before and after pi's file exists.

Run: pytest tests/test_watchlist_home_check.py -q
"""

from __future__ import annotations

import datetime as _dt
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, ".")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.watchlist.store import WatchlistStore  # noqa: E402


class FakeHomeMismatch(RuntimeError):
    pass


def _install_fake_home(monkeypatch, *, raise_mismatch: bool):
    """Register a fake src.watchlist.home with the F1 surface."""
    mod = types.ModuleType("src.watchlist.home")
    calls: list[str] = []

    class HomeMismatch(FakeHomeMismatch):
        pass

    def require_home(context: str) -> Path:
        calls.append(context)
        if raise_mismatch:
            raise HomeMismatch(
                "Vibe-Trading watchlist refuses to run: this process resolved "
                "home C:\\Users\\x\\.vibe-trading but the pinned home is "
                "E:\\vibe-trading\\home\\.vibe-trading. Start via serve-8765.sh "
                "/ source env.sh, or set VIBE_TRADING_HOME."
            )
        return Path("E:/vibe-trading/home/.vibe-trading")

    def home_fields() -> dict:
        return {"home": "E:\\vibe-trading\\home\\.vibe-trading", "pid": 4242}

    mod.HomeMismatch = HomeMismatch
    mod.require_home = require_home
    mod.home_fields = home_fields
    mod.expected_home = lambda: Path("E:/vibe-trading/home/.vibe-trading")
    mod.HOME_MARKER = "HOME_ID"
    monkeypatch.setitem(sys.modules, "src.watchlist.home", mod)
    return calls


def _svc(**kw):
    from src.watchlist.service import WatchlistService

    return WatchlistService(**kw)


def test_require_home_called_when_no_explicit_store(monkeypatch) -> None:
    calls = _install_fake_home(monkeypatch, raise_mismatch=False)
    svc = _svc()
    assert calls, "require_home must be called at init"
    assert svc.home == {"home": "E:\\vibe-trading\\home\\.vibe-trading", "pid": 4242}


def test_explicit_store_skips_the_home_check(monkeypatch, tmp_path) -> None:
    """Tests/tools pass their own DB path — the pin must not block them."""
    calls = _install_fake_home(monkeypatch, raise_mismatch=True)
    svc = _svc(store=WatchlistStore(path=tmp_path / "w.sqlite3"))
    assert calls == [], "an explicit store must skip require_home"
    assert svc.home is None


def test_home_mismatch_propagates_from_service_init(monkeypatch) -> None:
    _install_fake_home(monkeypatch, raise_mismatch=True)
    with pytest.raises(FakeHomeMismatch) as ei:
        _svc()
    assert "refuses to run" in str(ei.value)
    assert "E:\\vibe-trading\\home\\.vibe-trading" in str(ei.value)


def test_missing_home_module_is_a_noop(monkeypatch) -> None:
    """pi's file may not exist yet: the service must still construct."""
    import src.watchlist as pkg

    monkeypatch.delitem(sys.modules, "src.watchlist.home", raising=False)
    # Ensure a real import attempt fails (no file on disk yet).
    monkeypatch.setattr(pkg, "__path__", pkg.__path__, raising=False)
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

    def fake_import(name, *a, **k):
        if name == "src.watchlist.home":
            raise ImportError("no module named src.watchlist.home")
        return real_import(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", fake_import)
    svc = _svc()
    assert svc.home is None  # silently skipped


def test_home_mismatch_creates_nothing_in_the_wrong_home(monkeypatch, tmp_path) -> None:
    """The check must not mint directories in the home it is rejecting.

    Regression: the first implementation built WatchlistStore first (whose
    __init__ mkdirs its parent), so a mis-resolved home ended up with an
    empty `watchlist/` dir — the LIVE step D negative test caught it.
    """
    wrong = tmp_path / "wronghome"
    monkeypatch.setenv("VIBE_TRADING_HOME", str(wrong))
    _install_fake_home(monkeypatch, raise_mismatch=True)
    with pytest.raises(FakeHomeMismatch):
        _svc()
    assert not wrong.exists(), f"wrong home was created: {list(wrong.rglob('*'))}"


def test_api_maps_home_mismatch_to_503(monkeypatch) -> None:
    """The API turns HomeMismatch into HTTP 503 with the message verbatim."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api import watchlist_routes as wr

    _install_fake_home(monkeypatch, raise_mismatch=True)
    app = FastAPI()
    # Build a REAL service so require_home runs and raises HomeMismatch.
    # (Not `wr._service`: another test module may have left a fake there.)
    def real_service():
        from src.watchlist.service import WatchlistService

        return WatchlistService()

    monkeypatch.setattr(wr, "_service", real_service)
    wr.register_watchlist_routes(app)
    with TestClient(app) as c:
        r = c.get("/api/watchlist")
    assert r.status_code == 503, r.text
    assert "refuses to run" in r.json()["detail"]


def test_api_error_mapping_helper_directly(monkeypatch) -> None:
    from src.api import watchlist_routes as wr

    _install_fake_home(monkeypatch, raise_mismatch=True)
    mod = sys.modules["src.watchlist.home"]
    exc = mod.HomeMismatch("pinned home mismatch: details")
    mapped = wr._map_service_errors(exc)
    assert mapped.status_code == 503
    assert mapped.detail == "pinned home mismatch: details"
    # A generic error stays 500.
    other = wr._map_service_errors(ValueError("boom"))
    assert other.status_code == 500


def test_forget_endpoint_maps_errors(monkeypatch, tmp_path) -> None:
    """F9 API surface: 404 unknown, 409 auto, 200 manual."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api import watchlist_routes as wr
    from src.watchlist.service import WatchlistService

    store = WatchlistStore(path=tmp_path / "w.sqlite3")
    svc = WatchlistService(
        store=store,
        scan_fn=lambda cfg, **kw: None,
        now_fn=lambda: _dt.datetime(2026, 10, 16, 12),
    )
    monkeypatch.setattr(wr, "_service", lambda: svc)
    app = FastAPI()
    wr.register_watchlist_routes(app)
    with TestClient(app) as c:
        assert c.delete("/api/watchlist/items/600519/forget").status_code == 404
        svc.add_manual("600519", note="manual test")
        r = c.delete("/api/watchlist/items/600519/forget")
        assert r.status_code == 200, r.text
        assert r.json()["item"]["origin"] == "manual"
        assert store.get_item("600519") is None
        # auto row -> 409
        store.upsert_item("600520", "x", "auto", "active", "2026-10-16T12:00:00")
        assert c.delete("/api/watchlist/items/600520/forget").status_code == 409
