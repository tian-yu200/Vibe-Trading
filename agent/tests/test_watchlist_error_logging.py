"""RELFIX Bug C: an error mapped to a 500 must leave a trace (boss, 2026-10-24).

Live evidence: ``GET /api/watchlist/runs?limit=20`` answered 500 on 10-22 and
``GET /api/watchlist/settings`` answered 500 on 10-23, both fine on the next
retry — and neither server log held a traceback. The only path that can produce
that is ``_service()`` raising while constructing the service and
``_map_service_errors`` turning it into a 500 silently.

Logging rules under test:
  * unexpected exception mapped to 500 -> ``logger.error(..., exc_info=exc)``;
  * ``HomeMismatch`` -> ``logger.warning``, message only, no traceback;
  * ``HTTPException`` passthrough -> nothing logged.

No LLM, no network: the service constructor is the only thing stubbed.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api import watchlist_routes  # noqa: E402
from src.watchlist.home import HomeMismatch  # noqa: E402
from src.watchlist import service as service_mod  # noqa: E402

LOGGER = "src.api.watchlist_routes"


def _records(caplog, level: int) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == LOGGER and r.levelno == level]


def _raised(exc: BaseException) -> BaseException:
    """Return the exception after an actual raise, so it carries a traceback.

    The live bug was about the trace that is missing from the log, so the test
    exceptions must be real raised exceptions, not freshly built objects (whose
    ``__traceback__`` is ``None`` and whose rendering has no frames).
    """
    try:
        raise exc
    except type(exc) as caught:
        return caught


def _home_mismatch(tmp_path: Path) -> HomeMismatch:
    return _raised(
        HomeMismatch(
            expected=tmp_path / "pinned-home" / ".vibe-trading",
            resolved=tmp_path / "resolved-home" / ".vibe-trading",
            context="watchlist api",
            reason="the pin was written by another checkout",
        )
    )


# --------------------------------------------------------------------------- #
# 1. _map_service_errors itself
# --------------------------------------------------------------------------- #
def test_unexpected_error_is_logged_as_error_with_the_exception(caplog) -> None:
    caplog.set_level(logging.ERROR, logger=LOGGER)
    boom = _raised(RuntimeError("database is locked"))

    mapped = watchlist_routes._map_service_errors(boom)

    assert mapped.status_code == 500
    records = _records(caplog, logging.ERROR)
    assert len(records) == 1
    assert records[0].exc_info is not None
    assert records[0].exc_info[1] is boom
    assert "database is locked" in records[0].getMessage()
    assert "Traceback (most recent call last)" in caplog.text


def test_route_context_reaches_the_log_line(caplog) -> None:
    caplog.set_level(logging.ERROR, logger=LOGGER)

    watchlist_routes._map_service_errors(
        _raised(ValueError("bad row")), "POST /api/watchlist/scan"
    )

    records = _records(caplog, logging.ERROR)
    assert len(records) == 1
    assert "POST /api/watchlist/scan" in records[0].getMessage()


def test_home_mismatch_warns_without_a_traceback(caplog, tmp_path) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER)
    exc = _home_mismatch(tmp_path)

    mapped = watchlist_routes._map_service_errors(exc)

    assert mapped.status_code == 503
    assert mapped.detail == str(exc)
    records = _records(caplog, logging.WARNING)
    assert len(records) == 1
    assert records[0].exc_info is None, "a configuration state is not a crash"
    assert "Traceback" not in caplog.text
    assert _records(caplog, logging.ERROR) == []


def test_http_exception_passthrough_is_not_logged(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    exc = HTTPException(status_code=404, detail="unknown run r9")

    assert watchlist_routes._map_service_errors(exc) is exc

    assert _records(caplog, logging.ERROR) == []
    assert _records(caplog, logging.WARNING) == []


# --------------------------------------------------------------------------- #
# 2. The real route: a raising service constructor is a logged 500
# --------------------------------------------------------------------------- #
@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch):
    app = FastAPI()
    # No scheduler thread in a test process; the lifespan lock stays under the
    # sandboxed home conftest already points at.
    monkeypatch.setattr(watchlist_routes, "_SCHEDULER_FACTORY", lambda: None)
    monkeypatch.setattr(watchlist_routes, "_SCHEDULER", None)
    watchlist_routes._reset_scan_state()
    watchlist_routes.register_watchlist_routes(app)
    with TestClient(app) as test_client:
        yield test_client
    watchlist_routes._reset_scan_state()


def test_service_construction_failure_gives_a_logged_500(client, caplog, monkeypatch) -> None:
    """The exact live shape: GET /runs -> 500 with nothing in the log. Now it logs."""
    caplog.set_level(logging.ERROR, logger=LOGGER)
    boom = RuntimeError("watchlist store could not be opened")

    class ExplodingService:
        def __init__(self, *args, **kwargs):
            raise boom

    monkeypatch.setattr(service_mod, "WatchlistService", ExplodingService)

    response = client.get("/api/watchlist/runs?limit=20")

    assert response.status_code == 500
    records = _records(caplog, logging.ERROR)
    assert records, "a mapped 500 must leave a log record"
    assert records[0].exc_info is not None
    assert records[0].exc_info[1] is boom
    assert "watchlist service construction" in records[0].getMessage()
    assert "Traceback (most recent call last)" in caplog.text


def test_home_mismatch_at_construction_gives_a_quiet_503(
    client, caplog, monkeypatch, tmp_path
) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER)
    exc = _home_mismatch(tmp_path)

    class MismatchedService:
        def __init__(self, *args, **kwargs):
            raise exc

    monkeypatch.setattr(service_mod, "WatchlistService", MismatchedService)

    response = client.get("/api/watchlist/settings")

    assert response.status_code == 503
    assert _records(caplog, logging.WARNING), "the 503 must be reported too"
    assert _records(caplog, logging.ERROR) == []
    assert "Traceback" not in caplog.text
