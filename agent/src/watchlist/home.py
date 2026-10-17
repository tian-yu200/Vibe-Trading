"""F1: single-home enforcement for the watchlist feature.

The pin file ``R/agent/.vibe_home`` records the one runtime home the watchlist
must use. This module turns that pin into a hard check: a process that resolves
a different home (e.g. a fresh shell without ``env.sh``, or the legacy
``~/.vibe-trading``) refuses to run instead of silently opening a second,
empty store or cache.

Nothing here ever creates directories or files on the resolution path: a check
that made the very home it is validating would defeat its purpose.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from src.config.paths import get_runtime_root, read_pinned_home

logger = logging.getLogger(__name__)

#: Marker file written into a home to attest it is the real, pinned home.
HOME_MARKER = "HOME_ID"

#: The watchlist store that must exist in a pinned home.
_STORE_RELPATH = Path("watchlist") / "watchlist.sqlite3"

_FIX_HINT = (
    "Start via serve-8765.sh / source env.sh, or set VIBE_TRADING_HOME."
)


def expected_home() -> Path | None:
    """Return the home pinned in the pin file, or ``None`` when unpinned."""
    return read_pinned_home()


def marker_path(home: Path) -> Path:
    """Return the marker file path inside ``home`` (no filesystem access)."""
    return home / HOME_MARKER


def read_marker(home: Path) -> dict | None:
    """Return the decoded marker for ``home``, or ``None`` if absent/invalid."""
    path = marker_path(home)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def ensure_marker(home: Path, *, created_at: str | None = None) -> Path:
    """Write the home marker. Called explicitly (lead runs it once on H).

    Unlike :func:`require_home`, this is the one function allowed to write.
    It never creates the home directory itself: the home must already exist,
    so a typo cannot silently mint a fresh home.
    """
    home = Path(home)
    if not home.is_dir():
        raise FileNotFoundError(
            f"cannot write the home marker: {home} is not an existing directory"
        )
    payload = {
        "home": str(home),
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
    }
    path = marker_path(home)
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


class HomeMismatch(RuntimeError):
    """The process resolved a home that is not the pinned, prepared home."""

    def __init__(
        self,
        *,
        expected: Path | None,
        resolved: Path,
        context: str,
        reason: str,
    ) -> None:
        self.expected = expected
        self.resolved = resolved
        self.context = context
        self.reason = reason
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        expected = self.expected if self.expected is not None else "(no pin)"
        return (
            f"Vibe-Trading watchlist refuses to run ({self.context}): this "
            f"process resolved home {self.resolved} but the pinned home is "
            f"{expected}. {self.reason}. {_FIX_HINT}"
        )


def require_home(context: str) -> Path:
    """Return the resolved runtime home, or raise :class:`HomeMismatch`.

    Checks, only when a pin file exists:
      * the resolved home equals the pin;
      * the resolved home carries the :data:`HOME_MARKER`;
      * the resolved home has ``watchlist/watchlist.sqlite3``.

    With no pin file it returns the resolved home and logs a warning, so other
    users of the repo keep working. It never creates directories or files.
    """
    resolved = get_runtime_root()
    pinned = expected_home()
    if pinned is None:
        logger.warning(
            "Vibe-Trading watchlist: no home pin at %s; using resolved home %s",
            "agent/.vibe_home",
            resolved,
        )
        return resolved
    if resolved != pinned:
        raise HomeMismatch(
            expected=pinned,
            resolved=resolved,
            context=context,
            reason="the resolved home differs from the pin",
        )
    if not marker_path(resolved).exists():
        raise HomeMismatch(
            expected=pinned,
            resolved=resolved,
            context=context,
            reason=f"the resolved home has no {HOME_MARKER} marker",
        )
    if not (resolved / _STORE_RELPATH).exists():
        raise HomeMismatch(
            expected=pinned,
            resolved=resolved,
            context=context,
            reason=f"the resolved home has no {_STORE_RELPATH.as_posix()}",
        )
    return resolved


def home_fields() -> dict:
    """Return ``{"home": ..., "pid": ...}`` for logs and run rows."""
    return {"home": str(get_runtime_root()), "pid": os.getpid()}
