"""F1: home pin resolution and the single-home refusal (offline).

Covers ``src.config.paths.get_runtime_root`` precedence
(explicit ``config_path`` > ``VIBE_TRADING_HOME`` > pin file > ``Path.home()``)
and ``src.watchlist.home`` (marker, ``require_home``, ``HomeMismatch``). The pin
location is redirected per test via ``VIBE_TRADING_HOME_PIN`` so the repo's real
``agent/.vibe_home`` is never consulted or written.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from src.config.paths import get_runtime_root, read_pinned_home
from src.config import paths as paths_mod
from src.watchlist import home as home_mod
from src.watchlist.home import (
    HOME_MARKER,
    HomeMismatch,
    ensure_marker,
    expected_home,
    home_fields,
    read_marker,
    require_home,
)

_AGENT_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _AGENT_ROOT.parent


@pytest.fixture()
def pin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the pin-file lookup at a per-test file (created empty)."""
    pin_file = tmp_path / "pin" / ".vibe_home"
    pin_file.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("VIBE_TRADING_HOME_PIN", str(pin_file))
    return pin_file


def _write_pin(pin_file: Path, target: Path) -> None:
    pin_file.write_text(str(target) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# get_runtime_root precedence
# --------------------------------------------------------------------------


def test_explicit_config_path_wins_over_everything(
    tmp_path: Path, pin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Level 1: an explicit config_path outranks env, pin and home default."""
    _write_pin(pin, tmp_path / "pinned-home")
    monkeypatch.setenv("VIBE_TRADING_HOME", str(tmp_path / "env-home"))
    config_path = tmp_path / "explicit" / "agent.json"
    assert get_runtime_root(config_path) == config_path.parent


def test_env_wins_over_pin_file(
    tmp_path: Path, pin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Level 2: VIBE_TRADING_HOME beats the pin file when both are present."""
    _write_pin(pin, tmp_path / "pinned-home")
    monkeypatch.setenv("VIBE_TRADING_HOME", str(tmp_path / "env-home"))
    assert get_runtime_root() == tmp_path / "env-home"


def test_pin_file_wins_over_home_default(
    tmp_path: Path, pin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Level 3: with no env var, the pin file supplies the home."""
    _write_pin(pin, tmp_path / "pinned-home")
    monkeypatch.delenv("VIBE_TRADING_HOME", raising=False)
    assert get_runtime_root() == tmp_path / "pinned-home"


def test_home_default_when_nothing_else_is_set(
    tmp_path: Path, pin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Level 4: no env, no pin content -> Path.home()/.vibe-trading."""
    monkeypatch.delenv("VIBE_TRADING_HOME", raising=False)
    # ``pin`` exists but is empty, so read_pinned_home() returns None.
    assert read_pinned_home() is None
    assert get_runtime_root() == Path.home() / ".vibe-trading"


def test_blank_and_missing_pin_fall_through(
    tmp_path: Path, pin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VIBE_TRADING_HOME", raising=False)
    pin.write_text("   \n", encoding="utf-8")
    assert read_pinned_home() is None
    assert get_runtime_root() == Path.home() / ".vibe-trading"
    pin.unlink()
    assert read_pinned_home() is None


def test_pin_env_override_selects_the_file_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """VIBE_TRADING_HOME_PIN moves the lookup, not the resolved home."""
    custom = tmp_path / "elsewhere" / ".vibe_home"
    custom.parent.mkdir(parents=True)
    custom.write_text(str(tmp_path / "pinned"), encoding="utf-8")
    monkeypatch.setenv("VIBE_TRADING_HOME_PIN", str(custom))
    monkeypatch.delenv("VIBE_TRADING_HOME", raising=False)
    assert read_pinned_home() == tmp_path / "pinned"


def test_pin_file_bom_is_stripped(
    tmp_path: Path, pin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A BOM (Notepad's default) must not corrupt the pinned path."""
    monkeypatch.delenv("VIBE_TRADING_HOME", raising=False)
    pin.write_bytes(b"\xef\xbb\xbf" + str(tmp_path / "pinned").encode("utf-8") + b"\n")
    assert read_pinned_home() == tmp_path / "pinned"
    assert get_runtime_root() == tmp_path / "pinned"


def test_pin_file_rejects_unc_path(pin: Path) -> None:
    """A UNC pin is rejected like the env branch rejects one."""
    pin.write_text(r"\\server\share\home", encoding="utf-8")
    with pytest.raises(ValueError, match="UNC"):
        read_pinned_home()


def test_pin_file_rejects_relative_path(pin: Path) -> None:
    """A relative pin would resolve against the CWD and split homes again."""
    pin.write_text("home/.vibe-trading\n", encoding="utf-8")
    with pytest.raises(ValueError, match="absolute"):
        read_pinned_home()


def test_pin_file_rejects_tilde_only_after_expansion(pin: Path) -> None:
    """`~` expands to an absolute path, so it stays valid."""
    pin.write_text("~/pinned-home\n", encoding="utf-8")
    assert read_pinned_home() == Path("~/pinned-home").expanduser()


def test_default_pin_path_is_the_agent_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the override the pin is ``R/agent/.vibe_home``."""
    monkeypatch.delenv("VIBE_TRADING_HOME_PIN", raising=False)
    assert paths_mod._pin_file_path() == _AGENT_ROOT / ".vibe_home"


# --------------------------------------------------------------------------
# expected_home / marker
# --------------------------------------------------------------------------


def test_expected_home_reads_the_pin(pin: Path, tmp_path: Path) -> None:
    assert expected_home() is None
    _write_pin(pin, tmp_path / "pinned-home")
    assert expected_home() == tmp_path / "pinned-home"


def test_ensure_marker_writes_only_when_called(tmp_path: Path) -> None:
    target = tmp_path / "home"
    target.mkdir()
    assert read_marker(target) is None  # nothing auto-created
    path = ensure_marker(target, created_at="2026-10-16T00:00:00+00:00")
    assert path == target / HOME_MARKER
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {"home": str(target), "created_at": "2026-10-16T00:00:00+00:00"}
    assert read_marker(target) == payload


def test_ensure_marker_refuses_a_missing_home(tmp_path: Path) -> None:
    """It must not mint a fresh home from a typo."""
    with pytest.raises(FileNotFoundError):
        ensure_marker(tmp_path / "does-not-exist")
    assert not (tmp_path / "does-not-exist").exists()


def test_read_marker_ignores_corrupt_file(tmp_path: Path) -> None:
    target = tmp_path / "home"
    target.mkdir()
    (target / HOME_MARKER).write_text("{not json", encoding="utf-8")
    assert read_marker(target) is None


# --------------------------------------------------------------------------
# require_home
# --------------------------------------------------------------------------


def test_require_home_returns_resolved_without_a_pin(
    tmp_path: Path, pin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No pin -> return the resolved home and warn; never create anything."""
    monkeypatch.delenv("VIBE_TRADING_HOME", raising=False)
    monkeypatch.setenv("VIBE_TRADING_HOME", str(tmp_path / "resolved"))
    assert require_home("test") == tmp_path / "resolved"
    assert not (tmp_path / "resolved").exists()


def test_require_home_accepts_a_matching_prepared_home(
    tmp_path: Path, pin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "home"
    (target / "watchlist").mkdir(parents=True)
    (target / "watchlist" / "watchlist.sqlite3").write_bytes(b"")
    ensure_marker(target)
    _write_pin(pin, target)
    monkeypatch.setenv("VIBE_TRADING_HOME", str(target))
    assert require_home("test") == target


def test_require_home_rejects_a_different_resolved_home(
    tmp_path: Path, pin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pinned = tmp_path / "pinned-home"
    resolved = tmp_path / "other-home"
    (pinned / "watchlist").mkdir(parents=True)
    (pinned / "watchlist" / "watchlist.sqlite3").write_bytes(b"")
    ensure_marker(pinned)
    _write_pin(pin, pinned)
    monkeypatch.setenv("VIBE_TRADING_HOME", str(resolved))

    with pytest.raises(HomeMismatch) as excinfo:
        require_home("cli list")
    message = str(excinfo.value)
    # Names the expected home, the resolved home and the fix.
    assert str(pinned) in message
    assert str(resolved) in message
    assert "serve-8765.sh" in message and "VIBE_TRADING_HOME" in message
    assert "cli list" in message
    # The refusal must not have created the bogus home.
    assert not resolved.exists()


def test_require_home_rejects_a_home_without_the_marker(
    tmp_path: Path, pin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "home"
    (target / "watchlist").mkdir(parents=True)
    (target / "watchlist" / "watchlist.sqlite3").write_bytes(b"")
    _write_pin(pin, target)
    monkeypatch.setenv("VIBE_TRADING_HOME", str(target))
    with pytest.raises(HomeMismatch) as excinfo:
        require_home("cli list")
    assert HOME_MARKER in str(excinfo.value)


def test_require_home_rejects_a_home_without_the_store(
    tmp_path: Path, pin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "home"
    target.mkdir()
    ensure_marker(target)
    _write_pin(pin, target)
    monkeypatch.setenv("VIBE_TRADING_HOME", str(target))
    with pytest.raises(HomeMismatch) as excinfo:
        require_home("cli list")
    assert "watchlist.sqlite3" in str(excinfo.value)


def test_require_home_never_creates_files_or_dirs(
    tmp_path: Path, pin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even the refusal path must leave the filesystem untouched."""
    pinned = tmp_path / "pinned-home"  # deliberately never created
    resolved = tmp_path / "resolved-home"  # deliberately never created
    _write_pin(pin, pinned)
    monkeypatch.setenv("VIBE_TRADING_HOME", str(resolved))
    before = sorted(p.name for p in tmp_path.iterdir())
    with pytest.raises(HomeMismatch):
        require_home("cli list")
    assert sorted(p.name for p in tmp_path.iterdir()) == before
    assert not pinned.exists() and not resolved.exists()


def test_home_fields_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIBE_TRADING_HOME", str(tmp_path / "resolved"))
    fields = home_fields()
    assert fields == {"home": str(tmp_path / "resolved"), "pid": home_mod.os.getpid()}


# --------------------------------------------------------------------------
# pin file hygiene: git-ignored, never committed
# --------------------------------------------------------------------------


def test_pin_file_is_git_ignored() -> None:
    """`agent/.vibe_home` must be ignored by git (F1, boss condition)."""
    result = subprocess.run(
        ["git", "check-ignore", "-v", "agent/.vibe_home"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "agent/.vibe_home is not git-ignored; add a '.vibe_home' line to "
        f"agent/.gitignore (stdout={result.stdout!r} stderr={result.stderr!r})"
    )
    assert ".vibe_home" in result.stdout


def test_pin_file_never_shows_in_git_status() -> None:
    """A stray pin file must not appear as untracked in `git status`."""
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert status.returncode == 0
    assert ".vibe_home" not in status.stdout
