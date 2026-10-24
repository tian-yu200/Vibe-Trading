"""Path helpers for agent-level structured config."""

from __future__ import annotations

import getpass
import os
from pathlib import Path

_DEFAULT_FILENAMES = ("agent.json", "agent.yaml", "agent.yml")

_HOME_ENV_VAR = "VIBE_TRADING_HOME"

# F1: the home pin. ``R/agent/.vibe_home`` holds one line with the absolute
# runtime-home path, so a checkout (or a fresh shell) resolves the same home
# with or without the ``VIBE_TRADING_HOME`` OS variable. ``_PIN_ENV_VAR``
# overrides the pin *file location* (not the home) so tests can redirect the
# lookup into a sandbox; production leaves it unset.
_PIN_FILENAME = ".vibe_home"
_PIN_ENV_VAR = "VIBE_TRADING_HOME_PIN"


def _pin_file_path() -> Path:
    """Return the path of the home pin file.

    Defaults to ``R/agent/.vibe_home`` (``paths.py`` lives at
    ``R/agent/src/config/paths.py``). ``VIBE_TRADING_HOME_PIN`` overrides the
    location; tests point it at a sandbox so the repo pin never leaks in.
    """
    override = os.environ.get(_PIN_ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[2] / _PIN_FILENAME


def read_pinned_home() -> Path | None:
    """Return the home path pinned in the pin file, or ``None``.

    A missing, unreadable, empty or blank pin yields ``None`` so callers fall
    through to the default home (keeps a fresh checkout working). The file is
    read as ``utf-8-sig`` so a BOM (e.g. Notepad's default) does not corrupt
    the path. A UNC or relative path is rejected the same way the
    ``VIBE_TRADING_HOME`` branch rejects them: a relative pin would resolve
    against the current directory and silently split homes again.
    """
    try:
        text = _pin_file_path().read_text(encoding="utf-8-sig").strip()
    except OSError:
        return None
    if not text:
        return None
    if text.startswith(("//", "\\\\")):
        raise ValueError(f"home pin must not be a UNC path: {text!r}")
    pinned = Path(text).expanduser()
    if not pinned.is_absolute():
        raise ValueError(
            f"home pin must be an absolute path: {text!r} "
            f"(resolved against {Path.cwd()})"
        )
    return pinned


def get_runtime_root(config_path: Path | None = None) -> Path:
    """Return the runtime root directory for user-level agent state.

    Args:
        config_path: Optional explicit config file path. When provided, the
            runtime root is derived from that file's parent directory.

    Returns:
        The directory containing the explicit structured config file when one
        is provided, otherwise the ``VIBE_TRADING_HOME`` environment override,
        otherwise the pinned home (``R/agent/.vibe_home``), otherwise the
        default ``~/.vibe-trading`` runtime root.
    """
    if config_path is not None:
        return config_path.expanduser().parent
    env_root = os.environ.get(_HOME_ENV_VAR, "").strip()
    if env_root:
        if env_root.startswith(("//", "\\\\")):
            raise ValueError(
                f"{_HOME_ENV_VAR} must not be a UNC path: {env_root!r}"
            )
        return Path(env_root).expanduser()
    pinned = read_pinned_home()
    if pinned is not None:
        return pinned
    return Path.home() / ".vibe-trading"


def get_runtime_root_or_default() -> Path:
    """Return the runtime root, or the default home root when resolution fails.

    :func:`get_runtime_root` rejects a malformed override (UNC
    ``VIBE_TRADING_HOME``, relative home pin) with ``ValueError``. Modules that
    bake a ``.env`` path into a constant at import must not break over that, so
    they use this total variant and land on ``~/.vibe-trading`` instead; a
    caller that wants the misconfiguration to surface keeps using the strict
    function.
    """
    try:
        return get_runtime_root()
    except Exception:  # noqa: BLE001 — a rejected override must never break an import
        return Path.home() / ".vibe-trading"


def _login_name() -> str | None:
    """Return the OS login name, or ``None`` when it cannot be determined.

    ``USERNAME`` first (what Windows sets for every shell, including the
    service one), then :func:`getpass.getuser`, which can raise on an
    environment with no resolvable user. ``None`` is the fail-closed signal for
    :func:`display_path`.
    """
    name = os.environ.get("USERNAME", "").strip()
    if name:
        return name
    try:
        return (getpass.getuser() or "").strip() or None
    except Exception:  # noqa: BLE001 — unresolvable user: callers fail closed
        return None


def display_path(path: Path | str) -> str:
    """Return a display form of *path* that never shows the OS login name.

    The ``~/`` collapse is only safe when the absolute form would be the leak,
    i.e. the path sits under ``Path.home()`` **and** that home actually contains
    the login name. The live server pins ``USERPROFILE`` to a project directory
    whose name has no user in it; collapsing there would print the runtime-home
    ``.env`` as ``~/.vibe-trading/.env``, which reads exactly like the old OS
    profile location — the very confusion phase (a) removes. So a pinned home
    shows its absolute path (``E:/vibe-trading/home/.vibe-trading/.env``), which
    carries no username either.

    Any absolute form still carrying the login name as a directory segment gets
    that segment replaced with ``<user>``. If the login name cannot be resolved,
    the home-relative form wins when the path is under home (the shorter,
    username-free rendering) and the absolute form otherwise.
    """
    target = Path(path)
    home = Path.home()
    login = _login_name()
    try:
        relative = target.relative_to(home)
    except ValueError:
        relative = None
    if relative is not None and (login is None or login.lower() in str(home).lower()):
        return f"~/{relative.as_posix()}"
    rendered = target.as_posix()
    if login:
        needle = login.lower()
        rendered = "/".join(
            "<user>" if part.lower() == needle else part
            for part in rendered.split("/")
        )
    return rendered


def get_sessions_dir() -> Path:
    """Return the user-level directory holding chat session records."""
    return get_runtime_root() / "sessions"


def get_runs_dir() -> Path:
    """Return the user-level directory holding run artifacts."""
    return get_runtime_root() / "runs"


def get_swarm_runs_dir() -> Path:
    """Return the user-level directory holding swarm run records."""
    return get_runtime_root() / "swarm" / "runs"


def get_uploads_dir() -> Path:
    """Return the user-level directory holding uploaded files."""
    return get_runtime_root() / "uploads"


def get_config_candidates(config_path: Path | None = None) -> list[Path]:
    """Return supported config path candidates in lookup order.

    Returns:
        Candidate config paths ordered by lookup priority. When an explicit
        config path is provided, only that path is returned.
    """
    if config_path is not None:
        return [config_path.expanduser()]
    root = get_runtime_root()
    return [root / filename for filename in _DEFAULT_FILENAMES]


def get_config_path(config_path: Path | None = None) -> Path:
    """Return the active config file path.

    Prefers the first existing candidate. If an explicit path is provided,
    returns that path directly. If no candidate exists yet, returns the
    recommended default JSON path.

    Args:
        config_path: Optional explicit config file path.

    Returns:
        The selected config file path for the current runtime context.
    """
    candidates = get_config_candidates(config_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def get_data_dir(config_path: Path | None = None) -> Path:
    """Return and create the runtime data directory derived from config path.

    Args:
        config_path: Optional explicit config file path.

    Returns:
        The directory containing the active config file. The directory is
        created when it does not already exist.
    """
    data_dir = get_config_path(config_path).parent
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_workspace_path() -> Path:
    """Return the workspace path for channel state data.

    For channel adapters that need to persist state (e.g. conversation
    references, auth tokens), this returns ``~/.vibe-trading/workspace``.
    """
    p = get_runtime_root() / "workspace"
    p.mkdir(parents=True, exist_ok=True)
    return p