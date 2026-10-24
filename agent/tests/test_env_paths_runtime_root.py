"""Phase (a) regression: every `.env` read/write resolves the home via `get_runtime_root()`.

Before the fix, five sites baked ``Path.home()/".vibe-trading"/".env"`` at import, so a
plain shell whose OS profile is not the service home loaded (and `init` wrote) a
different file than the app — while a home moved by ``VIBE_TRADING_HOME`` or the
``agent/.vibe_home`` pin was ignored. All of them now route through
``src.config.paths.get_runtime_root()`` (precedence: explicit > ``VIBE_TRADING_HOME`` >
pin > ``Path.home()/".vibe-trading"``).

Tmp homes only. The project `.env` files and the real ``~/.vibe-trading`` are never
opened, and no env *value* is ever printed — paths only, all under ``tmp_path``.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import textwrap
from io import StringIO
from pathlib import Path

import getpass

from rich.console import Console

import src.api.helpers as helpers
import src.providers.llm as llm
import src.trading.tap_forward as tap_forward
from cli import onboard
from src.config import paths as _paths
from src.config.paths import display_path, get_runtime_root

# ``cli/__init__`` re-exports the ``main`` *callable*, which shadows the
# ``cli.main`` submodule attribute — importlib is the only reliable way in.
cli_main = importlib.import_module("cli.main")
cli_legacy = importlib.import_module("cli._legacy")

AGENT = Path(llm.__file__).resolve().parents[2]
_SITES = ("llm", "tap", "helpers_env", "main", "legacy_init")

# Runs in a fresh interpreter so the import-time bake is what is under test.
# Prints one JSON line; never touches a real home or prints a dotenv value.
_CHILD = textwrap.dedent(
    """
    import importlib, json, sys
    from pathlib import Path
    from src.config.paths import get_runtime_root

    llm = importlib.import_module("src.providers.llm")
    tap = importlib.import_module("src.trading.tap_forward")
    helpers = importlib.import_module("src.api.helpers")
    main = importlib.import_module("cli.main")
    legacy = importlib.import_module("cli._legacy")

    def _winner(paths):
        return next((str(p) for p in paths if p.exists()), None)

    data = {
        "home": str(Path.home()),
        "runtime_root": str(get_runtime_root()),
        "llm": [str(p) for p in llm._ENV_CANDIDATES],
        "llm_winner": _winner(llm._ENV_CANDIDATES),
        "tap": [str(p) for p in tap._ENV_CANDIDATES],
        "tap_winner": _winner(tap._ENV_CANDIDATES),
        "helpers_env": str(helpers.ENV_PATH),
        "main": [str(p) for p in main._env_candidates()],
        "main_winner": str(main._first_existing_env_path()) if main._first_existing_env_path() else None,
        "legacy_init": str(legacy._INIT_ENV_PATH),
    }
    sys.stdout.write("VT_RESULT:" + json.dumps(data))
    """
)


def _canon(path: str | os.PathLike[str] | None) -> str | None:
    """Return a case/separator-insensitive real path for comparison."""
    if path is None:
        return None
    return os.path.normcase(os.path.realpath(str(path)))


def _home_files(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Create a tmp service home, a *different* tmp user profile, and a cwd.

    Each carries its own marked ``.env`` so "which file won" is observable.
    """
    home = tmp_path / "home"
    runtime = home / ".vibe-trading"
    profile = tmp_path / "profile"
    cwd = tmp_path / "cwd"
    for leaf in (runtime, profile / ".vibe-trading", cwd):
        leaf.mkdir(parents=True, exist_ok=True)
    for leaf in (runtime, profile / ".vibe-trading", cwd):
        (leaf / ".env").write_text("MARKER=tmp\n", encoding="utf-8")
    return home, runtime, profile, cwd


def _resolve_fresh(
    tmp_path: Path,
    *,
    profile: Path,
    cwd: Path,
    vibe_home: Path | None = None,
    pin: Path | None = None,
) -> dict:
    """Resolve all five sites in a fresh interpreter under a tmp-home env."""
    env = dict(os.environ)
    env["USERPROFILE"] = str(profile)
    env["HOME"] = str(profile)
    env.pop("VIBE_TRADING_HOME", None)
    # An UNSET pin var is not "no pin": _pin_file_path() then falls back to the
    # repo's own agent/.vibe_home, which exists in a live checkout and would
    # resolve every one of these cases to the production home. Point the pin
    # *location* at a path that cannot exist unless the caller supplies one.
    env["VIBE_TRADING_HOME_PIN"] = str(pin or tmp_path / "_no_such_pin")
    if vibe_home is not None:
        env["VIBE_TRADING_HOME"] = str(vibe_home)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(AGENT), env.get("PYTHONPATH", "")) if part
    )
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", _CHILD],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(cwd),
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    payload = [ln for ln in proc.stdout.splitlines() if ln.startswith("VT_RESULT:")]
    assert payload, f"child printed no result line: {proc.stdout[-2000:]}"
    data = json.loads(payload[0][len("VT_RESULT:"):])
    assert set(_SITES) <= set(data)
    return data


def _site_order(data: dict) -> dict[str, list[str]]:
    """Return the ordered candidate list each site exposes."""
    return {
        "llm": data["llm"],
        "tap": data["tap"],
        "helpers_env": [data["helpers_env"]],
        "main": data["main"],
        "legacy_init": [data["legacy_init"]],
    }


def test_server_like_env_resolves_the_same_order_as_before(tmp_path: Path) -> None:
    """Acceptance (i): for the server env the change is a no-op.

    ``serve-8765.sh`` pins ``USERPROFILE`` to the service home *and* exports
    ``VIBE_TRADING_HOME`` inside it, so ``Path.home()/".vibe-trading"`` and
    ``get_runtime_root()`` are the same directory: the first slot, its order and
    the winner must be identical before and after the fix.
    """
    home, runtime, _profile, cwd = _home_files(tmp_path)
    data = _resolve_fresh(tmp_path, profile=home, cwd=cwd, vibe_home=runtime)

    # The old formula, spelled out: Path.home() / ".vibe-trading" / ".env".
    old_first = Path(data["home"]) / ".vibe-trading" / ".env"
    expected = {
        "llm": [old_first, AGENT / ".env", cwd / ".env"],
        "tap": [old_first, AGENT / ".env", cwd / ".env"],
        "helpers_env": [old_first],
        "main": [old_first, AGENT / ".env", cwd / ".env"],
        "legacy_init": [old_first],
    }
    got = _site_order(data)
    for site in _SITES:
        assert [_canon(p) for p in got[site]] == [_canon(p) for p in expected[site]], site

    # The home slot is the file that wins, exactly as it did pre-fix.
    assert _canon(data["llm_winner"]) == _canon(runtime / ".env")
    assert _canon(data["tap_winner"]) == _canon(runtime / ".env")
    assert _canon(data["main_winner"]) == _canon(runtime / ".env")
    assert _canon(data["runtime_root"]) == _canon(runtime)


def test_plain_shell_loads_the_runtime_home_not_the_user_profile(
    tmp_path: Path,
) -> None:
    """Acceptance (ii): ``VIBE_TRADING_HOME`` beats a different ``USERPROFILE``.

    The profile has its own ``.vibe-trading/.env``; the service home's file must
    win at all five places.
    """
    _home, runtime, profile, cwd = _home_files(tmp_path)
    data = _resolve_fresh(tmp_path, profile=profile, cwd=cwd, vibe_home=runtime)

    profile_env = profile / ".vibe-trading" / ".env"
    got = _site_order(data)
    for site in _SITES:
        assert _canon(got[site][0]) == _canon(runtime / ".env"), site
        assert _canon(profile_env) not in [_canon(p) for p in got[site]], site

    assert _canon(data["llm_winner"]) == _canon(runtime / ".env")
    assert _canon(data["tap_winner"]) == _canon(runtime / ".env")
    assert _canon(data["main_winner"]) == _canon(runtime / ".env")


def test_pinned_home_loads_the_runtime_home_in_a_plain_shell(tmp_path: Path) -> None:
    """Acceptance (ii) via the pin file instead of the env var."""
    _home, runtime, profile, cwd = _home_files(tmp_path)
    pin = tmp_path / "pinfile"
    pin.write_text(str(runtime), encoding="utf-8")
    data = _resolve_fresh(tmp_path, profile=profile, cwd=cwd, pin=pin)

    got = _site_order(data)
    for site in _SITES:
        assert _canon(got[site][0]) == _canon(runtime / ".env"), site
    assert _canon(data["llm_winner"]) == _canon(runtime / ".env")


def test_bare_shell_still_defaults_to_the_user_profile(tmp_path: Path) -> None:
    """No override at all keeps the pre-fix behaviour: ``~/.vibe-trading``."""
    _home, _runtime, profile, cwd = _home_files(tmp_path)
    data = _resolve_fresh(tmp_path, profile=profile, cwd=cwd)

    assert _canon(data["runtime_root"]) == _canon(profile / ".vibe-trading")
    got = _site_order(data)
    for site in _SITES:
        assert _canon(got[site][0]) == _canon(profile / ".vibe-trading" / ".env"), site


def test_sites_use_the_shared_helper() -> None:
    """The four importable sites share one resolver; only tap_forward duplicates it."""
    for module in (llm, helpers, cli_main, cli_legacy, onboard):
        assert not hasattr(module, "_runtime_home_root"), module.__name__
        assert module.get_runtime_root_or_default is _paths.get_runtime_root_or_default, (
            module.__name__
        )
    # tap_forward.py is imported without the ``src`` package guarantee, so it
    # keeps a standalone copy; it must still agree with the shared one.
    assert tap_forward._runtime_home_root() == _paths.get_runtime_root_or_default()


def test_shared_helper_follows_env_var_and_pin(monkeypatch, tmp_path: Path) -> None:
    """The single shared resolver covers control, env override and pin."""
    _home, runtime, profile, _cwd = _home_files(tmp_path)

    monkeypatch.delenv("VIBE_TRADING_HOME", raising=False)
    # Redirect the pin LOOKUP into the sandbox: deleting the var would instead
    # expose the checkout's own agent/.vibe_home (present in a live repo).
    monkeypatch.setenv("VIBE_TRADING_HOME_PIN", str(tmp_path / "_no_such_pin"))
    monkeypatch.setenv("USERPROFILE", str(profile))
    monkeypatch.setenv("HOME", str(profile))
    assert _canon(_paths.get_runtime_root_or_default()) == _canon(profile / ".vibe-trading")

    monkeypatch.setenv("VIBE_TRADING_HOME", str(runtime))
    assert _canon(_paths.get_runtime_root_or_default()) == _canon(runtime)
    assert _canon(tap_forward._runtime_home_root()) == _canon(runtime)

    monkeypatch.delenv("VIBE_TRADING_HOME")
    pin = tmp_path / "pinfile"
    pin.write_text(str(runtime), encoding="utf-8")
    monkeypatch.setenv("VIBE_TRADING_HOME_PIN", str(pin))
    assert _canon(_paths.get_runtime_root_or_default()) == _canon(runtime)
    assert _canon(tap_forward._runtime_home_root()) == _canon(runtime)


def test_module_constants_match_the_runtime_root_under_the_test_sandbox() -> None:
    """The import-time constants are built from the runtime root, not the profile."""
    env_file = _canon(get_runtime_root() / ".env")
    assert _canon(llm._ENV_CANDIDATES[0]) == env_file
    assert _canon(tap_forward._ENV_CANDIDATES[0]) == env_file
    assert _canon(helpers.ENV_PATH) == env_file
    assert _canon(cli_main._ENV_PATH) == env_file
    assert _canon(cli_legacy._INIT_ENV_PATH) == env_file
    assert _canon(onboard._env_path()) == env_file
    # tap_forward must stay consistent with llm.py, slot for slot.
    assert [_canon(p) for p in tap_forward._ENV_CANDIDATES] == [
        _canon(p) for p in llm._ENV_CANDIDATES
    ]


def test_runtime_root_failure_falls_back_to_the_default_home(monkeypatch) -> None:
    """A rejecting ``get_runtime_root()`` must never break an import."""

    def _boom(*_args, **_kwargs):
        raise ValueError("VIBE_TRADING_HOME must not be a UNC path")

    monkeypatch.setattr("src.config.paths.get_runtime_root", _boom)

    fallback = Path.home() / ".vibe-trading"
    assert _paths.get_runtime_root_or_default() == fallback
    assert tap_forward._runtime_home_root() == fallback
    assert onboard._env_dir() == fallback


def test_env_labels_stay_leak_free_constants() -> None:
    """Slot labels name the winner without echoing a path, home, or username."""
    assert len(llm._ENV_LABELS) == len(llm._ENV_CANDIDATES)
    joined = " ".join(llm._ENV_LABELS)
    assert ":" not in joined and not joined.startswith(("/", "\\"))
    assert str(Path.home()) not in joined
    assert getpass.getuser() not in joined
    for candidate in llm._ENV_CANDIDATES:
        label = llm._redact_env_source(candidate)
        assert label in llm._ENV_LABELS
        assert str(candidate) not in label
    assert llm._redact_env_source(llm._ENV_CANDIDATES[0]) == llm._ENV_LABELS[0]


# ---------------------------------------------------------------------------
# display_path — the '~' collapse happens only where it would be the leak
# ---------------------------------------------------------------------------


def _fake_user(monkeypatch, home: Path, login: str = "vtuser") -> None:
    """Pin ``Path.home()`` and the login name. Both are invented, never real."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("USERNAME", login)


def test_display_path_shows_absolute_when_the_pinned_home_has_no_username(
    monkeypatch,
) -> None:
    """(i) The server pins ``USERPROFILE`` to a project dir: no ``~/`` there.

    Collapsing would print the runtime-home file as ``~/.vibe-trading/.env``,
    which reads like the old OS-profile location — the confusion (a) removes.
    """
    home = Path("E:/vibe-trading/home")
    _fake_user(monkeypatch, home)
    rendered = display_path(home / ".vibe-trading" / ".env")
    assert rendered == "E:/vibe-trading/home/.vibe-trading/.env"
    assert rendered == (home / ".vibe-trading" / ".env").as_posix()
    assert "vtuser" not in rendered


def test_display_path_collapses_under_a_home_that_carries_the_login(
    monkeypatch,
) -> None:
    """(ii) A real profile home does leak the name, so ``~/`` replaces it."""
    home = Path("C:/Users/vtuser")
    _fake_user(monkeypatch, home)
    assert display_path(home / ".vibe-trading" / ".env") == "~/.vibe-trading/.env"


def test_display_path_fails_closed_without_a_resolvable_login(monkeypatch) -> None:
    """(iii) No login: ``~/`` under home (the shorter, safe form), else absolute."""
    home = Path("C:/Users/vtuser")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("USERNAME", raising=False)

    def _boom() -> str:
        raise RuntimeError("no such user")

    monkeypatch.setattr(_paths.getpass, "getuser", _boom)
    assert display_path(home / ".vibe-trading" / ".env") == "~/.vibe-trading/.env"
    assert display_path(Path("D:/data/.env")) == "D:/data/.env"


def test_display_path_masks_a_login_segment_outside_home(monkeypatch) -> None:
    """(iv) A path under the *old* profile while home is pinned elsewhere."""
    _fake_user(monkeypatch, Path("E:/vibe-trading/home"))
    rendered = display_path(Path("C:/Users/vtuser/.vibe-trading/.env"))
    assert rendered == "C:/Users/<user>/.vibe-trading/.env"
    assert "vtuser" not in rendered


def test_project_relative_path_shows_the_resolved_env_file(monkeypatch, tmp_path) -> None:
    """The settings API hint follows the real path instead of a fixed literal."""
    env_file = tmp_path / "svc" / ".vibe-trading" / ".env"
    monkeypatch.setattr(helpers, "ENV_PATH", env_file)
    _fake_user(monkeypatch, Path("E:/vibe-trading/home"))
    assert helpers._project_relative_path(env_file) == env_file.as_posix()


# ---------------------------------------------------------------------------
# cli/onboard.py — the wizard drafts and finalizes in the runtime home
# ---------------------------------------------------------------------------


def test_onboard_writes_the_runtime_home_not_the_user_profile(
    tmp_path, monkeypatch
) -> None:
    """``.env.partial`` and the final ``.env`` land under the resolved home."""
    runtime = tmp_path / "home" / ".vibe-trading"
    profile = tmp_path / "profile"
    runtime.mkdir(parents=True)
    (profile / ".vibe-trading").mkdir(parents=True)

    monkeypatch.delenv("VIBE_TRADING_HOME_PIN", raising=False)
    monkeypatch.setenv("VIBE_TRADING_HOME", str(runtime))
    monkeypatch.setenv("USERPROFILE", str(profile))
    monkeypatch.setenv("HOME", str(profile))

    answers = iter(["deepseek", "deepseek-v4-pro", "120", "__skip__", "__skip__"])
    monkeypatch.setattr(onboard, "_select_with_back", lambda *_a, **_k: next(answers))
    monkeypatch.setattr(onboard, "_prompt_secret", lambda *_a, **_k: "sk-fake-not-a-key")
    monkeypatch.setattr(onboard, "_prompt_text", lambda *_a, **_k: "deepseek-v4-pro")

    partials: list[Path] = []
    real_save = onboard._save_partial

    def _spy(values: dict[str, str]) -> None:
        real_save(values)
        partials.append(onboard._partial_path())

    monkeypatch.setattr(onboard, "_save_partial", _spy)

    written = onboard.run_onboarding(console=Console(file=StringIO(), width=200))

    assert partials, "the wizard never saved a partial"
    assert all(_canon(p) == _canon(runtime / ".env.partial") for p in partials)
    assert _canon(written) == _canon(runtime / ".env")
    assert (runtime / ".env").exists()
    assert not (runtime / ".env.partial").exists()
    assert list((profile / ".vibe-trading").iterdir()) == []
