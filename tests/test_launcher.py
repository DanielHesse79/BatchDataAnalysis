"""Tests for the desktop launcher and the installer scripts.

The launcher is what users actually start, so a broken app path or a shortcut
pointing at a renamed file is a total failure with no error message anywhere.
"""

import socket
import subprocess
import sys
from pathlib import Path

import pytest

import launcher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGING_DIR = PROJECT_ROOT / "packaging"


def test_every_app_points_at_a_real_script():
    """A renamed entry point would fail only when a user clicks the shortcut."""
    for key, app in launcher.APPS.items():
        script = PROJECT_ROOT / app["script"]
        assert script.exists(), f"{key} points at missing {app['script']}"


def test_every_app_has_a_window_title():
    for app in launcher.APPS.values():
        assert app["title"].strip()
        assert app["width"] > 0 and app["height"] > 0


def test_find_free_port_returns_a_bindable_port():
    port = launcher.find_free_port()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", port))  # raises if the port was not actually free


def test_find_free_port_varies():
    """Two apps launched together must not be handed the same port."""
    ports = {launcher.find_free_port() for _ in range(5)}

    assert len(ports) > 1


def test_waiting_gives_up_when_the_server_dies():
    """A crashed Streamlit must surface quickly, not hang until the timeout."""
    dead = subprocess.Popen([sys.executable, "-c", "raise SystemExit(1)"])
    dead.wait()

    assert launcher.wait_until_ready(launcher.find_free_port(), dead, timeout=30) is False


def test_stopping_an_already_dead_process_is_safe():
    finished = subprocess.Popen([sys.executable, "-c", "pass"])
    finished.wait()

    launcher.stop_streamlit(finished)  # must not raise


def test_an_unknown_app_is_rejected_by_the_parser(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["launcher.py", "--app", "nonexistent"])

    with pytest.raises(SystemExit):
        launcher.main()


@pytest.mark.parametrize(
    "file_name",
    ["install.ps1", "Install.cmd", "uninstall.ps1", "README.md"],
)
def test_packaging_files_are_present(file_name):
    assert (PACKAGING_DIR / file_name).exists()


def test_the_installer_references_the_real_requirement_files():
    """The installer names requirement files as strings; they must exist."""
    installer = (PACKAGING_DIR / "install.ps1").read_text(encoding="utf-8")

    for required in ["requirements.txt", "requirements-desktop.txt"]:
        assert required in installer
        assert (PROJECT_ROOT / required).exists()


def test_the_installer_only_offers_supported_python_versions():
    """3.14 is excluded on purpose: scientific wheels lag a new release."""
    installer = (PACKAGING_DIR / "install.ps1").read_text(encoding="utf-8")

    assert '"3.13", "3.12", "3.11"' in installer


def test_the_installer_creates_shortcuts_for_every_app():
    installer = (PACKAGING_DIR / "install.ps1").read_text(encoding="utf-8")

    for app in launcher.APPS.values():
        assert app["title"] in installer, f"no shortcut for {app['title']}"


def test_the_desktop_requirements_pull_in_the_base_set():
    desktop = (PROJECT_ROOT / "requirements-desktop.txt").read_text(encoding="utf-8")

    assert "-r requirements.txt" in desktop
    assert "pywebview" in desktop
