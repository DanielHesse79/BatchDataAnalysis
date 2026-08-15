"""Desktop launcher for the Streamlit apps.

Streamlit is a local web server, but the user does not have to see a browser.
This starts the server on a free port, waits for it to answer, and shows it in
its own window: taskbar entry, no address bar, no tab. Closing the window stops
the server.

That is what makes it feel like an installed application, and it costs about a
hundred lines instead of a rewrite in a native toolkit.

    python launcher.py --app qc
    python launcher.py --app batch --browser
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


def is_frozen() -> bool:
    """Return whether this is running from a PyInstaller bundle."""
    return getattr(sys, "frozen", False)


def resource_root() -> Path:
    """Return the directory holding the app scripts.

    In a bundle the scripts are unpacked beside the executable rather than
    living next to this source file.
    """
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent


PROJECT_ROOT = resource_root()

APPS = {
    "batch": {
        "script": "app.py",
        "title": "Batch Insight Analyzer",
        "width": 1440,
        "height": 940,
    },
    "qc": {
        "script": "qc_intel/app.py",
        "title": "QC Intelligence Layer",
        "width": 1440,
        "height": 940,
    },
}

STARTUP_TIMEOUT_SECONDS = 90
HEALTH_POLL_SECONDS = 0.4


def find_free_port() -> int:
    """Ask the OS for a port that is currently unused."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def serve_in_process(script: Path, port: int) -> None:
    """Run the Streamlit server in this process and block.

    A frozen build cannot shell out to `python -m streamlit`, because
    sys.executable is the bundled executable rather than an interpreter. The
    server is therefore started through Streamlit's own bootstrap, in a process
    whose main thread is free to take the signal handlers it installs.

    The sequence mirrors Streamlit's own CLI: record the script path, load the
    config overrides, then run. Setting environment variables alone is not
    enough - config is resolved before they are consulted, and the server comes
    up on its default port instead of the one the launcher reserved.
    """
    from streamlit import config as streamlit_config
    from streamlit.web import bootstrap

    flag_options = {
        "server.port": port,
        "server.address": "127.0.0.1",
        "server.headless": True,
        "server.fileWatcherType": "none",
        "browser.gatherUsageStats": False,
        "global.developmentMode": False,
    }

    streamlit_config._main_script_path = str(script)
    bootstrap.load_config_options(flag_options=flag_options)
    bootstrap.run(str(script), False, [], flag_options)


def check_imports() -> int:
    """Import every shipped module and report the ones that fail.

    A frozen build only contains what PyInstaller found by following imports
    from the entry point. The application packages are shipped as files, so
    nothing imports them during analysis and a missing dependency stays hidden
    until a user opens the page that needs it. Walking them here turns that into
    a build failure instead.
    """
    import importlib
    import pkgutil

    failures = []
    for package_name in ("analysis", "ui", "utils", "qc_intel"):
        try:
            package = importlib.import_module(package_name)
        except Exception as error:  # noqa: BLE001 - reporting, not handling
            failures.append(f"{package_name}: {error}")
            continue

        for module in pkgutil.walk_packages(package.__path__, f"{package_name}."):
            if ".tests" in module.name:
                continue
            try:
                importlib.import_module(module.name)
            except Exception as error:  # noqa: BLE001 - reporting, not handling
                failures.append(f"{module.name}: {error}")

    for failure in failures:
        print(failure, file=sys.stderr)
    return 1 if failures else 0


def start_streamlit(script: Path, port: int) -> subprocess.Popen:
    """Start the Streamlit server as a child process."""
    if is_frozen():
        # Re-run this same executable in server mode. Keeping the server in its
        # own process means the window can still be closed by killing a tree.
        command = [sys.executable, "--serve", "--app-script", str(script), "--port", str(port)]
    else:
        command = [
            sys.executable, "-m", "streamlit", "run", str(script),
            "--server.port", str(port),
            "--server.address", "127.0.0.1",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
            "--server.fileWatcherType", "none",
        ]

    creation_flags = 0
    if sys.platform == "win32":
        # Own process group, so a console Ctrl+C does not race the shutdown, and
        # no console window flashes up when launched from a shortcut.
        creation_flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP

    return subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        creationflags=creation_flags,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def wait_until_ready(port: int, process: subprocess.Popen, timeout: int) -> bool:
    """Poll Streamlit's health endpoint until it answers or the server dies."""
    health_url = f"http://127.0.0.1:{port}/_stcore/health"
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(HEALTH_POLL_SECONDS)

    return False


def stop_streamlit(process: subprocess.Popen) -> None:
    """Stop the server and any workers it started."""
    if process.poll() is not None:
        return

    if sys.platform == "win32":
        # Streamlit spawns children; terminate() would leave them holding the port.
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            capture_output=True,
            check=False,
        )
    else:
        process.terminate()

    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def show_in_window(url: str, title: str, width: int, height: int) -> bool:
    """Show the app in a native window. Returns False if that is unavailable."""
    try:
        import webview
    except ImportError:
        return False

    webview.create_window(title, url, width=width, height=height, min_size=(1024, 700))
    webview.start()
    return True


def run(app_key: str, force_browser: bool) -> int:
    """Launch one app and block until its window closes."""
    app = APPS[app_key]
    script = PROJECT_ROOT / app["script"]
    if not script.exists():
        print(f"Cannot find {script}", file=sys.stderr)
        return 1

    port = find_free_port()
    process = start_streamlit(script, port)
    url = f"http://127.0.0.1:{port}"

    try:
        if not wait_until_ready(port, process, STARTUP_TIMEOUT_SECONDS):
            error_output = ""
            if process.stderr is not None:
                error_output = process.stderr.read().decode("utf-8", "replace")[-2000:]
            print(f"{app['title']} did not start.\n{error_output}", file=sys.stderr)
            return 1

        if force_browser or not show_in_window(url, app["title"], app["width"], app["height"]):
            # No native window available: fall back to the default browser and
            # wait, so closing this window still stops the server.
            webbrowser.open(url)
            print(f"{app['title']} is running at {url}")
            print("Close this window to stop it.")
            try:
                process.wait()
            except KeyboardInterrupt:
                pass
    finally:
        stop_streamlit(process)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch a Batch Insight application.")
    parser.add_argument(
        "--app", choices=sorted(APPS), default="batch",
        help="Which application to start.",
    )
    parser.add_argument(
        "--browser", action="store_true",
        help="Open in the default browser instead of an application window.",
    )
    # Internal: the frozen build re-runs itself in this mode to host the server.
    parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--app-script", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=None, help=argparse.SUPPRESS)
    # Internal: the build script runs this against the bundle it just produced.
    parser.add_argument("--check-imports", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args()

    if arguments.check_imports:
        return check_imports()

    if arguments.serve:
        script = Path(arguments.app_script) if arguments.app_script else (
            PROJECT_ROOT / APPS[arguments.app]["script"]
        )
        serve_in_process(script, arguments.port or find_free_port())
        return 0

    return run(arguments.app, arguments.browser)


if __name__ == "__main__":
    sys.exit(main())
