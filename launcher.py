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
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

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


def start_streamlit(script: Path, port: int) -> subprocess.Popen:
    """Start Streamlit headless so this process owns the window."""
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
    arguments = parser.parse_args()
    return run(arguments.app, arguments.browser)


if __name__ == "__main__":
    sys.exit(main())
