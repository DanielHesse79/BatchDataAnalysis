# Installation and packaging

## For a user

Copy the project folder to the machine, open `packaging`, and double-click
**`Install.cmd`**. Pick an option, wait, done. The **Batch Insight** shortcut
opens a shared workspace chooser. Direct shortcuts for both analysis workspaces
also appear on the Desktop and in the Start Menu.

Everything is per-user: nothing is written to Program Files, no service is
registered, and no administrator rights are needed. That matters on managed
laboratory machines where users cannot elevate.

The only prerequisite is Python 3.11, 3.12 or 3.13. The installer says so
clearly and stops if it cannot find one.

## Options

| Option | Installs | Extra size |
|---|---|---|
| 1 Standard | Shared chooser, both workspaces and the desktop window | about 1 GB |
| 2 With optional models | Plus CatBoost, XGBoost, SHAP | about 500 MB more |
| 3 Developer | Plus pytest | small |

From PowerShell directly:

```powershell
.\packaging\install.ps1
.\packaging\install.ps1 -IncludeOptional -IncludeDev
.\packaging\install.ps1 -PythonVersion 3.13 -NoShortcuts
```

To remove it:

```powershell
.\packaging\uninstall.ps1                     # shortcuts only
.\packaging\uninstall.ps1 -RemoveEnvironment  # also delete .venv
```

The uninstaller never touches project files, databases or generated reports.

## Why it is still Streamlit

The apps are Streamlit, which is a local web server. They are not, however, a
browser experience any more: `launcher.py` starts the server on a free port,
waits for it to answer, and shows it in a native window through the operating
system's web view — Edge WebView2 on Windows. The user gets a taskbar entry, no
address bar and no tab. Closing the window stops the server and its workers.

That is about a hundred lines. Rewriting the UI in a native toolkit would cost
weeks and throw away the part of Streamlit that makes changes cheap, for a
result the user cannot distinguish from this one.

If pywebview is missing the launcher falls back to the default browser rather
than failing, so a partial install still works.

```powershell
.\.venv\Scripts\python.exe launcher.py
.\.venv\Scripts\python.exe launcher.py --app batch
.\.venv\Scripts\python.exe launcher.py --app qc
.\.venv\Scripts\python.exe launcher.py --app qc --browser   # force a browser
```

## For users who should never see a dependency

The installer above still needs Python on the machine. For colleagues who are
not developers, build the self-contained bundle instead:

```powershell
.\packaging\build_bundle.ps1 -Clean -Zip
```

That produces `dist\BatchInsight\` holding `BatchInsight.exe` and everything it
needs. Users copy the folder and double-click **Open Batch Insight.cmd** to
choose a workspace. The two direct-entry command files remain available. No
Python, no pip, no administrator rights, nothing to configure.

The build is a developer step. Users never run it.

### How the frozen build differs

A frozen application cannot start Streamlit the usual way, because
`sys.executable` is the bundled executable rather than an interpreter. The
launcher therefore re-runs itself in a hidden server mode and calls Streamlit's
own `bootstrap.run` in that process. Setting `STREAMLIT_SERVER_PORT` is not
enough: configuration is resolved before the environment is consulted, so the
server has to be configured through `load_config_options`, exactly as
Streamlit's CLI does it.

Streamlit also executes the app as a *script*, so `home.py`, `app.py`, `qc_intel/app.py`
and every package they import are shipped as real files inside the bundle
rather than only as frozen bytecode.

The bundle is built one-dir rather than one-file deliberately. One-file unpacks
several hundred megabytes to a temporary directory on every launch, which turns
a double click into a long wait.

## Remaining friction

**An unsigned build will be flagged.** SmartScreen shows "Windows protected
your PC" and corporate antivirus may quarantine it outright. The bundle ships a
READ ME explaining the warning, but for real distribution budget for a
code-signing certificate. It is the difference between a warning dialog and a
normal launch, and no amount of engineering substitutes for it.

**Consider hosting it once instead.** If the people who need this are colleagues
on one network, running a single instance on an internal machine and sending
them a URL removes installation entirely, for everyone, forever. It also gives
one place to update and one database for QC trending across the laboratory. For
that audience it is less work than distributing bundles and worth deciding
before committing to per-machine installs.

**A regulated site has its own process.** On a GxP machine, installing software
is a controlled activity regardless of how good the installer is. The apps
declare themselves not-validated and read-only, which keeps them out of the
qualification path, but site IT still decides what gets installed.

**The QC Intelligence Layer starts on example data.** An installed copy seeds
itself from the bundled example dataset and says so, in the dashboard, until
real data replaces it. Loading real data is done in the Load data tab, and the
notice disappears with the first file.
