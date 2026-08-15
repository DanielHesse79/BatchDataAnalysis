# Installation and packaging

## For a user

Copy the project folder to the machine, open `packaging`, and double-click
**`Install.cmd`**. Pick an option, wait, done. Shortcuts for both applications
appear on the Desktop and in the Start Menu, and each opens in its own window.

Everything is per-user: nothing is written to Program Files, no service is
registered, and no administrator rights are needed. That matters on managed
laboratory machines where users cannot elevate.

The only prerequisite is Python 3.11, 3.12 or 3.13. The installer says so
clearly and stops if it cannot find one.

## Options

| Option | Installs | Extra size |
|---|---|---|
| 1 Standard | Both apps and the desktop window | about 1 GB |
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
.\.venv\Scripts\python.exe launcher.py --app batch
.\.venv\Scripts\python.exe launcher.py --app qc
.\.venv\Scripts\python.exe launcher.py --app qc --browser   # force a browser
```

## What this does not yet solve

**The machine still needs Python.** The installer finds it, but cannot conjure
it on a locked-down machine where the user may not install software at all. If
the target is customer machines rather than your own, that is the next problem
to solve, and there are two routes:

- **Offline wheelhouse.** Ship the dependency wheels beside the installer and
  install with `--no-index --find-links`. Removes the need for PyPI access but
  still needs a Python interpreter.
- **Frozen bundle.** PyInstaller plus an Inno Setup or WiX installer produces a
  true self-contained `.exe` with no Python prerequisite. It is the real answer
  for external distribution, and it is a genuine piece of work: the environment
  is over 1 GB, Streamlit's file layout needs explicit collection, and the
  scientific wheels bring binary dependencies that PyInstaller does not always
  find on its own. Expect a 400–700 MB installer and a slow first start.

**An unsigned installer will be flagged.** SmartScreen warns on unsigned
executables and corporate AV may quarantine them outright. For anything beyond
your own machines, budget for a code-signing certificate; it is the difference
between "Windows protected your PC" and a normal install.

**A regulated site has its own process.** On a GxP machine, installing software
is a controlled activity regardless of how good the installer is. The apps
declare themselves not-validated and read-only, which keeps them out of the
qualification path, but site IT still decides what gets installed.
