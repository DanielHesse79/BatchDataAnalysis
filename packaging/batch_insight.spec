# PyInstaller build definition for a self-contained Batch Insight bundle.
#
# Two things make a Streamlit application awkward to freeze, and both are
# handled here.
#
# Streamlit executes the app as a *script*, not as an imported module. So
# app.py and every package it imports must be shipped as real files inside the
# bundle, not only as frozen bytecode. They go in as datas.
#
# Streamlit also loads its own static assets and reads its distribution
# metadata at runtime, so collect_all is used rather than a hand-written list
# of hidden imports.
#
# Built one-dir rather than one-file on purpose: one-file unpacks several
# hundred megabytes to a temporary directory on every launch, which turns a
# double click into a thirty second wait.

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

PROJECT_ROOT = Path(SPECPATH).parent

datas = []
binaries = []
hiddenimports = []

# Streamlit, Plotly and Altair carry static assets and metadata they read at
# runtime; a partial collection produces an app that starts and then renders
# nothing.
for package in ("streamlit", "plotly", "altair", "pyarrow"):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

# Scientific stack: PyInstaller finds most of scikit-learn and SciPy, but the
# lazily imported solver and metric modules are only discovered explicitly.
hiddenimports += collect_submodules("sklearn.utils")
hiddenimports += collect_submodules("sklearn.neighbors")
hiddenimports += collect_submodules("scipy.special")
hiddenimports += collect_submodules("scipy.stats")
hiddenimports += [
    "sklearn.cross_decomposition",
    "sklearn.decomposition",
    "sklearn.ensemble",
    "sklearn.inspection",
    "scipy._lib.array_api_compat.numpy.fft",
    "reportlab.graphics.barcode.code128",
    "PIL._tkinter_finder",
]

datas += collect_data_files("reportlab")

# The application's own packages are shipped as data (below), which makes their
# imports invisible to PyInstaller: nothing in the entry point imports them, so
# their third-party dependencies are never analysed and never collected. Listing
# them as hidden imports is what pulls in the long tail - PIL.ImageDraw for the
# PDF export, and everything else reached only from app code.
def is_not_a_test_module(module_name):
    return ".tests" not in module_name


for package in ("analysis", "ui", "utils"):
    hiddenimports += collect_submodules(package, filter=is_not_a_test_module)

# The application source, shipped as files so Streamlit can run them.
for source in ("app.py", "launcher.py"):
    datas.append((str(PROJECT_ROOT / source), "."))

for package_dir in ("analysis", "ui", "utils"):
    datas.append((str(PROJECT_ROOT / package_dir), package_dir))

for document in ("README.md",):
    if (PROJECT_ROOT / document).exists():
        datas.append((str(PROJECT_ROOT / document), "."))

if (PROJECT_ROOT / "docs").exists():
    datas.append((str(PROJECT_ROOT / "docs"), "docs"))

# Data files the apps offer as examples. Kept because a first-run user with no
# data of their own has nothing to look at otherwise.
if (PROJECT_ROOT / "data").exists():
    datas.append((str(PROJECT_ROOT / "data"), "data"))


analysis_step = Analysis(
    [str(PROJECT_ROOT / "launcher.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Excluded to keep the bundle down: these are development or optional
    # extras that the shipped applications never import.
    excludes=[
        "pytest", "PyInstaller", "IPython", "jupyter", "notebook",
        "torch", "tensorflow", "matplotlib.tests", "numpy.tests",
        "pandas.tests", "scipy.tests", "sklearn.tests",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(analysis_step.pure)

executable = EXE(
    pyz,
    analysis_step.scripts,
    [],
    exclude_binaries=True,
    name="BatchInsight",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # no console window behind the app window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

collection = COLLECT(
    executable,
    analysis_step.binaries,
    analysis_step.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="BatchInsight",
)
