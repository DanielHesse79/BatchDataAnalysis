@echo off
REM Double-clickable entry point for install.ps1.
REM
REM PowerShell scripts do not run on a double click under the default execution
REM policy, so this wrapper calls the installer with a bypass scoped to this one
REM process. Nothing about the machine's policy is changed.

setlocal
set "SCRIPT_DIR=%~dp0"

echo Batch Insight installer
echo.
echo   [1] Standard install
echo   [2] Standard install plus optional models (CatBoost, XGBoost, SHAP; ~500 MB more)
echo   [3] Developer install (adds the test suite)
echo   [4] Cancel
echo.

choice /C 1234 /N /M "Choose an option [1-4]: "

if errorlevel 4 goto :cancelled
if errorlevel 3 set "EXTRA_ARGS=-IncludeDev -IncludeOptional" & goto :run
if errorlevel 2 set "EXTRA_ARGS=-IncludeOptional" & goto :run
if errorlevel 1 set "EXTRA_ARGS=" & goto :run

:run
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%install.ps1" %EXTRA_ARGS%
set "INSTALL_EXIT=%ERRORLEVEL%"
echo.
if not "%INSTALL_EXIT%"=="0" (
    echo Installation failed with exit code %INSTALL_EXIT%.
    echo Read the messages above; the most common cause is a missing Python.
)
pause
exit /b %INSTALL_EXIT%

:cancelled
echo Cancelled.
exit /b 0
