@echo off
setlocal
cd /d "%~dp0"
set "PYTHON_EXE="
if defined MVP_PYTHON set "PYTHON_EXE=%MVP_PYTHON%"
if not defined PYTHON_EXE for /f "delims=" %%P in ('py -3.12 -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON_EXE=%%P"
if not defined PYTHON_EXE if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python312\python.exe"
if not defined PYTHON_EXE if exist "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" set "PYTHON_EXE=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if not defined PYTHON_EXE goto no_python
"%PYTHON_EXE%" -c "import sys; raise SystemExit(sys.version_info[:2] != (3,12))" >nul 2>&1
if errorlevel 1 goto no_python
echo Using Python: %PYTHON_EXE%
"%PYTHON_EXE%" -m venv .venv
if errorlevel 1 goto error
".venv\Scripts\python.exe" -c "import sys; raise SystemExit(sys.version_info[:2] != (3,12))" >nul 2>&1
if errorlevel 1 goto error
".venv\Scripts\python.exe" -m pip install -r requirements-lock.txt
if errorlevel 1 goto error
".venv\Scripts\python.exe" -c "import joblib, streamlit; joblib.load('artifacts/model.joblib')"
if errorlevel 1 goto error
echo Setup completed. Run start_demo.bat.
if not defined MVP_NO_PAUSE pause
exit /b 0
:no_python
echo Python 3.12 was not found. Install Python 3.12 x64 or set MVP_PYTHON to its python.exe path.
echo Python 3.13 is a different version and does not satisfy this package's pinned environment.
if not defined MVP_NO_PAUSE pause
exit /b 1
:error
echo Setup failed. See the error above and README.md. Internet is required for the first installation.
if not defined MVP_NO_PAUSE pause
exit /b 1
