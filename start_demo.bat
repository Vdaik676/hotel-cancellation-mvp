@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo First run setup_windows.bat. See README.md.
 pause
 exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run streamlit_app.py --server.address 127.0.0.1
pause
