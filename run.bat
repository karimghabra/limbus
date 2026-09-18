@echo off
REM Launch the camera recorder (Windows equivalent of run.sh)
"%~dp0venv\Scripts\pythonw.exe" "%~dp0camera_recorder.py" %*
