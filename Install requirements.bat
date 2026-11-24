@echo off
echo Installing the Python packages Ocean Spectrometer Live needs
echo (seabreeze, pyusb, libusb-package, numpy, matplotlib). This needs internet.
echo.
set PY=
where py >nul 2>nul && set PY=py
if not defined PY where python >nul 2>nul && set PY=python
if not defined PY (
  echo Python was not found on this computer.
  echo Install Python 3.10 or newer from https://www.python.org/downloads/
  echo ^(tick "Add python.exe to PATH" in the installer^), then run this file again.
  echo.
  pause
  exit /b 1
)
%PY% -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
  echo.
  echo Installing the packages failed - see the messages above.
  pause
  exit /b 1
)
echo.
echo Done. Next:
echo   1. "Install HR4000 driver (run once).bat"  (only the first time on this computer)
echo   2. "Ocean Spectrometer Live.bat"  (or the demo version to try it without the spectrometer)
pause
