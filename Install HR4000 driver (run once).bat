@echo off
:: Installs Ocean Optics' WinUSB driver for the HR4000 spectrometer (USB 2457:1012).
:: The driver package is Ocean's own, WHQL-signed by Microsoft, taken from python-seabreeze's
:: os_support\windows-driver-files.zip. It only binds Windows' built-in WinUSB.sys to the HR4000.
:: Windows will ask for administrator permission (UAC) - click Yes.

net session >nul 2>&1
if errorlevel 1 (
    rem Not elevated yet: start this same file again as administrator.
    set "OCEAN_DRIVER_BAT=%~f0"
    powershell -NoProfile -Command "Start-Process -FilePath $env:OCEAN_DRIVER_BAT -Verb RunAs"
    exit /b
)

echo Installing the Ocean Optics HR4000 driver...
echo.
pnputil /add-driver "%~dp0driver\OOI_HR4000.inf" /install
echo.
echo Next: unplug the HR4000's USB cable, wait 5 seconds, and plug it back in.
echo (If it has its own power supply, unplug that for a few seconds too.)
echo Then start "Ocean Spectrometer Live".
echo.
pause
