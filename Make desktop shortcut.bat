@echo off
:: Puts an "Ocean Spectrometer Live" shortcut (with the app's icon) on this user's desktop.
:: Optional argument: a different folder to put the shortcut in.
set "OCEAN_DIR=%~dp0"
set "OCEAN_DEST=%~1"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$d = $env:OCEAN_DIR.TrimEnd('\');" ^
  "$desk = if ($env:OCEAN_DEST) { $env:OCEAN_DEST } else { [Environment]::GetFolderPath('Desktop') };" ^
  "$s = (New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $desk 'Ocean Spectrometer Live.lnk'));" ^
  "$s.TargetPath = Join-Path $d 'Ocean Spectrometer Live.bat';" ^
  "$s.WorkingDirectory = $d;" ^
  "$s.IconLocation = (Join-Path $d 'ocean_live.ico') + ',0';" ^
  "$s.WindowStyle = 7;" ^
  "$s.Description = 'Live spectra from an Ocean Optics spectrometer';" ^
  "$s.Save(); Write-Host ('Shortcut created: ' + $s.FullName)"
echo.
pause
