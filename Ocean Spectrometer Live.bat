@echo off
rem Starts Ocean Spectrometer Live without a console window.
where pyw >nul 2>nul && (start "" pyw "%~dp0app.py" %* & exit /b)
start "" pythonw "%~dp0app.py" %*
