@echo off
rem Double-click entry point for start.ps1 (avoids the PowerShell execution policy).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"
if errorlevel 1 pause
