@echo off
chcp 65001 >nul
cd /d "%~dp0"
call START.bat --limit 50 --no-media
