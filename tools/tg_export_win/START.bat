@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Выгрузка Telegram-канала

set "PY="
where py >nul 2>nul && py -3 -c "import sys" >nul 2>nul && set "PY=py -3"
if not defined PY python -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>nul && set "PY=python"
if not defined PY call :find_python
if not defined PY call :install_python
if not defined PY goto :no_python

%PY% -c "import telethon" >nul 2>nul && goto :run
echo Устанавливаю библиотеку telethon...
%PY% -m pip install --disable-pip-version-check -q telethon
if errorlevel 1 goto :no_pip

:run
echo.
%PY% tg_export.py %*
echo.
echo Окно можно закрыть.
pause
exit /b 0

:find_python
for %%P in ("%LOCALAPPDATA%\Programs\Python\Python313\python.exe" "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" "%ProgramFiles%\Python313\python.exe" "%ProgramFiles%\Python312\python.exe") do if exist %%P set PY="%%~P"
exit /b 0

:install_python
echo Python не найден. Устанавливаю автоматически, 1-3 минуты...
where winget >nul 2>nul || exit /b 0
winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
call :find_python
exit /b 0

:no_python
echo.
echo Не получилось поставить Python автоматически.
echo Сейчас откроется python.org: скачайте, при установке поставьте галочку "Add python.exe to PATH",
echo затем снова запустите START.bat
start "" https://www.python.org/downloads/
pause
exit /b 1

:no_pip
echo.
echo Не удалось скачать библиотеку telethon. Проверьте интернет и запустите START.bat снова.
pause
exit /b 1
