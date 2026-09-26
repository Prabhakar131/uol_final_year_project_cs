@echo off
rem Starts CloudIR Trainer and opens it in the browser. Run setup.bat once first.
rem If port 5000 is busy:  set CLOUDIR_PORT=5050  then  start.bat
setlocal
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe goto needsetup
if not exist .env goto needsetup

if not defined CLOUDIR_PORT set "CLOUDIR_PORT=5000"
set "URL=http://127.0.0.1:%CLOUDIR_PORT%"

rem Open the browser once the server answers.
start "" /b powershell -NoProfile -Command "for ($i = 0; $i -lt 120; $i++) { try { Invoke-WebRequest -UseBasicParsing '%URL%/api/health' -TimeoutSec 2 | Out-Null; Start-Process '%URL%'; break } catch { Start-Sleep 1 } }"

echo Starting CloudIR Trainer at %URL% (press Ctrl+C to stop)
echo If port %CLOUDIR_PORT% is busy, run: set CLOUDIR_PORT=5050  then start.bat
.venv\Scripts\python.exe app.py
pause
exit /b 0

:needsetup
echo Run setup.bat first.
pause
exit /b 1
