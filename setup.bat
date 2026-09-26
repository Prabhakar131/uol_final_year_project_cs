@echo off
rem One-time setup on Windows: Python environment, packages, .env and models.
rem
rem   setup.bat                 everything, including the ~20 GB of models
rem   setup.bat --skip-models   models download on first use instead
setlocal
cd /d "%~dp0"

rem The pinned PyTorch has no builds for Python 3.13 or newer.
set "CHECK=import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)"

if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe -c "%CHECK%" >nul 2>&1 || (
        echo The existing .venv uses an unsupported Python version. Delete the .venv folder and run this again.
        goto fail
    )
    goto install
)

set "PY="
for %%V in (3.11 3.12 3.10) do (
    if not defined PY py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
)
if not defined PY (
    python -c "%CHECK%" >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo Python 3.10, 3.11 or 3.12 is required ^(3.11 is tested^); none was found.
    echo Install Python 3.11 from https://www.python.org/downloads/ and run this again.
    goto fail
)

echo =^> Creating the Python environment (.venv)
%PY% -m venv .venv || goto fail

:install
echo =^> Installing packages (compiling llama-cpp-python can take several minutes)
.venv\Scripts\python.exe -m pip install --upgrade pip || goto fail
.venv\Scripts\python.exe -m pip install -r requirements.txt || goto pipfail

if not exist .env (
    echo =^> Creating .env from .env.example
    copy .env.example .env >nul
)

if /i "%~1"=="--skip-models" (
    echo =^> Skipping model downloads: each model downloads the first time it is needed
) else (
    echo =^> Downloading the models ^(about 20 GB^)
    .venv\Scripts\python.exe download_models.py || goto fail
)

echo.
echo Setup complete. Start the app with start.bat
pause
exit /b 0

:pipfail
echo.
echo Installing the packages failed. llama-cpp-python needs a C++ compiler on Windows:
echo install Visual Studio Build Tools with "Desktop development with C++" from
echo https://visualstudio.microsoft.com/visual-cpp-build-tools/ and run this again.
:fail
echo.
echo Setup did not finish. See the message above.
pause
exit /b 1
