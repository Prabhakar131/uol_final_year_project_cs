@echo off
rem One-time setup on Windows: Python environment, packages, .env and models.
rem
rem   setup.bat                 everything, including the ~20 GB of models
rem   setup.bat --skip-models   no model download now; each model downloads
rem                             the first time the app needs it
rem
rem Nothing is compiled: every package, including llama-cpp-python, installs
rem from a ready-built Windows wheel, so no C++ compiler is needed.
setlocal
cd /d "%~dp0"

rem llama-cpp-python has ready-built Windows wheels up to 0.3.19; macOS and
rem Linux build the 0.3.35 pinned in requirements.txt.
set "LLAMA_WINDOWS_VERSION=0.3.19"
set "LLAMA_WHEELS=https://abetlen.github.io/llama-cpp-python/whl/cpu"

rem The deepest installed file adds about 160 characters to this folder's path,
rem and Windows stops at 260 unless long paths are enabled.
set "REST=%~dp0"
set /a DIR_LENGTH=0
:measure
if defined REST (set "REST=%REST:~1%" & set /a DIR_LENGTH+=1 & goto measure)
reg query "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled 2>nul | find "0x1" >nul
if not errorlevel 1 goto pathok
if %DIR_LENGTH% LEQ 90 goto pathok
echo This folder's path is %DIR_LENGTH% characters long:
echo   "%~dp0"
echo Windows limits file paths to 260 characters, and the installation needs
echo about 160 of them inside this folder. Move the folder somewhere short,
echo for example C:\cloudir, and run setup.bat again.
goto fail
:pathok

rem The pinned PyTorch has no builds for Python 3.13 or newer, and the Windows
rem wheels need 64-bit Python.
set "CHECK=import struct, sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) and struct.calcsize('P') == 8 else 1)"

if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe -c "%CHECK%" >nul 2>&1 || (
        echo The existing .venv uses an unsupported Python version. Delete the .venv folder and run this again.
        goto fail
    )
    goto runtime
)

set "PY="
for %%V in (3.11 3.12 3.10) do (
    if not defined PY py -%%V -c "%CHECK%" >nul 2>&1 && set "PY=py -%%V"
)
if not defined PY (
    python -c "%CHECK%" >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo 64-bit Python 3.10, 3.11 or 3.12 is required ^(3.11 is tested^); none was found.
    echo Install Python 3.11 ^(64-bit^) from https://www.python.org/downloads/ and run this again.
    goto fail
)

echo =^> Creating the Python environment (.venv)
%PY% -m venv .venv || goto fail

:runtime
rem llama.cpp needs the Microsoft Visual C++ runtime, which most PCs already have.
if exist "%SystemRoot%\System32\msvcp140.dll" if exist "%SystemRoot%\System32\vcomp140.dll" goto install
echo =^> Installing the Microsoft Visual C++ Redistributable (needed by llama.cpp)
echo     Windows may ask for permission to install it.
powershell -NoProfile -Command "$ProgressPreference = 'SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -UseBasicParsing https://aka.ms/vs/17/release/vc_redist.x64.exe -OutFile $env:TEMP\vc_redist.x64.exe"
if exist "%TEMP%\vc_redist.x64.exe" "%TEMP%\vc_redist.x64.exe" /install /passive /norestart
if exist "%SystemRoot%\System32\msvcp140.dll" if exist "%SystemRoot%\System32\vcomp140.dll" goto install
echo.
echo The Visual C++ Redistributable could not be installed. Install it from
echo https://aka.ms/vs/17/release/vc_redist.x64.exe and run setup.bat again.
goto fail

:install
echo =^> Installing packages (ready-built, nothing is compiled)
.venv\Scripts\python.exe -m pip install --upgrade pip || goto pipfail
findstr /v /i /b /c:"llama-cpp-python" requirements.txt > "%TEMP%\cloudir-requirements.txt"
.venv\Scripts\python.exe -m pip install --only-binary=:all: -r "%TEMP%\cloudir-requirements.txt" || goto pipfail
.venv\Scripts\python.exe -m pip install --only-binary=:all: --extra-index-url %LLAMA_WHEELS% llama-cpp-python==%LLAMA_WINDOWS_VERSION% imageio-ffmpeg==0.6.0 || goto pipfail

rem Whisper decodes recorded answers with ffmpeg; start.bat puts .venv\Scripts on PATH.
if not exist .venv\Scripts\ffmpeg.exe (
    .venv\Scripts\python.exe -c "import imageio_ffmpeg, shutil; shutil.copy(imageio_ffmpeg.get_ffmpeg_exe(), r'.venv\Scripts\ffmpeg.exe')" || goto fail
)

if not exist .env (
    echo =^> Creating .env from .env.example
    copy .env.example .env >nul
)

if /i "%~1"=="--skip-models" (
    echo =^> Skipping the model download: each model ^(about 20 GB in total^) downloads the first time the app needs it
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
echo Installing the Python packages failed. The error printed above says why.
echo Common causes:
echo  - No internet connection, or a proxy or firewall blocking downloads:
echo    check the connection and run setup.bat again.
echo  - "No matching distribution found": Python must be 64-bit 3.10, 3.11 or 3.12.
echo    Delete the .venv folder, install Python 3.11 ^(64-bit^) and run this again.
echo  - A "No such file or directory" or path error: move this folder somewhere
echo    short, for example C:\cloudir, delete .venv and run setup.bat again.
:fail
echo.
echo Setup did not finish. See the message above.
pause
exit /b 1
