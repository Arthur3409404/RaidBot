@echo off
setlocal

set "BATCHPATH=%~dp0"
set "ENVNAME=RaidEnv"
set "ENVFILE=%BATCHPATH%data\config\env.yml"

for /f "delims=" %%I in ('where conda 2^>nul') do if not defined CONDA_CMD set "CONDA_CMD=%%I"
if defined CONDA_CMD (
    rem Found Conda on PATH.
) else if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" (
    set "CONDA_CMD=%USERPROFILE%\anaconda3\condabin\conda.bat"
) else if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" (
    set "CONDA_CMD=%USERPROFILE%\miniconda3\condabin\conda.bat"
) else (
    echo Conda was not found.
    echo Install Anaconda or Miniconda, then run this file again.
    pause
    exit /b 1
)

call "%CONDA_CMD%" env list | findstr /R /C:"^%ENVNAME%[ ]" >nul
if %ERRORLEVEL% NEQ 0 (
    echo Environment "%ENVNAME%" not found.
    echo Creating it from "%ENVFILE%". This can take several minutes.
    pause
    call "%CONDA_CMD%" env create --file "%ENVFILE%"
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to create "%ENVNAME%".
        pause
        exit /b %ERRORLEVEL%
    )
)

call "%CONDA_CMD%" run --no-capture-output -n "%ENVNAME%" python "%BATCHPATH%Raid_Bot.py"
pause
