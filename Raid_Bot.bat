@echo OFF
set BATCHPATH=%cd%
set ENVNAME=RaidEnv
set CONDAPATH= C:\Users\%USERNAME%\anaconda3
set BASE = base

if %ENVNAME%==base (set ENVPATH=%CONDAPATH%) else (set ENVPATH=%CONDAPATH%\envs\%ENVNAME%)

if not exist %CONDAPATH%\Scripts\activate.bat (
echo "Default Folder Not Found Please paste the PATH to your anaconda3 Folder"
set /p "CONDAPATH= "
set ENVPATH=%CONDAPATH%\envs\%ENVNAME%
)

if not exist %ENVPATH%\python.exe (
echo "Environment not found. Press ENTER to install the required Python Environment."
echo "The Installation does not require Admin rights and will have NO IMPACT ON GLOBAL PATH VARIABLE or PROGRAM FILES."
echo "The Environment will be INSTALLED for the USER ONLY."
echo "IF THE INSTALLATION PROCESS IS INTERRUPTED, PLEASE REINSTALL THE ENV MANUALLY --> SEE CONDA DOCS."
echo "THIS PROCESS TAKES UP TO APPROXIMATLY 5 MINUTES"
echo "PRESS ANY BUTTON TO PROCEED"
pause
set BASEPATH=%CONDAPATH%
call %CONDAPATH%\Scripts\activate.bat %BASEPATH%
call conda env create --file %BATCHPATH%\data\env.yml
call conda deactivate
)

call %CONDAPATH%\Scripts\activate.bat %ENVPATH%
python %BATCHPATH%/Raid_Bot.py
pause
call conda deactivate