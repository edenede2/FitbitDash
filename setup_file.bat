@echo off
setlocal

REM Check if Python is available via py (Python launcher) or python command.
where py >nul 2>nul
if %ERRORLEVEL%==0 (
    set "PY_CMD=py -3"
) else (
    where python >nul 2>nul
    if %ERRORLEVEL%==0 (
        set "PY_CMD=python"
    ) else (
        echo Error: Python not found. Please install Python 3 and ensure it's in your PATH.
        exit /b 1
    )
)

REM Create the virtual environment if it doesn't exist
if not exist .venv (
    echo Creating virtual environment in .venv...
    %PY_CMD% -m venv .venv
) else (
    echo .venv already exists, skipping creation.
)

REM Activate the virtual environment
call .venv\Scripts\activate

REM Upgrade pip
echo Upgrading pip...
pip install --upgrade pip

REM Install requirements if present
if exist requirements.txt (
    echo Installing requirements from requirements.txt...
    pip install -r requirements.txt
) else (
    echo No requirements.txt found in the current directory.
    echo Skipping requirements installation.
)

echo.
echo Setup complete. To activate the environment later in a new cmd session, run:
echo call .venv\Scripts\activate

endlocal
