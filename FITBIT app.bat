@echo off

if exist .venv (
    echo Found virtual environment. Activating...
    call .venv\Scripts\activate
) else (
    echo No virtual environment found. Running with system Python...
)

python app.py
pause