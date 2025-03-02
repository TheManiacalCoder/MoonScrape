@echo off
echo [STARTING FULL MODULE INSTALLATION]

:: Ensure pip is up to date
python -m pip install --upgrade pip
if errorlevel 1 (
    echo Failed to update pip
    exit /b 1
)

:: Install core dependencies from requirements.txt
echo Installing core dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install core dependencies
    exit /b 1
)

:: Install additional required packages
echo Installing additional packages...
pip install aiohttp transformers torch sqlalchemy beautifulsoup4 python-dotenv colorama requests chardet numpy pandas openrouter
if errorlevel 1 (
    echo Failed to install additional packages
    exit /b 1
)

:: Install development tools
echo Installing development tools...
pip install mypy pylint black isort
if errorlevel 1 (
    echo Failed to install development tools
    exit /b 1
)

echo [ALL MODULES INSTALLED SUCCESSFULLY]
pause 