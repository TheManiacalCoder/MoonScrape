@echo off
echo [NUCLEAR RESET AND INSTALLATION INITIATED]

:: Obliterate database
if exist "data\" (
    rmdir /s /q "data"
    echo Obliterated database directory
)

:: Destroy analysis artifacts
if exist "analysis\" (
    rmdir /s /q "analysis"
    echo Annihilated analysis directory
)

:: Eradicate benchmark data
del /f /q benchmark_results.txt 2> nul

:: Nuke Hugging Face cache
if exist "%USERPROFILE%\.cache\huggingface\" (
    rmdir /s /q "%USERPROFILE%\.cache\huggingface"
    echo Terminated Hugging Face cache
)

:: Purge temporary files
powershell -Command "Remove-Item -Path 'C:\tmp\moonscrape_*' -Force -Recurse -ErrorAction SilentlyContinue"
echo Purged system temp files

:: Eliminate Python cache
if exist "__pycache__\" (
    rmdir /s /q "__pycache__"
    echo Eradicated Python cache
)

:: Destroy session state
del /f /q session_state.json 2> nul

:: Add analysis file cleanup
del /f /q "analysis\*.txt" 2> nul
echo Purged analysis text files

:: Enhanced model cache purge
python -c "import torch; [f.unlink() for f in Path(torch.hub.get_dir()).glob('*') if f.is_file()]"
echo Obliterated PyTorch hub cache

:: Add complete system sterilization
del /f /q *.log *.tmp *.cache 2> nul
echo Purged all system logs and temp files

:: Enhanced analysis purge
if exist "analysis\*" (
    del /f /q "analysis\*.*"
    echo Obliterated all analysis artifacts
)

:: Nuclear model cache purge
python -c "import transformers; transformers.utils.hub.move_cache(); from pathlib import Path; [f.unlink() for f in Path(transformers.utils.hub.get_cache_dir()).rglob('*') if f.is_file()]"
echo Annihilated ALL model caches

:: Full registry cleanse
reg delete "HKCU\Software\Python" /f 2> nul
echo Purged Python registry entries

:: Add line 85 to purge all analysis versions
del /f /q "analysis\aggregated_analysis.*" 2> nul

echo [System Sterilization Complete]

:: Start installation process
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

:: Add these lines after the core dependencies installation
echo Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo Failed to install PyTorch
    exit /b 1
)

echo Installing transformers with compatibility...
pip install transformers==4.35.0
if errorlevel 1 (
    echo Failed to install transformers
    exit /b 1
)

echo Verifying installations...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

echo [ALL MODULES INSTALLED SUCCESSFULLY]
echo [RESET AND INSTALLATION COMPLETE]
pause 