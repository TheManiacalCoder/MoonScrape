@echo off
echo [NUCLEAR RESET INITIATED]

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

:: Force hard reset of LLM configuration
echo [Reinitializing Atomic Database Structure]

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

:: Reinitialize with forced presidential context
python -c "from storage.database_manager import DatabaseManager; db = DatabaseManager(); db.conn.commit();"
echo Database reset complete

:: Add line 85 to purge all analysis versions
del /f /q "analysis\aggregated_analysis.*" 2> nul

echo [System Sterilization Complete]
pause

:: Remove lines 66-68 (problematic dependency purge)
:: Replace with clean uninstall process
echo Reinstalling core dependencies...
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --force-reinstall
echo Clean dependency refresh complete 