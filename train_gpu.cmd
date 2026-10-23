@echo off
rem ============================================================
rem  GPU training launcher - mini-deepseek-pipeline
rem  Runs inside WSL2 (Ubuntu-24.04) with the CUDA venv.
rem  Usage (same args as run_pipeline.py):
rem    train_gpu.cmd --stage pretrain --quick --skip-tests
rem    train_gpu.cmd --help
rem ============================================================
setlocal
cd /d "%~dp0"
echo [GPU] WSL2 distro: Ubuntu-24.04, device: cuda
wsl -d Ubuntu-24.04 bash -lc "source /root/venvs/torch-cu/bin/activate && python run_pipeline.py --device cuda %*"
exit /b %errorlevel%
