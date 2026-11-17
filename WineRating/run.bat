@echo off
cd /d "%~dp0"
echo Starting app...
echo Installing required packages...
python -m pip install --upgrade pip >nul 2>&1
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.port 8501
pause
