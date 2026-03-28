@echo off
echo Installing dependencies and starting AI Hub...
pip install -r requirements.txt
streamlit run app.py
pause
