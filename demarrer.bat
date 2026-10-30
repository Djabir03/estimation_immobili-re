@echo off
REM Lance l'application d'estimation immobiliere puis ouvre le navigateur.
cd /d "%~dp0"

if not exist "model\house_price_model.joblib" (
  echo Entrainement du modele...
  python train_model.py
)

start "Estimation immobiliere" python app.py
timeout /t 3 /nobreak >nul
start "" http://127.0.0.1:5000
