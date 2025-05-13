# PowerShell script to install dependencies and run the app
pip install -r requirements.txt
$Env:FLASK_APP = "app.py"
$Env:FLASK_ENV = "development"
flask run --host=0.0.0.0 --port=5000
