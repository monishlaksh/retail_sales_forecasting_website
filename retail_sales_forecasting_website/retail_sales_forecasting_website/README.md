# Retail Sales Forecasting Website

A local Streamlit web app to compare XGBoost, LSTM and ARIMA on the Kaggle Store Sales dataset.

## 1. Install Python
Use Python 3.10 or 3.11 for broad TensorFlow compatibility. Install Jupyter or use a terminal.

## 2. Extract this folder
Keep `app.py` and `requirements.txt` in the same folder.

## 3. Create and activate a virtual environment (recommended)

Windows PowerShell:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:
```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 4. Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

TensorFlow is the largest dependency and may take several minutes. If its installation fails, use a Python version supported by the TensorFlow release for your operating system.

## 5. Run the website
From the folder containing `app.py`:
```bash
streamlit run app.py
```
Streamlit will open the app in your browser, usually at http://localhost:8501.

## 6. Use it
1. Download `train.csv` from https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data
2. Upload `train.csv` in the website.
3. Choose forecast/test settings and models in the sidebar.
4. Click **Train models and compare**.
5. Review the metrics and charts; download forecast and metric CSV files.

## Notes
- The app aggregates sales across all stores and product families into one daily series.
- The app evaluates on a chronological holdout set; it does not shuffle dates.
- The forecast horizon is the number of days in the holdout period. The sidebar's forecast horizon is not used for the holdout comparison in this prototype; use the test split to control the evaluation window.
- LSTM may be slow on CPU.
- This is a local development app, not a deployed public website.
