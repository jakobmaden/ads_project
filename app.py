from flask import Flask, render_template, request
from joblib import dump, load
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import numpy as np
import requests
import pandas as pd
import datetime

loaded_model = load('ridge.joblib')


app = Flask(__name__)



API_KEY = '6f79dac7a70a6800e82f7c7e40ca6a52'
url = 'https://api.stlouisfed.org/fred/series/observations'

series_ids = {
    'CPI': 'CPIAUCSL',             
    'Unemployment Rate_Lag1': 'UNRATE', 
    'Interest Rate_Lag1': 'FEDFUNDS', 
    'S&P 500_Lag1': 'SP500'  
}

features = ['S&P 500_Lag1', 'Interest Rate_Lag1', 'Unemployment Rate_Lag1', 'CPI_Diff1']
latest_data = {}

def fetch_latest_observation(series_id):
    """Fetches the most recent observation for a given FRED series ID."""
    params = {
        'api_key': API_KEY,
        'file_type': 'json',
        'series_id': series_id,
        'sort_order': 'desc',  
        'limit': 1             
    }


    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status() 
    data = response.json()

    if data.get('observations') and data['observations'][0].get('value') != '.':

        date_str = data['observations'][0]['date']
        value_str = data['observations'][0]['value']
        
        # Convert to appropriate types
        date_obj = pd.to_datetime(date_str)
        latest_value = pd.to_numeric(value_str, errors='coerce') 
        
        return date_obj, latest_value



for label, series_id in series_ids.items():
    date, value = fetch_latest_observation(series_id)
    
    if value is not None:
        latest_data[label] = [value]
        
        if 'Date' not in latest_data:
            latest_data['Date'] = [date]
        if label == 'CPI':
          latest_cpi_value = value

df_latest = pd.DataFrame(latest_data)
df_latest = df_latest.set_index('Date')


val = round(latest_cpi_value,0)
cur_val = int(val)
DYNAMIC_MIN = int(val)-25
DYNAMIC_MAX = int(val) + 25

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html',
                           min_val=DYNAMIC_MIN,
                           max_val=DYNAMIC_MAX,
                           current_value=cur_val)


@app.route('/process_slider', methods=['POST'])
def process_slider():

    try:
        slider_value = float(request.form.get('slider_input'))
    except (ValueError, TypeError):
        return "Error: Invalid slider value received.", 400
    
    df_prediction = df_latest.copy()
    cpi_diff_calculated = slider_value - cur_val
    df_prediction['CPI_Diff1'] = cpi_diff_calculated
    df_prediction = df_prediction[features]


    result = round(loaded_model.predict(df_prediction)[0],2)


    return render_template('result.html', 
                           input_value=slider_value, 
                           calculated_result=result)

if __name__ == '__main__':
    app.run(debug=True)