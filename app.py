from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os

app = Flask(__name__)

# Load or train model and scaler
def load_or_train_model():
    model_file = 'heart_model.pkl'
    scaler_file = 'scaler.pkl'

    if os.path.exists(model_file) and os.path.exists(scaler_file):
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    
    # Load dataset
    df = pd.read_csv('heart_failure_data.csv')
    X = df.drop('heart_disease', axis=1)
    y = df['heart_disease']
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Save model and scaler
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    
    return model, scaler

model, scaler = load_or_train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['chest_pain_type']),
            float(request.form['resting_bp']),
            float(request.form['cholesterol']),
            float(request.form['fasting_bs']),
            float(request.form['resting_ecg']),
            float(request.form['max_hr']),
            float(request.form['exercise_angina']),
            float(request.form['oldpeak']),
            float(request.form['st_slope'])
        ]
        
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        
        result = {
            'prediction': 'High Risk' if prediction == 1 else 'Low Risk',
            'probability': round(probability * 100, 2)
        }
        
        return render_template('result.html', result=result)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)