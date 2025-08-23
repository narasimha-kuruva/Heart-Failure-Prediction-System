import pandas as pd
import numpy as np

def generate_heart_failure_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'age': np.random.randint(40, 95, n_samples),
        'sex': np.random.choice([0, 1], n_samples),  # 0: Female, 1: Male
        'chest_pain_type': np.random.choice([1, 2, 3, 4], n_samples),
        'resting_bp': np.random.randint(90, 200, n_samples),
        'cholesterol': np.random.randint(120, 400, n_samples),
        'fasting_bs': np.random.choice([0, 1], n_samples),
        'resting_ecg': np.random.choice([0, 1, 2], n_samples),
        'max_hr': np.random.randint(60, 200, n_samples),
        'exercise_angina': np.random.choice([0, 1], n_samples),
        'oldpeak': np.round(np.random.uniform(0, 6, n_samples), 1),
        'st_slope': np.random.choice([0, 1, 2], n_samples),
        'heart_disease': np.zeros(n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Simple rule-based heart disease simulation
    for i in range(n_samples):
        risk_score = (
            (df.loc[i, 'age'] > 60) * 0.3 +
            (df.loc[i, 'resting_bp'] > 140) * 0.2 +
            (df.loc[i, 'cholesterol'] > 240) * 0.2 +
            (df.loc[i, 'fasting_bs'] == 1) * 0.15 +
            (df.loc[i, 'exercise_angina'] == 1) * 0.15
        )
        df.loc[i, 'heart_disease'] = 1 if risk_score > 0.5 else 0
    
    df.to_csv('heart_failure_data.csv', index=False)
    return df

if __name__ == "__main__":
    generate_heart_failure_data()