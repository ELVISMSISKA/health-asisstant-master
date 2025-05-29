import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Example: Load your historical health data
def load_data():
    # Sample historical data (you would replace this with actual data)
    data = {
        'age': [45, 56, 65, 30, 55, 40, 50, 70, 60, 35],
        'bmi': [30, 28, 35, 25, 29, 32, 31, 37, 27, 24],
        'blood_pressure': [130, 140, 160, 120, 145, 135, 150, 170, 125, 110],
        'cholesterol': [200, 220, 240, 180, 210, 230, 250, 260, 190, 170],
        'health_risk': [1, 2, 3, 0, 2, 1, 2, 3, 1, 0]  # Risk levels from 0 to 3
    }
    df = pd.DataFrame(data)
    return df

# Function to train the model
def train_model():
    df = load_data()
    X = df[['age', 'bmi', 'blood_pressure', 'cholesterol']]
    y = df['health_risk']
    
    # Scaling the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train a regression model (you could replace this with a neural network)
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    return model, scaler

# Function to predict health risk
def predict_health_risk(input_data):
    model, scaler = train_model()
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Get prediction
    prediction = model.predict(input_data_scaled)
    return prediction
