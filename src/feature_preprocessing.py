import pandas as pd
import numpy as np

def create_weather_features(df):
    """
    Create derived weather features
    """
    # Example derived features
    if 'temperature' in df.columns and 'humidity' in df.columns:
        df['heat_index'] = calculate_heat_index(df['temperature'], df['humidity'])
    
    return df

def calculate_heat_index(temperature, humidity):
    """
    Calculate heat index based on temperature and humidity
    """
    # Simplified heat index calculation
    return temperature + (0.05 * humidity)

def encode_categorical_features(df):
    """
    Encode categorical variables
    """
    from sklearn.preprocessing import LabelEncoder
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    encoders = {}
    for column in categorical_columns:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
        encoders[column] = encoder
    
    return df, encoders