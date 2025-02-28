import pandas as pd
import numpy as np

def load_data(filepath):
    """
    Load the raw weather data from a file
    """
    return pd.read_csv(filepath)

def handle_missing_values(df):
    """
    Handle missing values in the dataset
    """
    # Fill numeric columns with median
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    
    # Fill categorical columns with mode
    categorical_columns = df.select_dtypes(include=['object']).columns
    df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])
    
    return df

def scale_features(df):
    """
    Scale numerical features
    """
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df, scaler