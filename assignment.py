import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import numpy as np

data = pd.read_csv('Cars93.csv')


def handle_missing_values(df):
    df.fillna(df.mean(numeric_only=True), inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def reduce_noise(df):
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip().str.lower()
    return df


def encode_categorical(df):
    encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoders[col] = encoder
    return df, encoders


def normalize_features(df, method='minmax'):
    scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df, scaler


def split_data(df, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_state=42):
    train_data, temp_data = train_test_split(df, test_size=1-train_ratio, random_state=random_state)
    val_data, test_data = train_test_split(temp_data, test_size=test_ratio/(test_ratio + val_ratio), random_state=random_state)
    return train_data, val_data, test_data




data_cleaned = handle_missing_values(data.copy())

data_noisy_cleaned = reduce_noise(data_cleaned)


data_encoded, encoders = encode_categorical(data_noisy_cleaned)


data_normalized, scaler = normalize_features(data_encoded)


train_data, val_data, test_data = split_data(data_normalized)

print("Train Data Sample:\n", train_data.head())
print("\nValidation Data Sample:\n", val_data.head())
print("\nTest Data Sample:\n", test_data.head())
