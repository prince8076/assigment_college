import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import numpy as np

# Load the dataset (update the file path as per your local setup)
file_path = 'Cars93.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Step 1: Function to handle missing values in the dataset
def handle_missing_values(df):
    # Fill numerical columns with the mean
    df.fillna(df.mean(numeric_only=True), inplace=True)
    # Fill categorical columns with the mode (most frequent value)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    return df

# Step 2: Function to reduce noise in individual attributes (basic cleaning)
def reduce_noise(df):
    # Strip leading/trailing spaces and handle inconsistent casing for categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip().str.lower()
    return df

# Step 3: Function to encode categorical variables
def encode_categorical(df):
    encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoders[col] = encoder
    return df, encoders

# Step 4: Function to normalize/scale numerical features
def normalize_features(df, method='minmax'):
    scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df, scaler

# Step 5: Function to split data into train, validation, and test sets
def split_data(df, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_state=42):
    train_data, temp_data = train_test_split(df, test_size=1-train_ratio, random_state=random_state)
    val_data, test_data = train_test_split(temp_data, test_size=test_ratio/(test_ratio + val_ratio), random_state=random_state)
    return train_data, val_data, test_data

# Applying the functions step-by-step

# Step 1: Handle missing values
data_cleaned = handle_missing_values(data.copy())

# Step 2: Reduce noise in the dataset
data_noisy_cleaned = reduce_noise(data_cleaned)

# Step 3: Encode categorical variables
data_encoded, encoders = encode_categorical(data_noisy_cleaned)

# Step 4: Normalize the features
data_normalized, scaler = normalize_features(data_encoded)

# Step 5: Split the dataset into train, validation, and test sets
train_data, val_data, test_data = split_data(data_normalized)

# Display the results to verify each step
print("Train Data Sample:\n", train_data.head())
print("\nValidation Data Sample:\n", val_data.head())
print("\nTest Data Sample:\n", test_data.head())
