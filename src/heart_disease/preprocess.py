import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# Function for data ingestion
def ingest_data(file_path, file_type='csv'):
    """
    Ingests data from a file and returns a DataFrame.
    Supports CSV and Excel formats.
    """
    if file_type == 'csv':
        data = pd.read_csv(file_path)
    elif file_type == 'excel':
        data = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file type. Use 'csv' or 'excel'.")
    return data

# Function to split data into train and test sets
def split_data(data, target_column, test_size=0.2, random_state=42):
    """
    Splits the data into training and testing sets.
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Pipeline for data cleaning and preprocessing
def create_pipeline(numerical_features, categorical_features):
    """
    Creates a preprocessing pipeline for numerical and categorical data.
    """
    # Pipeline for numerical data
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),  # Fill missing values with the mean
        ('scaler', StandardScaler())  # Normalize data
    ])
    
    # Pipeline for categorical data
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values with the most frequent value
        ('encoder', OneHotEncoder(handle_unknown='ignore'))  # Encode categorical features
    ])
    
    # Combine both pipelines
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_features),
        ('cat', cat_pipeline, categorical_features)
    ])
    
    return preprocessor

def main(file_path, target_column, output_dir, file_type):
    """
    Main function to execute the data pipeline.
    """
    # Step 1: Data Ingestion
    data = ingest_data(file_path, file_type=file_type)
    print("Data Ingested Successfully.")
    
    # Step 2: Train-Test Split
    X_train, X_test, y_train, y_test = split_data(data, target_column)
    print("Data Split into Train and Test Sets.")
    
    # Step 3: Identify numerical and categorical columns
    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Step 4: Create Preprocessing Pipeline
    preprocessor = create_pipeline(numerical_features, categorical_features)
    
    # Step 5: Fit the pipeline
    preprocessor.fit(X_train)
    X_train_cleaned = preprocessor.transform(X_train)
    X_test_cleaned = preprocessor.transform(X_test)
    
    # Extract feature names using `get_feature_names_out`
    column_names = preprocessor.get_feature_names_out(input_features=numerical_features + categorical_features)
    
    # Convert processed data to DataFrame
    X_train_cleaned = pd.DataFrame(X_train_cleaned, columns=column_names)
    X_test_cleaned = pd.DataFrame(X_test_cleaned, columns=column_names)
    
    # Combine cleaned data with target column
    train_cleaned = pd.concat([X_train_cleaned, y_train.reset_index(drop=True)], axis=1)
    test_cleaned = pd.concat([X_test_cleaned, y_test.reset_index(drop=True)], axis=1)
    
    # Step 6: Save the cleaned data
    os.makedirs(output_dir, exist_ok=True)
    train_cleaned.to_csv(os.path.join(output_dir, 'train_cleaned.csv'), index=False)
    test_cleaned.to_csv(os.path.join(output_dir, 'test_cleaned.csv'), index=False)
    print(f"Cleaned data saved to {output_dir}.")


