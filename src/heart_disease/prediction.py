import joblib
import numpy as np
import pandas as pd


def predict_with_preprocessing(features_array, model_path, training_data_sample):
    """
    Predicts the target using the saved model and preprocesses the feature array.

    Parameters:
    - features_array (list or np.ndarray): Input features as a 1D array, arranged in the correct order.
    - model_path (str): Path to the saved model pipeline.
    - training_data_sample (pd.DataFrame): A sample of the training data used for identifying features.

    Returns:
    - prediction (any): The predicted value from the model.
    """
    # Load the saved model pipeline
    model_pipeline = joblib.load(model_path)
    
    # Ensure the input features are in a 2D array format
    if isinstance(features_array, list):
        features_array = np.array(features_array)
    features_array = features_array.reshape(1, -1)
    
    # Identify numerical and categorical features
    numerical_features = training_data_sample.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = training_data_sample.select_dtypes(include=['object', 'category']).columns.tolist()
    all_features = numerical_features + categorical_features

    # Check if the input feature array matches the expected number of columns
    if len(all_features) != features_array.shape[1]:
        raise ValueError(f"Expected {len(all_features)} features, but got {features_array.shape[1]}.")

    # Convert the feature array to a DataFrame for preprocessing compatibility
    input_data = pd.DataFrame(features_array, columns=all_features)
    
    # Extract the preprocessing pipeline from the saved model
    preprocessor = model_pipeline.named_steps['preprocessor']
    
    # Preprocess the input data
    preprocessed_data = preprocessor.transform(input_data)
    
    # Make the prediction using the trained model
    prediction = model_pipeline.named_steps['model'].predict(preprocessed_data)
    return prediction[0]  # Return the first prediction
