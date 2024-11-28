import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import joblib
import warnings

warnings.filterwarnings('ignore')


def ingest_data(file_path):
    """
    Ingest data from a CSV file and return as a DataFrame.
    """
    return pd.read_csv(file_path)


def preprocess_data(data, target_column):
    """
    Preprocess data by splitting into train/test and setting up pipelines for numerical/categorical features.
    """
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Identify numerical and categorical features
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Preprocessing pipelines
    numerical_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])
    
    return X_train, X_test, y_train, y_test, preprocessor


def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor, artifacts_dir):
    """
    Train and evaluate multiple models with hyperparameter tuning.
    Save the best model with the highest accuracy.
    """
    models = {
        'LogisticRegression': LogisticRegression(),
        'RandomForest': RandomForestClassifier(),
        'SVC': SVC(),
        'DecisionTree': DecisionTreeClassifier()
    }
    
    hyperparameters = {
        'LogisticRegression': {'model__C': [0.1, 1, 10]},
        'RandomForest': {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 10, 20]
        },
        'SVC': {'model__C': [0.1, 1, 10], 'model__kernel': ['linear', 'rbf']},
        'DecisionTree': {'model__max_depth': [None, 10, 20]}
    }
    
    results = []
    best_model = None
    best_accuracy = 0.0
    best_model_name = None
    
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        param_grid = hyperparameters[model_name]
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Best model evaluation
        best_pipeline = grid_search.best_estimator_
        y_pred = best_pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results.append({
            'Model': model_name,
            'Best Params': grid_search.best_params_,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })
        
        # Update best model if current model has higher accuracy
        if accuracy > best_accuracy:
            best_model = best_pipeline
            best_accuracy = accuracy
            best_model_name = model_name
        
        print(f"{model_name} evaluation completed.")
    
    # Save the best model and preprocessing pipeline
    os.makedirs(artifacts_dir, exist_ok=True)
    best_model_path = os.path.join(artifacts_dir, f"best_model_{best_model_name}.h5")
    joblib.dump(best_model, best_model_path)
    print(f"Best model saved at: {best_model_path}")
    
    # Convert results to DataFrame for easy comparison
    results_df = pd.DataFrame(results)
    print("\nExperiment Results:")
    print(results_df)
    
    return results_df


def train(file_path, target_column):
    # Artifacts directory
    artifacts_dir = "artifacts"
    
    # Data ingestion
    data = ingest_data(file_path)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(data, target_column)
    
    # Train and evaluate models
    results_df = train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor, artifacts_dir)
    
    # Save results to a CSV file
    results_path = os.path.join(artifacts_dir, 'experiment_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}.")



