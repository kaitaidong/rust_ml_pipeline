#!/usr/bin/env python3
"""
Python ML Pipeline - Equivalent to the Rust complex_ml_pipeline.rs
Demonstrates the same ML workflow for performance comparison.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float

@dataclass
class HyperParameters:
    max_depth: Optional[int]
    min_samples_split: int
    criterion: str

def main():
    """Main function equivalent to Rust main"""
    run_complex_pipeline()

def run_complex_pipeline():
    """Main pipeline function equivalent to Rust run_complex_pipeline"""
    print("=== Complex Python ML Pipeline ===\n")
    pipeline_start = time.time()
    
    # Step 1: Generate complex dataset with missing values
    print("Step 1: Generating complex dataset...")
    df, target = generate_complex_dataset(50000, 20)
    print(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Step 2: Advanced preprocessing with multiple imputation strategies
    print("\nStep 2: Advanced preprocessing and imputation...")
    imputation_start = time.time()
    df_processed = advanced_preprocessing(df)
    print(f"Imputation time: {time.time() - imputation_start:.4f}s")
    
    # Step 3: Feature engineering
    print("\nStep 3: Feature engineering...")
    feature_start = time.time()
    df_engineered = feature_engineering(df_processed)
    print(f"Feature engineering time: {time.time() - feature_start:.4f}s")
    
    # Step 4: Convert to matrix for ML
    X_matrix, y_vector = dataframe_to_matrix(df_engineered, target)
    
    # Step 5: Hyperparameter tuning with parallel grid search
    print("\nStep 4: Hyperparameter tuning (parallel grid search)...")
    tuning_start = time.time()
    best_params = parallel_hyperparameter_tuning(X_matrix, y_vector)
    print(f"Best parameters found: {best_params}")
    print(f"Tuning time: {time.time() - tuning_start:.4f}s")
    
    # Step 6: Train final model with best parameters
    print("\nStep 5: Training final model with best parameters...")
    final_start = time.time()
    final_metrics = train_final_model(X_matrix, y_vector, best_params)
    print("Final model metrics:")
    print(f"  Accuracy:  {final_metrics.accuracy:.4f}")
    print(f"  Precision: {final_metrics.precision:.4f}")
    print(f"  Recall:    {final_metrics.recall:.4f}")
    print(f"  F1 Score:  {final_metrics.f1:.4f}")
    print(f"Final training time: {time.time() - final_start:.4f}s")
    
    # Step 7: Ensemble learning
    print("\nStep 6: Ensemble learning with Random Forest...")
    ensemble_start = time.time()
    ensemble_metrics = train_ensemble(X_matrix, y_vector)
    print("Ensemble model metrics:")
    print(f"  Accuracy:  {ensemble_metrics.accuracy:.4f}")
    print(f"  Precision: {ensemble_metrics.precision:.4f}")
    print(f"  Recall:    {ensemble_metrics.recall:.4f}")
    print(f"  F1 Score:  {ensemble_metrics.f1:.4f}")
    print(f"Ensemble training time: {time.time() - ensemble_start:.4f}s")
    
    total_time = time.time() - pipeline_start
    print(f"\n=== Total pipeline execution time: {total_time:.4f}s ===")
    
    return total_time

def generate_complex_dataset(n_rows: int, n_features: int) -> Tuple[pd.DataFrame, List[int]]:
    """Generate synthetic dataset with missing values"""
    np.random.seed(42)  # For reproducibility
    data = {}
    
    # Generate numeric features with varying missing rates
    for i in range(n_features // 2):
        missing_rate = 0.05 + (i * 0.02)  # 5% to 35% missing
        values = np.random.uniform(-100, 100, n_rows)
        
        # Introduce missing values
        missing_mask = np.random.random(n_rows) < missing_rate
        values[missing_mask] = np.nan
        
        data[f"num_feat_{i}"] = values
    
    # Generate categorical features
    categories = ["A", "B", "C", "D", "E"]
    for i in range(n_features // 4):
        missing_rate = 0.1
        values = np.random.choice(categories, n_rows)
        
        # Introduce missing values
        missing_mask = np.random.random(n_rows) < missing_rate
        values = values.astype(object)
        values[missing_mask] = None
        
        data[f"cat_feat_{i}"] = values
    
    # Generate binary features
    for i in range(n_features // 4):
        values = np.random.choice([0, 1], n_rows)
        data[f"bin_feat_{i}"] = values
    
    # Generate target variable (binary classification)
    target = np.random.choice([0, 1], n_rows, p=[0.6, 0.4]).tolist()
    
    df = pd.DataFrame(data)
    return df, target

def advanced_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Advanced preprocessing with imputation and encoding"""
    df_processed = df.copy()
    
    # Get column names by type
    numeric_cols = [col for col in df.columns if col.startswith("num_")]
    categorical_cols = [col for col in df.columns if col.startswith("cat_")]
    
    # Numeric: median imputation
    for col_name in numeric_cols:
        median_val = df_processed[col_name].median()
        df_processed[col_name].fillna(median_val, inplace=True)
    
    # Categorical: label encoding (equivalent to Rust approach)
    for col_name in categorical_cols:
        # Fill missing values
        df_processed[col_name].fillna("Unknown", inplace=True)
        
        # Simple label encoding
        mapping = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "Unknown": 0}
        encoded_col_name = f"{col_name}_encoded"
        df_processed[encoded_col_name] = df_processed[col_name].map(mapping)
        
        # Drop original categorical column
        df_processed.drop(col_name, axis=1, inplace=True)
    
    return df_processed

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering with polynomial and statistical features"""
    df_engineered = df.copy()
    
    # Get numeric column names
    numeric_cols = [col for col in df.columns if col.startswith("num_")]
    
    # Add polynomial features for first few numeric columns
    if len(numeric_cols) >= 2:
        df_engineered["interaction_0_1"] = df_engineered[numeric_cols[0]] * df_engineered[numeric_cols[1]]
        df_engineered["squared_0"] = df_engineered[numeric_cols[0]] ** 2
    
    # Add statistical features (normalization)
    for col_name in numeric_cols[:5]:
        mean_val = df_engineered[col_name].mean()
        std_val = df_engineered[col_name].std()
        if std_val != 0:
            df_engineered[f"{col_name}_normalized"] = (df_engineered[col_name] - mean_val) / std_val
        else:
            df_engineered[f"{col_name}_normalized"] = 0
    
    return df_engineered

def dataframe_to_matrix(df: pd.DataFrame, target: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert dataframe to matrix format"""
    X = df.values.astype(np.float64)
    y = np.array(target, dtype=np.int32)
    return X, y

def parallel_hyperparameter_tuning(X: np.ndarray, y: np.ndarray) -> HyperParameters:
    """Parallel hyperparameter tuning with grid search"""
    # Define hyperparameter grid
    max_depths = [5, 10, 15, None]
    min_samples_splits = [2, 5, 10]
    criteria = ["gini", "entropy"]
    
    # Generate all combinations
    param_grid = []
    for depth in max_depths:
        for min_split in min_samples_splits:
            for criterion in criteria:
                param_grid.append(HyperParameters(
                    max_depth=depth,
                    min_samples_split=min_split,
                    criterion=criterion
                ))
    
    print(f"Testing {len(param_grid)} parameter combinations...")
    
    # Use a subset for faster tuning (equivalent to Rust approach)
    subset_size = min(1000, len(X))
    if subset_size < len(X):
        X_subset, _, y_subset, _ = train_test_split(
            X, y, train_size=subset_size, random_state=42, stratify=y
        )
    else:
        X_subset, y_subset = X, y
    
    # Parallel grid search using ProcessPoolExecutor
    n_cores = min(mp.cpu_count(), len(param_grid))
    
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        # Create tasks for parallel execution
        futures = []
        for params in param_grid:
            future = executor.submit(evaluate_params, X_subset, y_subset, params)
            futures.append((future, params))
        
        # Collect results
        results = []
        for future, params in futures:
            try:
                score = future.result()
                results.append((params, score))
            except Exception as e:
                print(f"Error evaluating params {params}: {e}")
                results.append((params, 0.0))
    
    # Find best parameters
    best_params, best_score = max(results, key=lambda x: x[1])
    print(f"Best CV score: {best_score:.4f}")
    
    return best_params

def evaluate_params(X: np.ndarray, y: np.ndarray, params: HyperParameters) -> float:
    """Evaluate parameters using cross-validation (for parallel execution)"""
    try:
        clf = DecisionTreeClassifier(
            max_depth=params.max_depth,
            min_samples_split=params.min_samples_split,
            criterion=params.criterion,
            random_state=42
        )
        
        # 3-fold cross-validation
        scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
        return scores.mean()
    except Exception:
        return 0.0

def train_final_model(X: np.ndarray, y: np.ndarray, params: HyperParameters) -> ModelMetrics:
    """Train final model with best parameters"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    clf = DecisionTreeClassifier(
        max_depth=params.max_depth,
        min_samples_split=params.min_samples_split,
        criterion=params.criterion,
        random_state=42
    )
    
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    
    return ModelMetrics(
        accuracy=accuracy_score(y_test, predictions),
        precision=precision_score(y_test, predictions, pos_label=1, zero_division=0),
        recall=recall_score(y_test, predictions, pos_label=1, zero_division=0),
        f1=f1_score(y_test, predictions, pos_label=1, zero_division=0)
    )

def train_ensemble(X: np.ndarray, y: np.ndarray) -> ModelMetrics:
    """Train ensemble model with Random Forest"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    
    return ModelMetrics(
        accuracy=accuracy_score(y_test, predictions),
        precision=precision_score(y_test, predictions, pos_label=1, zero_division=0),
        recall=recall_score(y_test, predictions, pos_label=1, zero_division=0),
        f1=f1_score(y_test, predictions, pos_label=1, zero_division=0)
    )

if __name__ == "__main__":
    main()