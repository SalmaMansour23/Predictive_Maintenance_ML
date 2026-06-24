"""
XGBoost Pipeline for Predictive Maintenance
Trains gradient boosting models to predict:
1. Machine failure probability (binary classification)
2. Failure type (multi-class: TWF, HDF, PWF, OSF, RNF, or None)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
DATA_PATH = Path("ai4i2020.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

FAILURE_TYPES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
INPUT_FEATURES = ["Type", "Air temperature [K]", "Process temperature [K]", 
                  "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
TARGET_FAILURE = "Machine failure"


def save_confusion_matrix_heatmap(cm, labels, title, filename):
    """Save confusion matrix as a heatmap JPG image."""
    plt.figure(figsize=(10, 8))
    
    # Create DataFrame for better labels
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    
    # Create heatmap with blue/white colormap
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'}, 
                linewidths=2, linecolor='white', square=True, cbar=True)
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Save as JPG
    plt.savefig(filename, format='jpg', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def load_and_prepare_data():
    """Load dataset and prepare features and targets."""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Failure distribution:\n{df[TARGET_FAILURE].value_counts()}")
    
    # Create composite failure type target (one-hot encoded failures)
    # If multiple failures occur, pick the first non-zero one; else "None"
    def get_primary_failure(row):
        for ft in FAILURE_TYPES:
            if row[ft] == 1:
                return ft
        return "None"
    
    df["Failure_Type"] = df.apply(get_primary_failure, axis=1)
    print(f"\nFailure type distribution:\n{df['Failure_Type'].value_counts()}")
    
    # Encode categorical features
    le_type = LabelEncoder()
    df["Type_encoded"] = le_type.fit_transform(df["Type"])
    
    # Prepare feature matrix
    X = df[INPUT_FEATURES].copy()
    X["Type"] = df["Type_encoded"]
    X = X.drop("Type", axis=1).copy()
    X.insert(0, "Type_encoded", df["Type_encoded"])
    X.columns = ["Type_encoded", "Air_temp", "Process_temp", "RPM", "Torque", "Tool_wear"]
    
    # Targets
    y_failure = df[TARGET_FAILURE]
    y_failure_type = df["Failure_Type"]
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Features: {X.columns.tolist()}")
    
    return X, y_failure, y_failure_type, le_type


def train_models(X, y_failure, y_failure_type, le_type):
    """Train XGBoost models for failure detection and classification."""
    
    # Split data
    X_train, X_test, y_fail_train, y_fail_test, y_type_train, y_type_test = train_test_split(
        X, y_failure, y_failure_type, test_size=0.2, random_state=42, stratify=y_failure
    )
    
    print("\n" + "="*60)
    print("TRAINING XGBOOST FAILURE DETECTION MODEL")
    print("="*60)
    
    # Train failure detection model with XGBoost
    model_failure = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method='hist',
        eval_metric='logloss',
        verbosity=0
    )
    
    model_failure.fit(
        X_train, y_fail_train,
        eval_set=[(X_test, y_fail_test)],
        verbose=False
    )
    
    y_fail_pred = model_failure.predict(X_test)
    fail_acc = accuracy_score(y_fail_test, y_fail_pred)
    fail_cm = confusion_matrix(y_fail_test, y_fail_pred)
    
    print(f"Failure Detection Accuracy: {fail_acc:.4f}")
    print("\nClassification Report (Failure Detection):")
    print(classification_report(y_fail_test, y_fail_pred))
    print(f"\nConfusion Matrix (Failure Detection):\n{fail_cm}")
    
    print("\n" + "="*60)
    print("TRAINING XGBOOST FAILURE TYPE CLASSIFICATION MODEL")
    print("="*60)
    
    # Encode failure types
    le_failure_type = LabelEncoder()
    y_type_train_encoded = le_failure_type.fit_transform(y_type_train)
    y_type_test_encoded = le_failure_type.transform(y_type_test)
    
    # Train failure type model with XGBoost
    model_type = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method='hist',
        eval_metric='mlogloss',
        verbosity=0
    )
    
    model_type.fit(
        X_train, y_type_train_encoded,
        eval_set=[(X_test, y_type_test_encoded)],
        verbose=False
    )
    
    y_type_pred = model_type.predict(X_test)
    type_acc = accuracy_score(y_type_test_encoded, y_type_pred)
    type_cm = confusion_matrix(y_type_test_encoded, y_type_pred)
    
    print(f"Failure Type Classification Accuracy: {type_acc:.4f}")
    print("\nClassification Report (Failure Type):")
    print(classification_report(y_type_test_encoded, y_type_pred,
                              target_names=le_failure_type.classes_,
                              zero_division=0))
    print(f"\nConfusion Matrix (Failure Type):\n{type_cm}")
    
    # Save models and preprocessing objects
    print("\n" + "="*60)
    print("SAVING XGBOOST MODELS & VISUALIZATIONS")
    print("="*60)
    
    model_failure.save_model(str(MODEL_DIR / "model_failure_detection_xgb.json"))
    model_type.save_model(str(MODEL_DIR / "model_failure_type_xgb.json"))
    joblib.dump(le_type, MODEL_DIR / "label_encoder_type.pkl")
    joblib.dump(le_failure_type, MODEL_DIR / "label_encoder_failure_type.pkl")
    
    # Save confusion matrices
    joblib.dump(fail_cm, MODEL_DIR / "confusion_matrix_failure.pkl")
    joblib.dump(type_cm, MODEL_DIR / "confusion_matrix_type.pkl")
    
    print(f"Models saved to {MODEL_DIR}/")
    print("- model_failure_detection_xgb.json")
    print("- model_failure_type_xgb.json")
    print("- label_encoder_type.pkl")
    print("- label_encoder_failure_type.pkl")
    print("- confusion_matrix_failure.pkl")
    print("- confusion_matrix_type.pkl")
    
    # Generate and save confusion matrix heatmaps
    print("\nGenerating confusion matrix heatmap visualizations...")
    save_confusion_matrix_heatmap(
        fail_cm,
        ['No Failure', 'Failure'],
        'Failure Detection - XGBoost - Confusion Matrix',
        MODEL_DIR / "confusion_matrix_failure.jpg"
    )
    
    save_confusion_matrix_heatmap(
        type_cm,
        le_failure_type.classes_,
        'Failure Type Classification - XGBoost - Confusion Matrix',
        MODEL_DIR / "confusion_matrix_type.jpg"
    )
    
    return model_failure, model_type, le_type, le_failure_type


def main():
    """Main training pipeline."""
    X, y_failure, y_failure_type, le_type = load_and_prepare_data()
    train_models(X, y_failure, y_failure_type, le_type)
    print("\n✓ XGBoost Training complete!")


if __name__ == "__main__":
    main()
