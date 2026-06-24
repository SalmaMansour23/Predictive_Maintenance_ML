# Predictive Maintenance ML Pipeline

XGBoost-based machine learning pipeline for predicting machine failure type and probability.

## 📋 Overview

This system uses **XGBoost Gradient Boosting** to predict:
- **Failure Probability**: Likelihood of machine failure (0-100%)
- **Failure Type**: Classification into 5 failure categories
  - **TWF**: Tool Wear Failure
  - **HDF**: Heat Dissipation Failure
  - **PWF**: Power Failure
  - **OSF**: Overstrain Failure
  - **RNF**: Random Failure

## 🚀 Quick Start

### 1. Activate Virtual Environment

**PowerShell:**
```powershell
.\myenv\Scripts\Activate.ps1
```

**Command Prompt (cmd):**
```cmd
myenv\Scripts\activate.bat
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train XGBoost Models

```bash
python train_model_xgb.py
```

Trains two XGBoost models:
- Failure detection (binary classification)
- Failure type classification (multi-class)
- Generates confusion matrix visualizations as JPG files

### 4. Launch Web Interface

```bash
python gradio_app.py
```

Open browser at: `http://localhost:7860`

## 📁 Project Structure

```
ML Integration/
├── train_model_xgb.py       # XGBoost training pipeline
├── inference_xgb.py         # XGBoost inference engine
├── gradio_app.py            # Web interface (Gradio)
├── requirements.txt         # Python dependencies
├── models/                  # Trained model files
│   ├── model_failure_detection_xgb.json
│   ├── model_failure_type_xgb.json
│   ├── label_encoder_type.pkl
│   ├── label_encoder_failure_type.pkl
│   ├── confusion_matrix_failure.jpg
│   └── confusion_matrix_type.jpg
└── ai4i2020.csv            # Dataset (10,000 samples)     
```

### Output Example

```
======================================================================
PREDICTIVE MAINTENANCE REPORT (XGBOOST)
======================================================================

Failure Probability: 82.0%
No Failure Probability: 18.0%

Most Likely Failure: Heat Dissipation Failure (HDF)
Description: Failure due to inadequate heat dissipation and thermal stress

🔧 Recommended Action:
   Reduce load or improve cooling system; check ventilation.

Failure Type Probabilities:
  HDF: 45.2%
  TWF: 28.1%
  OSF: 18.3%
  PWF: 5.2%
  RNF: 3.2%
======================================================================
```

## 📊 Model Details

### Algorithm: XGBoost
- **Failure Detection**: Binary classifier (gradient boosting)
- **Failure Type**: Multi-class classifier (gradient boosting)
- **Trees**: 200 per model
- **Max Depth**: 5
- **Learning Rate**: 0.1

### Training Data
- Dataset: AI4I 2020 Predictive Maintenance
- Samples: 10,000 machine records
- Train/Test Split: 80/20

### Features (Inputs)
1. Machine Type (L, M, H)
2. Air Temperature (K)
3. Process Temperature (K)
4. Rotational Speed (RPM)
5. Torque (Nm)
6. Tool Wear (minutes)
