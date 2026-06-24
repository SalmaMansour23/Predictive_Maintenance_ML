"""
Inference Engine for Predictive Maintenance (XGBoost Version)
Predicts machine failure probability and failure type
"""

import numpy as np
import joblib
from pathlib import Path
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

MODEL_DIR = Path("models")

FAILURE_TYPES = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# Failure type descriptions and recommended actions
FAILURE_DESCRIPTIONS = {
    "TWF": {
        "name": "Tool Wear Failure",
        "description": "Failure due to tool degradation and wear over time",
        "action": "Replace or sharpen tool; schedule preventive maintenance."
    },
    "HDF": {
        "name": "Heat Dissipation Failure",
        "description": "Failure due to inadequate heat dissipation and thermal stress",
        "action": "Reduce load or improve cooling system; check ventilation."
    },
    "PWF": {
        "name": "Power Failure",
        "description": "Failure due to insufficient power supply or power fluctuations",
        "action": "Check power supply; stabilize voltage and current."
    },
    "OSF": {
        "name": "Overstrain Failure",
        "description": "Failure due to mechanical overstrain and excessive stress",
        "action": "Reduce torque or RPM; check load capacity limits."
    },
    "RNF": {
        "name": "Random Failure",
        "description": "Unpredictable failure not correlated with monitored parameters",
        "action": "Perform full system diagnostics; consider replacement."
    },
    "None": {
        "name": "No Failure",
        "description": "Machine is operating normally",
        "action": "Continue normal operation; maintain routine maintenance schedule."
    }
}


class PredictiveMaintenancePredictor:
    """Load XGBoost models and perform predictions."""
    
    def __init__(self, model_dir=MODEL_DIR):
        """Initialize predictor by loading trained XGBoost models."""
        self.model_dir = Path(model_dir)
        
        # Load models
        self.model_failure = xgb.XGBClassifier()
        self.model_failure.load_model(str(self.model_dir / "model_failure_detection_xgb.json"))
        
        self.model_type = xgb.XGBClassifier()
        self.model_type.load_model(str(self.model_dir / "model_failure_type_xgb.json"))
        
        # Load preprocessing objects
        self.le_type = joblib.load(self.model_dir / "label_encoder_type.pkl")
        self.le_failure_type = joblib.load(self.model_dir / "label_encoder_failure_type.pkl")
        
        print("✓ XGBoost models loaded successfully")
    
    def predict(self, type_encoded, air_temp, process_temp, rpm, torque, tool_wear):
        """
        Predict failure probability and failure type.
        
        Args:
            type_encoded (int): Encoded machine type (0, 1, 2, ...)
            air_temp (float): Air temperature in Kelvin
            process_temp (float): Process temperature in Kelvin
            rpm (float): Rotational speed in revolutions per minute
            torque (float): Torque in Newton-meters
            tool_wear (float): Tool wear in minutes
        
        Returns:
            dict: Prediction results with probability and failure type
        """
        # Prepare features in same order as training
        features = np.array([[type_encoded, air_temp, process_temp, rpm, torque, tool_wear]])
        
        # Predict failure probability
        failure_prob = self.model_failure.predict_proba(features)[0][1]  # Probability of failure (class 1)
        is_failure = self.model_failure.predict(features)[0]
        
        # Predict failure type probabilities
        type_probs_array = self.model_type.predict_proba(features)[0]
        failure_type_idx = self.model_type.predict(features)[0]
        failure_type_name = self.le_failure_type.classes_[failure_type_idx]
        
        # If no failure, override to "None"
        if is_failure == 0:
            failure_type_name = "None"
        
        # Get description and action
        desc_info = FAILURE_DESCRIPTIONS.get(failure_type_name, FAILURE_DESCRIPTIONS["None"])
        
        # Create probability dict for display
        type_probs = {}
        for i, ft in enumerate(self.le_failure_type.classes_):
            type_probs[ft] = float(type_probs_array[i]) * 100
        
        result = {
            "failure_probability": float(failure_prob) * 100,
            "no_failure_probability": (1 - float(failure_prob)) * 100,
            "is_failure": bool(is_failure),
            "failure_type": failure_type_name,
            "failure_type_full_name": desc_info["name"],
            "description": desc_info["description"],
            "recommended_action": desc_info["action"],
            "type_probabilities": type_probs
        }
        
        return result
    
    def predict_with_names(self, machine_type, air_temp, process_temp, rpm, torque, tool_wear):
        """
        Predict using machine type name instead of encoded value.
        
        Args:
            machine_type (str): Machine type letter (e.g., 'L', 'M', 'H')
            air_temp (float): Air temperature in Kelvin
            process_temp (float): Process temperature in Kelvin
            rpm (float): Rotational speed in RPM
            torque (float): Torque in Nm
            tool_wear (float): Tool wear in minutes
        
        Returns:
            dict: Prediction results
        """
        try:
            type_encoded = self.le_type.transform([machine_type])[0]
        except ValueError:
            raise ValueError(f"Unknown machine type: {machine_type}. Expected one of {self.le_type.classes_.tolist()}")
        
        return self.predict(type_encoded, air_temp, process_temp, rpm, torque, tool_wear)


def format_prediction_report(prediction):
    """Format prediction results as a readable report."""
    report = []
    report.append("=" * 70)
    report.append("PREDICTIVE MAINTENANCE REPORT (XGBOOST)")
    report.append("=" * 70)
    report.append(f"\nFailure Probability: {prediction['failure_probability']:.1f}%")
    report.append(f"No Failure Probability: {prediction['no_failure_probability']:.1f}%")
    report.append(f"\nMost Likely Failure: {prediction['failure_type_full_name']} ({prediction['failure_type']})")
    report.append(f"Description: {prediction['description']}")
    report.append(f"\n🔧 Recommended Action:\n   {prediction['recommended_action']}")
    report.append("\nFailure Type Probabilities:")
    for ft, prob in sorted(prediction['type_probabilities'].items(), key=lambda x: x[1], reverse=True):
        report.append(f"  {ft}: {prob:.1f}%")
    report.append("=" * 70)
    
    return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    predictor = PredictiveMaintenancePredictor()
    
    # Example prediction
    prediction = predictor.predict_with_names(
        machine_type="L",
        air_temp=298.0,
        process_temp=308.5,
        rpm=1500,
        torque=45.0,
        tool_wear=100.0
    )
    
    print(format_prediction_report(prediction))
