"""
Gradio Web Interface for Predictive Maintenance ML Simulator
Access at: http://localhost:7860
"""

import gradio as gr
import pandas as pd
import joblib
import sys
from pathlib import Path
from inference_xgb import PredictiveMaintenancePredictor, FAILURE_DESCRIPTIONS

# Global predictor instance
try:
    predictor = PredictiveMaintenancePredictor()
except FileNotFoundError:
    print("❌ ERROR: XGBoost models not found!")
    print("   Please run 'python train_model_xgb.py' first to train the models.")
    sys.exit(1)

# Load confusion matrices
MODEL_DIR = Path("models")
try:
    cm_failure = joblib.load(MODEL_DIR / "confusion_matrix_failure.pkl")
    cm_type = joblib.load(MODEL_DIR / "confusion_matrix_type.pkl")
    le_failure_type = joblib.load(MODEL_DIR / "label_encoder_failure_type.pkl")
except FileNotFoundError:
    cm_failure = None
    cm_type = None
    le_failure_type = None
    print("⚠ Warning: Confusion matrices not found. Run train_model.py to generate them.")


def predict_failure(machine_type, air_temp, process_temp, rpm, torque, tool_wear):
    """
    Make prediction and return formatted results.
    
    Returns tuple of outputs for Gradio interface.
    """
    try:
        prediction = predictor.predict_with_names(
            machine_type=machine_type,
            air_temp=air_temp,
            process_temp=process_temp,
            rpm=rpm,
            torque=torque,
            tool_wear=tool_wear
        )
        
        # Format outputs for display
        failure_prob = f"{prediction['failure_probability']:.1f}%"
        
        failure_type = prediction['failure_type_full_name']
        action = prediction['recommended_action']
        
        # Create a DataFrame for the failure type probabilities chart
        type_probs = prediction['type_probabilities']
        chart_data = pd.DataFrame({
            "Failure Type": list(type_probs.keys()),
            "Probability (%)": list(type_probs.values())
        })
        
        # Format as markdown for better readability
        report = f"""
## 🎯 Prediction Results

### Failure Probability
**{failure_prob}** likely to fail  
**{prediction['no_failure_probability']:.1f}%** no failure

### Most Likely Failure Type
**{prediction['failure_type_full_name']}** ({prediction['failure_type']})

### Description
{prediction['description']}

### 🔧 Recommended Action
{prediction['recommended_action']}

### 📊 All Failure Type Probabilities
"""
        for ft in ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']:
            prob = type_probs.get(ft, 0)
            bar = "█" * int(prob / 5) + "░" * (20 - int(prob / 5))
            report += f"\n- **{ft}**: {bar} {prob:.1f}%"
        
        return (
            failure_prob,
            failure_type,
            action,
            report,
            chart_data
        )
    
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        return (
            "Error",
            "Error",
            error_msg,
            error_msg,
            pd.DataFrame()
        )


def get_confusion_matrix_failure():
    """Return confusion matrix for failure detection as DataFrame."""
    if cm_failure is None:
        return pd.DataFrame({"Error": ["Confusion matrix not loaded"]})
    
    df = pd.DataFrame(
        cm_failure,
        index=["No Failure", "Failure"],
        columns=["Pred: No Failure", "Pred: Failure"]
    )
    df.index.name = "Actual"
    return df


def get_confusion_matrix_type():
    """Return confusion matrix for failure type as DataFrame."""
    if cm_type is None or le_failure_type is None:
        return pd.DataFrame({"Error": ["Confusion matrix not loaded"]})
    
    # Use actual class labels from the label encoder
    labels = le_failure_type.classes_
    
    df = pd.DataFrame(
        cm_type,
        index=labels,
        columns=[f"Pred: {l}" for l in labels]
    )
    df.index.name = "Actual"
    return df


# Create Gradio interface
def create_interface():
    """Create and return the Gradio interface."""
    
    with gr.Blocks(title="Predictive Maintenance ML Simulator", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("# 🤖 Predictive Maintenance ML Simulator")
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### ⚙️ Parameters")
                
                machine_type = gr.Dropdown(
                    choices=["L", "M", "H"],
                    value="L",
                    label="Type"
                )
                
                with gr.Row():
                    air_temp = gr.Slider(
                        minimum=280,
                        maximum=320,
                        value=298.0,
                        step=0.1,
                        label="Air Temp (K)"
                    )
                    process_temp = gr.Slider(
                        minimum=300,
                        maximum=330,
                        value=308.5,
                        step=0.1,
                        label="Process Temp (K)"
                    )
                
                with gr.Row():
                    rpm = gr.Slider(
                        minimum=800,
                        maximum=3000,
                        value=1500,
                        step=10,
                        label="RPM"
                    )
                    torque = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=45.0,
                        step=0.5,
                        label="Torque (Nm)"
                    )
                
                tool_wear = gr.Slider(
                    minimum=0,
                    maximum=500,
                    value=50.0,
                    step=5,
                    label="Tool Wear (min)"
                )
                
                predict_btn = gr.Button("🔍 PREDICT", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Results")
                
                failure_prob_display = gr.Textbox(
                    label="Failure %",
                    interactive=False
                )
                
                failure_type_display = gr.Textbox(
                    label="Failure Type",
                    interactive=False
                )
                
                action_display = gr.Textbox(
                    label="Action",
                    interactive=False,
                    lines=2
                )
                
                prob_chart = gr.BarPlot(
                    title="Type Probabilities",
                    x="Failure Type",
                    y="Probability (%)",
                    height=250,
                    color="Probability (%)"
                )
        
        # Full report (collapsible via tabs)
        with gr.Tabs():
            with gr.TabItem("📋 Full Report"):
                report_display = gr.Markdown()
            
            with gr.TabItem("📌 Quick Scenarios"):
                with gr.Row():
                    gr.Button("Normal", size="sm").click(
                        fn=predict_failure,
                        inputs=[gr.State("L"), gr.State(298.0), gr.State(308.5), gr.State(1500), gr.State(45.0), gr.State(50.0)],
                        outputs=[failure_prob_display, failure_type_display, action_display, report_display, prob_chart]
                    )
                    gr.Button("High Temp", size="sm").click(
                        fn=predict_failure,
                        inputs=[gr.State("H"), gr.State(302.0), gr.State(315.0), gr.State(1800), gr.State(65.0), gr.State(150.0)],
                        outputs=[failure_prob_display, failure_type_display, action_display, report_display, prob_chart]
                    )
                    gr.Button("Extreme Load", size="sm").click(
                        fn=predict_failure,
                        inputs=[gr.State("M"), gr.State(300.0), gr.State(312.0), gr.State(2200), gr.State(75.0), gr.State(200.0)],
                        outputs=[failure_prob_display, failure_type_display, action_display, report_display, prob_chart]
                    )
            
            with gr.TabItem("📊 Model Performance"):
                with gr.Column():
                    gr.Markdown("### 🎯 Failure Detection Confusion Matrix")
                    cm_failure_table = gr.Dataframe(
                        value=get_confusion_matrix_failure(),
                        interactive=False,
                        scale=1
                    )
                    
                    gr.Markdown("### 🔍 Failure Type Confusion Matrix")
                    cm_type_table = gr.Dataframe(
                        value=get_confusion_matrix_type(),
                        interactive=False,
                        scale=1
                    )
        
        # Connect button to prediction function
        predict_btn.click(
            fn=predict_failure,
            inputs=[machine_type, air_temp, process_temp, rpm, torque, tool_wear],
            outputs=[
                failure_prob_display,
                failure_type_display,
                action_display,
                report_display,
                prob_chart
            ]
        )
    
    return demo


if __name__ == "__main__":
    print("Starting Gradio interface...")
    print("🌐 Open your browser and go to: http://localhost:7860")
    print("Press Ctrl+C to stop the server\n")
    
    demo = create_interface()
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
