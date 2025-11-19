"""
Streamlit Frontend for Poisoning Attack Detection System
"""
import streamlit as st
import requests
import json
import base64
from PIL import Image
import io
import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np

# Configuration
API_BASE_URL = "http://127.0.0.1:8081"

# Page config
st.set_page_config(
    page_title="Poisoning Attack Detection",
    page_icon="🛡️",
    layout="wide"
)

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "current_dataset" not in st.session_state:
    st.session_state.current_dataset = None
if "detection_results" not in st.session_state:
    st.session_state.detection_results = None
if "feedback_results" not in st.session_state:
    st.session_state.feedback_results = None
if "original_accuracy" not in st.session_state:
    st.session_state.original_accuracy = None
if "cleaned_metrics" not in st.session_state:
    st.session_state.cleaned_metrics = None
if "visualization_data" not in st.session_state:
    st.session_state.visualization_data = None


def make_api_request(endpoint: str, method: str = "GET", data: dict = None):
    """Make API request to backend."""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, params=data)
        else:
            response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        return result
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None


def decode_base64_image(img_str: str) -> Image.Image:
    """Decode base64 image string to PIL Image."""
    img_data = base64.b64decode(img_str)
    return Image.open(io.BytesIO(img_data))


# Header
st.title("🛡️ Detecting Poisoning Attacks in ML Pipelines")
st.subheader("In Real-Time with Feedback Detector")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    if st.button("🚀 Initialize System", type="primary"):
        with st.spinner("Initializing datasets and models..."):
            result = make_api_request("/initialize", method="POST")
            if result and result.get("status") == "success":
                st.session_state.initialized = True
                st.success("System initialized!")
                st.json(result)
            else:
                st.error("Initialization failed!")
    
    st.markdown("---")
    
    st.subheader("🧪 Poison Mix Settings")
    mix_flipped = st.slider("Flipped (%)", 0, 100, 30, 5)
    mix_corrupted = st.slider("Corrupted (%)", 0, 100, 30, 5)
    mix_fgsm = st.slider("FGSM (%)", 0, 100, 30, 5)
    mix_clean = max(0, 100 - (mix_flipped + mix_corrupted + mix_fgsm))
    st.caption(f"Clean (%) auto-adjusted to: {mix_clean}%")

    dataset_type = st.selectbox(
        "📊 Base Dataset",
        ["mixed", "clean", "flipped", "corrupted", "fgsm"],
        index=0
    )
    
    if st.button("📥 Load Dataset"):
        payload = {
            "dataset_type": dataset_type,
            "mix_flipped": mix_flipped,
            "mix_corrupted": mix_corrupted,
            "mix_fgsm": mix_fgsm,
        }
        with st.spinner(f"Loading {dataset_type} dataset..."):
            result = make_api_request("/send_data", method="POST", data=payload)
            if result and result.get("status") == "success":
                st.session_state.current_dataset = dataset_type
                st.success(f"Dataset loaded! ({result['total_samples']} samples)")
                if result.get("mix_config"):
                    st.info("Mix configuration:")
                    st.json(result["mix_config"])
                # Reset accuracy and visualization data when new dataset is loaded
                st.session_state.original_accuracy = None
                st.session_state.cleaned_metrics = None
                st.session_state.visualization_data = None
            else:
                st.error("Failed to load dataset!")

# Main Content
if not st.session_state.initialized:
    st.warning("⚠️ Please initialize the system first using the sidebar.")
    st.info("Click '🚀 Initialize System' to download datasets and train models.")
else:
    # Dataset Selection Panel
    st.header("📊 Dataset Selection")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        dataset_options = ["clean", "flipped", "corrupted", "fgsm", "mixed"]
        current_index = 1  # default to "flipped"
        if st.session_state.current_dataset is not None:
            try:
                current_index = dataset_options.index(st.session_state.current_dataset)
            except ValueError:
                current_index = 1  # fallback to "flipped" if not found
        
        selected_dataset = st.selectbox(
            "Select dataset type",
            dataset_options,
            index=current_index
        )
    
    with col2:
        if st.button("📥 Load Dataset", type="primary"):
            with st.spinner(f"Loading {selected_dataset} dataset..."):
                result = make_api_request("/send_data", method="POST", data={"dataset_type": selected_dataset})
                if result and result.get("status") == "success":
                    st.session_state.current_dataset = selected_dataset
                    st.success("Dataset loaded!")
                    st.rerun()
    
    if st.session_state.current_dataset:
        # Sample Images Gallery
        st.subheader("🖼️ Sample Images")
        result = make_api_request("/get_sample_images", data={
            "dataset_type": st.session_state.current_dataset,
            "num_samples": 10
        })
        
        if result and result.get("status") == "success":
            samples = result.get("samples", [])
            cols = st.columns(5)
            for idx, sample in enumerate(samples[:10]):
                with cols[idx % 5]:
                    img = decode_base64_image(sample["image"])
                    st.image(img, caption=f"Label: {sample['label']}", width=100)
        
        st.markdown("---")
        
        # NEW: Get Original Accuracy Button
        st.header("📊 Model Performance on Poisoned Data")
        if st.button("🎯 Get Accuracy (Poisoned Data)"):
            with st.spinner("Computing model accuracy on poisoned data..."):
                result = make_api_request("/metrics", data={"dataset_type": st.session_state.current_dataset})
                if result and result.get("status") == "success":
                    st.session_state.original_accuracy = result["accuracy"]
                    st.metric("Accuracy on Poisoned Data", f"{result['accuracy']:.4f}")
                    st.info("This shows how the poisoning attack has degraded model performance.")
                else:
                    st.error("Failed to compute accuracy!")
        
        # Display original accuracy if already computed
        if st.session_state.original_accuracy is not None:
            st.metric("Accuracy on Poisoned Data", f"{st.session_state.original_accuracy:.4f}")
        
        st.markdown("---")
        
        # Detection Panel
        st.header("🔍 Detection Algorithms")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            threshold = st.slider("Detection Threshold (Percentile)", 80.0, 99.0, 95.0, 1.0)
        
        with col2:
            if st.button("▶️ Run Detection", type="primary"):
                with st.spinner("Running detection algorithms..."):
                    result = make_api_request("/run_detection", method="POST", data={
                        "dataset_type": st.session_state.current_dataset,
                        "threshold": threshold
                    })
                    if result and result.get("status") == "success":
                        st.session_state.detection_results = result
                        st.success("Detection completed!")
                        
                        # Automatically compute cleaned metrics after detection
                        with st.spinner("Computing accuracy on cleaned data..."):
                            cleaned_result = make_api_request("/metrics_cleaned", method="POST", data={
                                "dataset_type": st.session_state.current_dataset,
                                "threshold": result.get("threshold", threshold)
                            })
                            if cleaned_result and cleaned_result.get("status") in ["success", "warning"]:
                                st.session_state.cleaned_metrics = cleaned_result
                                st.session_state.visualization_data = None  # Reset visualization data
                            else:
                                st.error("Failed to compute cleaned data accuracy!")
                    else:
                        st.error("Detection failed!")

        if st.session_state.detection_results:
            st.subheader("📈 Detection Results")
            
            # Detector results table
            detectors_data = []
            for detector_name, detector_result in st.session_state.detection_results.get("detectors", {}).items():
                if "anomalies_count" in detector_result:
                    detectors_data.append({
                        "Detector": detector_name.upper(),
                        "Anomalies Found": detector_result["anomalies_count"]
                    })
            
            if detectors_data:
                df = pd.DataFrame(detectors_data)
                st.dataframe(df, width="stretch")
                
                # Bar chart
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(df["Detector"], df["Anomalies Found"], color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#9370DB'])
                ax.set_ylabel("Anomalies Found")
                ax.set_title("Detection Results by Detector")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            
            st.info(f"**Total Unique Anomalies:** {st.session_state.detection_results.get('total_anomalies', 0)}")
            if "threshold" in st.session_state.detection_results:
                st.info(f"**Final Threshold:** {st.session_state.detection_results['threshold']:.2f}%")

            adaptive_log = st.session_state.detection_results.get("adaptive_log", [])
            if adaptive_log:
                st.subheader("🔧 Adaptive Threshold Tuning (Detection)")
                log_df = pd.DataFrame(adaptive_log)
                st.dataframe(log_df, width="stretch")
                
            # NEW: Accuracy comparison section with visualizations
            st.subheader("📊 Accuracy Comparison")
            st.markdown("Comparing model accuracy before and after removing flagged samples:")
            
            # Get accuracy metrics for original dataset (if not already computed)
            if st.session_state.original_accuracy is None:
                original_metrics = make_api_request("/metrics", data={"dataset_type": st.session_state.current_dataset})
                original_accuracy = original_metrics.get("accuracy", 0) if original_metrics else 0
            else:
                original_accuracy = st.session_state.original_accuracy
            
            # Display accuracy comparison with visualizations
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Accuracy", f"{original_accuracy:.4f}")
            with col2:
                if st.session_state.cleaned_metrics and st.session_state.cleaned_metrics.get("status") in ["success", "warning"]:
                    st.metric("Cleaned Accuracy", f"{st.session_state.cleaned_metrics['cleaned_accuracy']:.4f}")
                else:
                    st.metric("Cleaned Accuracy", "Run Detection First", "N/A")
            with col3:
                if st.session_state.cleaned_metrics and st.session_state.cleaned_metrics.get("status") in ["success", "warning"]:
                    improvement = st.session_state.cleaned_metrics['accuracy_improvement']
                    st.metric("Improvement", f"{improvement:+.4f}", f"{improvement:+.4f}")
                    # Add warning message if applicable
                    if st.session_state.cleaned_metrics.get("status") == "warning":
                        st.warning(st.session_state.cleaned_metrics.get("message", "Warning during evaluation"))
                else:
                    st.metric("Improvement", "Run Detection First", "N/A")
            
            # Visualize accuracy comparison
            if st.session_state.cleaned_metrics and st.session_state.cleaned_metrics.get("status") in ["success", "warning"]:
                fig, ax = plt.subplots(figsize=(8, 4))
                accuracies = [original_accuracy, st.session_state.cleaned_metrics['cleaned_accuracy']]
                labels = ['Original', 'Cleaned']
                colors = ['#FF6B6B', '#4ECDC4']
                bars = ax.bar(labels, accuracies, color=colors)
                ax.set_ylabel('Accuracy')
                ax.set_title('Accuracy Comparison: Original vs Cleaned Data')
                ax.set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, acc in zip(bars, accuracies):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{acc:.4f}', ha='center', va='bottom')
                
                st.pyplot(fig)
        
        # NEW: Feedback Loop Section
        st.markdown("---")
        st.header("🔄 Feedback Loop Optimization")
        st.markdown("Run the DDPG agent to optimize detection thresholds through reinforcement learning:")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            num_iterations = st.slider("Number of Feedback Iterations", 3, 10, 5)
        
        with col2:
            if st.button("🚀 Run Feedback Loop", type="primary"):
                with st.spinner("Running feedback loop optimization..."):
                    result = make_api_request("/feedback_loop", method="POST", data={
                        "dataset_type": st.session_state.current_dataset,
                        "num_iterations": num_iterations
                    })
                    if result and result.get("status") == "success":
                        st.session_state.feedback_results = result
                        st.success("Feedback loop completed!")
                        
                        # Display final weights instead of metrics
                        final_weights = result.get("final_weights", [])
                        if final_weights and len(final_weights) == 4:
                            st.subheader("🎯 Optimized Detector Weights")
                            weight_labels = ["Centroid", "KNN", "Autoencoder", "Gradient"]
                            cols = st.columns(4)
                            for i, (label, weight) in enumerate(zip(weight_labels, final_weights)):
                                with cols[i]:
                                    st.metric(label, f"{weight:.3f}")
                        
                        # Display learning curve if available
                        feedback_viz_result = make_api_request("/get_visuals", data={"dataset_type": st.session_state.current_dataset})
                        if feedback_viz_result and feedback_viz_result.get("status") == "success":
                            available = feedback_viz_result.get("available_visuals", {})
                            feedback_learning_viz = available.get("feedback_learning")
                            if feedback_learning_viz:
                                st.subheader("📈 Feedback Learning Curve")
                                img_url = f"{API_BASE_URL}{feedback_learning_viz}"
                                st.image(img_url, caption="DDPG Agent Learning Progress", width=600)
                    else:
                        st.error("Feedback loop failed!")
        
        # Display previous feedback results if available
        if st.session_state.feedback_results:
            st.subheader("🎯 Previous Optimized Weights")
            final_weights = st.session_state.feedback_results.get("final_weights", [])
            if final_weights and len(final_weights) == 4:
                weight_labels = ["Centroid", "KNN", "Autoencoder", "Gradient"]
                cols = st.columns(4)
                for i, (label, weight) in enumerate(zip(weight_labels, final_weights)):
                    with cols[i]:
                        st.metric(label, f"{weight:.3f}")
        
        st.markdown("---")
        
        # Visualizations with SHAP and LIME only
        st.header("🎨 Explainability Analysis")
        if st.button("🖼️ Get SHAP & LIME Visualizations"):
            with st.spinner("Loading visualizations..."):
                # Only fetch visualizations if we don't have them or if detection was run
                if st.session_state.visualization_data is None or st.session_state.detection_results is not None:
                    result = make_api_request("/get_visuals", data={"dataset_type": st.session_state.current_dataset})
                    if result and result.get("status") == "success":
                        st.session_state.visualization_data = result
                        st.success("Visualizations loaded and cached.")
                    else:
                        st.error("Failed to load visualizations.")
                        if result:
                            st.write(result)
                
                # Display visualizations if available
                if st.session_state.visualization_data:
                    available = st.session_state.visualization_data.get("available_visuals", {})
                    if available:
                        # Separate visualizations by type - ONLY SHAP and LIME
                        shap_viz = available.get("shap")
                        lime_viz = available.get("lime")
                        
                        # Display SHAP visualizations
                        if shap_viz:
                            st.subheader("🔍 SHAP Explanations")
                            img_url = f"{API_BASE_URL}{shap_viz}"
                            try:
                                response = requests.get(img_url)
                                if response.status_code == 200:
                                    st.image(img_url, caption="SHAP Feature Importance Analysis", width=600)
                                else:
                                    st.error(f"Failed to load SHAP image. Status: {response.status_code}")
                            except Exception as e:
                                st.error(f"Error loading SHAP image: {str(e)}")
                        
                        # Display LIME visualizations
                        if lime_viz:
                            st.subheader("🔍 LIME Explanations")
                            img_url = f"{API_BASE_URL}{lime_viz}"
                            try:
                                response = requests.get(img_url)
                                if response.status_code == 200:
                                    st.image(img_url, caption="LIME Feature Importance Analysis", width=600)
                                else:
                                    st.error(f"Failed to load LIME image. Status: {response.status_code}")
                            except Exception as e:
                                st.error(f"Error loading LIME image: {str(e)}")
                    else:
                        st.warning("No visualizations found. Run detection first.")
                else:
                    st.warning("No visualizations available. Run detection first.")
        
        st.markdown("---")

        # Footer
        st.markdown("---")
        st.markdown("**Feedback Driven Poisoning Attack Detection System**")