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

# Configuration
API_BASE_URL = "http://localhost:8000"

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


def make_api_request(endpoint: str, method: str = "GET", data: dict = None):
    """Make API request to backend."""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, params=data)
        else:
            response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None


def decode_base64_image(img_str: str) -> Image.Image:
    """Decode base64 image string to PIL Image."""
    img_data = base64.b64decode(img_str)
    return Image.open(io.BytesIO(img_data))


# Header
st.title("🛡️ Detecting Poisoning Attacks in ML Pipelines")
st.subheader("Interactive Real-Time Feedback Demonstration")
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
    
    dataset_type = st.selectbox(
        "📊 Select Dataset",
        ["clean", "flipped", "corrupted", "fgsm"],
        index=1
    )
    
    if st.button("📥 Load Dataset"):
        with st.spinner(f"Loading {dataset_type} dataset..."):
            result = make_api_request("/send_data", method="POST", data={"dataset_type": dataset_type})
            if result and result.get("status") == "success":
                st.session_state.current_dataset = dataset_type
                st.success(f"Dataset loaded! ({result['total_samples']} samples)")
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
        selected_dataset = st.selectbox(
            "Select dataset type",
            ["clean", "flipped", "corrupted", "fgsm"],
            index=1 if st.session_state.current_dataset is None else 
                  ["clean", "flipped", "corrupted", "fgsm"].index(st.session_state.current_dataset)
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
                    st.image(img, caption=f"Label: {sample['label']}", width="stretch")
        
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
                ax.bar(df["Detector"], df["Anomalies Found"], color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
                ax.set_ylabel("Anomalies Found")
                ax.set_title("Detection Results by Detector")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            
            st.info(f"**Total Unique Anomalies:** {st.session_state.detection_results.get('total_anomalies', 0)}")
        
        st.markdown("---")
        
        # Feedback Loop Panel
        st.header("🔄 Feedback-Driven Adaptive Learning")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            num_iterations = st.slider("Number of Iterations", 1, 20, 10, 1)
        
        with col2:
            if st.button("▶️ Run Feedback Loop", type="primary"):
                with st.spinner("Running adaptive feedback loop..."):
                    result = make_api_request("/feedback_loop", method="POST", data={
                        "dataset_type": st.session_state.current_dataset,
                        "num_iterations": num_iterations
                    })
                    if result and result.get("status") == "success":
                        st.session_state.feedback_results = result
                        st.success("Feedback loop completed!")
        
        if st.session_state.feedback_results:
            st.subheader("📊 Feedback Loop Results")
            
            results = st.session_state.feedback_results.get("results", [])
            if results:
                # Create metrics dataframe
                metrics_data = []
                for r in results:
                    metrics_data.append({
                        "Iteration": r["iteration"],
                        "Threshold": f"{r['threshold']:.2f}%",
                        "Accuracy": f"{r['metrics']['accuracy']:.4f}",
                        "F1 Score": f"{r['metrics']['f1_score']:.4f}",
                        "Det F1": f"{r['metrics']['detection_f1']:.4f}",
                        "Reward": f"{r['reward']:.4f}"
                    })
                
                df = pd.DataFrame(metrics_data)
                st.dataframe(df, width="stretch")
                
                # Plot learning curves
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                
                iterations = [r["iteration"] for r in results]
                accuracies = [r["metrics"]["accuracy"] for r in results]
                f1_scores = [r["metrics"]["f1_score"] for r in results]
                det_f1_scores = [r["metrics"]["detection_f1"] for r in results]
                rewards = [r["reward"] for r in results]
                thresholds = [r["threshold"] for r in results]
                
                axes[0, 0].plot(iterations, accuracies, 'o-', label='Accuracy', linewidth=2)
                axes[0, 0].plot(iterations, f1_scores, 's-', label='F1 Score', linewidth=2)
                axes[0, 0].set_xlabel('Iteration')
                axes[0, 0].set_ylabel('Score')
                axes[0, 0].set_title('Classification Performance')
                axes[0, 0].legend()
                axes[0, 0].grid(alpha=0.3)
                
                axes[0, 1].plot(iterations, det_f1_scores, 'o-', color='red', linewidth=2)
                axes[0, 1].set_xlabel('Iteration')
                axes[0, 1].set_ylabel('Detection F1 Score')
                axes[0, 1].set_title('Anomaly Detection Performance')
                axes[0, 1].grid(alpha=0.3)
                
                axes[1, 0].plot(iterations, rewards, 'o-', color='green', linewidth=2)
                axes[1, 0].set_xlabel('Iteration')
                axes[1, 0].set_ylabel('Reward')
                axes[1, 0].set_title('DDPG Agent Reward')
                axes[1, 0].grid(alpha=0.3)
                
                axes[1, 1].plot(iterations, thresholds, 'o-', color='purple', linewidth=2)
                axes[1, 1].set_xlabel('Iteration')
                axes[1, 1].set_ylabel('Threshold Percentile')
                axes[1, 1].set_title('Adaptive Threshold')
                axes[1, 1].grid(alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
        
        st.markdown("---")
        
        # Metrics Panel
        st.header("📊 Evaluation Metrics")
        if st.button("📈 Get Metrics"):
            with st.spinner("Computing metrics..."):
                result = make_api_request("/metrics", data={"dataset_type": st.session_state.current_dataset})
                if result and result.get("status") == "success":
                    st.metric("Accuracy", f"{result['accuracy']:.4f}")
        
        # Visualizations
        st.header("🎨 Visualizations")
        if st.button("🖼️ Get Visualizations"):
            with st.spinner("Loading visualizations..."):
                result = make_api_request("/get_visuals", data={"dataset_type": st.session_state.current_dataset})
                if result and result.get("status") == "success":
                    available = result.get("available_visuals", {})
                    if available:
                        st.info("Visualizations available. Check backend/static/outputs/ directory.")
                        st.json(available)
                    else:
                        st.warning("No visualizations found. Run detection and feedback loop first.")

# Footer
st.markdown("---")
st.markdown("**Poisoning Attack Detection System** | Feedback-Driven Real-Time Detection")

