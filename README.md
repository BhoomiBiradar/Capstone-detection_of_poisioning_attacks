# Poisoning Attack Detection in Machine Learning Pipelines

Advanced poisoning attack detection system for ML models using multiple detection algorithms, feedback-driven learning, and explainability.

## Features

- **Multiple Detection Algorithms**:
  - Centroid-based anomaly detection
  - K-Nearest Neighbors (KNN) density estimation
  - Autoencoder reconstruction error analysis
  - Gradient magnitude filtering
  - **NEW: Fusion Detector with Weighted Voting**

- **Feedback-Driven Adaptive Learning**:
  - Deep Deterministic Policy Gradient (DDPG) agent
  - Dynamic threshold adjustment
  - Performance-based reward system
  - **NEW: Adaptive Weight Optimization**

- **Explainability**:
  - SHAP (SHapley Additive exPlanations)
  - LIME (Local Interpretable Model-agnostic Explanations)

- **Attack Simulation**:
  - Label flipping attacks
  - Data corruption attacks
  - FGSM (Fast Gradient Sign Method) adversarial attacks

- **NEW: Web-based Interactive Interface**:
  - Streamlit-powered dashboard for real-time monitoring
  - Visual comparison of model performance
  - Interactive parameter tuning

## NEW: Fusion Detector with Weighted Voting

We've implemented a sophisticated detector fusion system that combines all individual detectors using weighted voting based on their strengths:

- **Autoencoder** (35% weight): Strong for feature corruption detection
- **Gradient Filter** (25% weight): Strong for backdoor attack detection
- **Centroid Detector** (20% weight): Strong for general corruption detection
- **KNN Detector** (20% weight): Strong for label flip detection

The fusion detector uses a DDPG agent to adaptively adjust both individual detector thresholds and the fusion threshold for optimal performance. The DDPG agent also dynamically optimizes detector weights based on attack characteristics.

## Enhanced System Architecture

```
Data Preparation → Attack Simulation → Base Model Training → Multi-Detector Analysis → 
Detector Fusion (Weighted Voting) → DDPG Feedback Loop → Data Cleaning → 
Model Retraining → Accuracy Comparison → Explainability Analysis → Results & Metrics
```

## Installation

### Prerequisites
- Python 3.8+
- Conda (recommended)

### Setup Environment
```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate poison-detect

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Web Interface (Recommended)
```bash
# Start the backend API server
cd backend
python -m api

# In a new terminal, start the Streamlit frontend
cd frontend
streamlit run streamlit_app.py
```

### Running the Command Line Pipeline
```bash
python main.py
```

## Directory Structure

```
├── backend/
│   ├── api.py                 # FastAPI server
│   ├── detectors/             # Detection algorithms
│   ├── feedback/              # DDPG agent and feedback loop
│   ├── models/                # CNN and Autoencoder architectures
│   ├── static/                # Generated visualizations
│   └── utils/                 # Utilities and helpers
├── frontend/
│   └── streamlit_app.py       # Streamlit web interface
├── data/                      # Generated datasets and attack samples
└── outputs/                   # Results, plots, and explanations
```

## Key Improvements

1. **Full Dataset Processing**: Now processes entire datasets instead of subsets for better accuracy
2. **Optimized Detector Execution**: Detectors run only once for main detection (not multiple times)
3. **Enhanced Web Interface**: Streamlit dashboard with step-by-step workflow
4. **Improved Accuracy Comparison**: Clear before/after metrics with visualizations
5. **Dynamic Weight Adjustment**: DDPG agent optimizes detector weights in real-time
6. **Better Visualization**: SHAP/LIME explanations and learning curves
7. **Performance Optimization**: Faster execution with caching and efficient algorithms

## Results

The system provides comprehensive evaluation metrics including:
- Accuracy comparison before and after cleaning
- Detection precision, recall, and F1-score
- Confusion matrices
- SHAP and LIME explanations for flagged samples
- Learning curves from the DDPG agent
- Weight optimization visualizations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.