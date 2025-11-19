# Machine Learning Poisoning Attack Detection System - Project Summary

## Overview
This project implements an advanced poisoning attack detection system for machine learning models using multiple detection algorithms, feedback-driven learning, and explainability techniques. The system is designed to identify and remove poisoned samples from datasets to improve model accuracy and robustness.

## Key Components

### 1. Multi-Detector Architecture
The system employs four independent detection algorithms:
- **Centroid Detector**: Identifies anomalies based on distance from class centroids
- **KNN Detector**: Flags samples that are far from their k-nearest neighbors
- **Autoencoder Detector**: Detects samples with high reconstruction errors
- **Gradient Filter**: Identifies samples that cause large gradient changes

### 2. Detector Fusion with Weighted Voting
A sophisticated fusion system combines all detector outputs using weighted voting:
- Autoencoder: 35% (strong for feature corruption)
- Gradient Filter: 25% (strong for backdoor attacks)
- Centroid Detector: 20% (general corruption detection)
- KNN Detector: 20% (label flip attack detection)

### 3. DDPG-Based Adaptive Learning
A Deep Deterministic Policy Gradient (DDPG) reinforcement learning agent optimizes:
- Detection thresholds for individual detectors
- Fusion weights for optimal performance
- Balances precision and recall dynamically

### 4. Web-Based Interactive Interface
Streamlit-powered dashboard provides:
- Step-by-step workflow execution
- Real-time monitoring of detection progress
- Visual comparison of model performance
- Interactive parameter tuning

### 5. Explainability Analysis
SHAP and LIME techniques provide:
- Feature importance explanations for flagged samples
- Visual analysis of why samples were detected as anomalies
- Transparent decision-making process

## Workflow

1. **Data Loading**: User selects and loads datasets with various attack types
2. **Initial Assessment**: Model accuracy evaluated on potentially poisoned data
3. **Multi-Detector Analysis**: All four detectors run independently
4. **Fusion Processing**: Detector outputs combined with weighted voting
5. **Adaptive Optimization**: DDPG agent refines thresholds and weights
6. **Data Cleaning**: Flagged samples removed from dataset
7. **Model Retraining**: Model retrained on cleaned dataset
8. **Performance Evaluation**: Accuracy comparison before/after cleaning
9. **Explainability**: SHAP/LIME analysis of detected anomalies
10. **Results Presentation**: Comprehensive visualizations and metrics

## Technical Improvements

### Performance Optimizations
- Full dataset processing instead of subsets for better accuracy
- Optimized detector execution (runs only once for main detection)
- Efficient caching mechanisms
- Faster model retraining with appropriate epochs

### Accuracy Enhancements
- Dynamic threshold adjustment based on dataset characteristics
- Adaptive weight optimization for different attack types
- Proper model retraining on cleaned data for accurate comparisons
- Better handling of edge cases with few anomalies

### User Experience Improvements
- Streamlit web interface with intuitive workflow
- Clear accuracy comparison metrics
- Visual feedback at each step
- Detailed error handling and logging

## Results

The system demonstrates significant improvements in:
- Detection accuracy for various poisoning attack types
- Model performance recovery after cleaning
- Reduction of false positives and false negatives
- Interpretability of detection decisions

## File Structure

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

## Usage

### Web Interface (Recommended)
1. Start backend API: `cd backend && python -m api`
2. Start frontend: `cd frontend && streamlit run streamlit_app.py`

### Command Line
Run complete pipeline: `python main.py`

## Future Enhancements

1. **Advanced Attack Detection**: Integration of more sophisticated attack detection methods
2. **Real-time Processing**: Streaming data processing capabilities
3. **Model-Agnostic Detection**: Support for different model architectures
4. **Cloud Deployment**: Containerized deployment options
5. **Extended Explainability**: Additional interpretability techniques