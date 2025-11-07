# Detecting Poisoning Attacks in Machine Learning Pipelines

## Feedback-Driven Real-Time Detection System

This project implements a robust and adaptive framework for detecting, adapting to, and responding in real-time to data poisoning attacks in ML pipelines.

## Project Structure

```
project/
│
├── data/
│   ├── clean.pt
│   ├── flipped.pt
│   ├── corrupted.pt
│   ├── fgsm.pt
│
├── models/
│   ├── cnn_model.py
│   ├── autoencoder_model.py
│
├── detectors/
│   ├── centroid_detector.py
│   ├── knn_detector.py
│   ├── autoencoder_detector.py
│   ├── gradient_filter.py
│
├── feedback/
│   ├── ddpg_agent.py
│   ├── feedback_loop.py
│
├── explainability/
│   ├── shap_explain.py
│   ├── lime_explain.py
│
├── evaluation_metrics.py
├── main.py
└── requirements.txt
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the complete pipeline:
```bash
python main.py
```

The pipeline will:
1. Preprocess CIFAR-10 and generate attack datasets
2. Train CNN and Autoencoder models
3. Run four detection algorithms (Centroid, KNN, Autoencoder, Gradient)
4. Generate evaluation metrics and visualizations
5. Create SHAP and LIME explanations
6. Run feedback-driven adaptive learning with DDPG

## Outputs

All outputs are saved in the `outputs/` directory:
- Accuracy comparison plots
- Confusion matrices
- SHAP and LIME explanations
- Feedback learning curves
- DDPG agent checkpoints
