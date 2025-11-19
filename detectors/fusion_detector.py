import os
import torch
import numpy as np
from torch.utils.data import TensorDataset
from typing import Dict, List, Tuple

# Try backend structure first, fallback to root structure
try:
    from backend.detectors.centroid_detector import detect_anomalies_centroid
    from backend.detectors.knn_detector import detect_anomalies_knn
    from backend.detectors.autoencoder_detector import detect_anomalies_autoencoder
    from backend.detectors.gradient_filter import detect_anomalies_gradient
except ImportError:
    from detectors.centroid_detector import detect_anomalies_centroid
    from detectors.knn_detector import detect_anomalies_knn
    from detectors.autoencoder_detector import detect_anomalies_autoencoder
    from detectors.gradient_filter import detect_anomalies_gradient

def run_all_detectors(dataset: TensorDataset, cnn_model, autoencoder_model, device: torch.device, 
                     percentile: float = 95.0) -> Dict[str, np.ndarray]:
    """
    Run all detectors and return their anomaly scores as numpy arrays.
    
    Args:
        dataset: TensorDataset with (x, y) tensors
        cnn_model: Trained CNN model
        autoencoder_model: Trained autoencoder model
        device: torch device
        percentile: Threshold percentile (default 95th)
    
    Returns:
        Dictionary with detector names as keys and anomaly scores as values
    """
    detector_scores = {}
    dataset_size = len(dataset)
    
    # Centroid detector (strong for corruption)
    try:
        anomaly_idx_centroid, _ = detect_anomalies_centroid(dataset, percentile=percentile)
        scores = np.zeros(dataset_size)
        scores[anomaly_idx_centroid] = 1.0
        detector_scores['centroid'] = scores
    except Exception as e:
        print(f"Centroid detector failed: {e}")
        detector_scores['centroid'] = np.zeros(dataset_size)
    
    # KNN detector (strong for label flip)
    try:
        anomaly_idx_knn, _ = detect_anomalies_knn(dataset, k=5, percentile=percentile)
        scores = np.zeros(dataset_size)
        scores[anomaly_idx_knn] = 1.0
        detector_scores['knn'] = scores
    except Exception as e:
        print(f"KNN detector failed: {e}")
        detector_scores['knn'] = np.zeros(dataset_size)
    
    # Autoencoder detector (strong for feature corruption)
    try:
        anomaly_idx_ae, _ = detect_anomalies_autoencoder(
            dataset, autoencoder_model, device, percentile=percentile
        )
        scores = np.zeros(dataset_size)
        scores[anomaly_idx_ae] = 1.0
        detector_scores['autoencoder'] = scores
    except Exception as e:
        print(f"Autoencoder detector failed: {e}")
        detector_scores['autoencoder'] = np.zeros(dataset_size)
    
    # Gradient filter (strong for backdoor)
    try:
        anomaly_idx_grad, _ = detect_anomalies_gradient(
            dataset, cnn_model, device, percentile=percentile
        )
        scores = np.zeros(dataset_size)
        scores[anomaly_idx_grad] = 1.0
        detector_scores['gradient'] = scores
    except Exception as e:
        print(f"Gradient detector failed: {e}")
        detector_scores['gradient'] = np.zeros(dataset_size)
    
    return detector_scores

def weighted_voting_fusion(detector_scores: Dict[str, np.ndarray], 
                          weights: List[float] = None) -> np.ndarray:
    """
    Combine detector scores using weighted voting.
    
    Args:
        detector_scores: Dictionary with detector names as keys and anomaly scores as values
        weights: Weights for each detector [centroid, knn, autoencoder, gradient]
                Default weights: [0.2, 0.2, 0.35, 0.25]
    
    Returns:
        Weighted scores array for all samples
    """
    if weights is None:
        # Default weights based on detector strengths
        weights = [0.2, 0.2, 0.35, 0.25]  # [centroid, knn, autoencoder, gradient]
    
    # Calculate weighted scores for each sample
    scores = np.zeros(len(list(detector_scores.values())[0]))
    detector_names = ['centroid', 'knn', 'autoencoder', 'gradient']
    
    for i, name in enumerate(detector_names):
        if name in detector_scores:
            scores += detector_scores[name] * weights[i]
    
    return scores

def apply_adaptive_threshold(scores: np.ndarray, threshold: float = 0.55) -> List[int]:
    """
    Apply adaptive threshold to get final anomaly flags.
    
    Args:
        scores: Weighted scores for all samples
        threshold: Fusion threshold (default 0.55)
    
    Returns:
        List of anomaly indices
    """
    anomaly_mask = scores >= threshold
    anomaly_indices = np.where(anomaly_mask)[0].tolist()
    
    return anomaly_indices

def create_cleaned_dataset(dataset: TensorDataset, anomaly_indices: List[int]) -> TensorDataset:
    """
    Create a cleaned dataset by removing flagged anomalies.
    
    Args:
        dataset: Original dataset
        anomaly_indices: List of indices to remove
    
    Returns:
        Cleaned TensorDataset
    """
    x, y = dataset.tensors
    
    # Create mask for clean samples
    clean_mask = np.ones(len(x), dtype=bool)
    if anomaly_indices:
        clean_mask[anomaly_indices] = False
    
    # Create cleaned dataset
    x_clean = x[clean_mask]
    y_clean = y[clean_mask]
    cleaned_dataset = TensorDataset(x_clean, y_clean)
    
    return cleaned_dataset

def save_fusion_results(anomaly_indices: List[int], scores: np.ndarray, 
                       cleaned_dataset: TensorDataset, base_name: str, 
                       output_dir: str = "data"):
    """
    Save fusion detector results.
    
    Args:
        anomaly_indices: Final anomaly indices
        scores: Weighted scores for all samples
        cleaned_dataset: Cleaned dataset after removing anomalies
        base_name: Base name for saved files
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save anomaly indices
    torch.save(torch.tensor(anomaly_indices), 
               os.path.join(output_dir, f"{base_name}_fusion_anomalies.pt"))
    
    # Save scores
    torch.save(torch.tensor(scores), 
               os.path.join(output_dir, f"{base_name}_fusion_scores.pt"))
    
    # Save cleaned dataset
    x, y = cleaned_dataset.tensors
    torch.save({"x": x, "y": y}, 
               os.path.join(output_dir, f"{base_name}_fusion_cleaned.pt"))

def detect_anomalies_fusion(dataset: TensorDataset, cnn_model, autoencoder_model, 
                           device: torch.device, percentile: float = 95.0,
                           weights: List[float] = None, threshold: float = 0.55) -> Tuple[List[int], TensorDataset, np.ndarray]:
    """
    Fusion detector: Combine all detectors with weighted voting.
    
    Workflow:
    detector outputs → combine → DDPG agent → adaptive threshold → final flags
    
    Args:
        dataset: TensorDataset with (x, y) tensors
        cnn_model: Trained CNN model
        autoencoder_model: Trained autoencoder model
        device: torch device
        percentile: Threshold percentile for individual detectors (default 95th)
        weights: Weights for each detector [centroid, knn, autoencoder, gradient]
        threshold: Fusion threshold for final flags (default 0.55)
    
    Returns:
        Tuple of (anomaly_indices, cleaned_dataset, scores)
    """
    # Step 1: Run all detectors to get their outputs
    detector_scores = run_all_detectors(
        dataset, cnn_model, autoencoder_model, device, percentile
    )
    
    # Step 2: Combine detector outputs using weighted voting
    # Use optimized weights based on detector strengths for different attack types
    if weights is None:
        # Default weights based on detector strengths
        # Autoencoder (35%): Strong for feature corruption
        # Gradient Filter (25%): Strong for backdoor attacks
        # Centroid Detector (20%): Strong for general corruption
        # KNN Detector (20%): Strong for label flip attacks
        weights = [0.2, 0.2, 0.35, 0.25]  # [centroid, knn, autoencoder, gradient]
    
    scores = weighted_voting_fusion(detector_scores, weights)
    
    # Step 3: Apply adaptive threshold to get final flags
    # Note: In the full implementation, the DDPG agent would adjust this threshold
    # For better results, we might want to adjust the threshold based on the dataset
    adjusted_threshold = threshold
    
    # If we have very few anomalies, we might want to lower the threshold
    anomaly_ratio = np.mean(scores >= adjusted_threshold)
    if anomaly_ratio < 0.01:  # Less than 1% anomalies
        adjusted_threshold = np.percentile(scores, 90)  # Use 90th percentile instead
        print(f"[Fusion] Low anomaly ratio ({anomaly_ratio:.4f}), adjusting threshold to {adjusted_threshold:.4f}")
    
    anomaly_indices = apply_adaptive_threshold(scores, adjusted_threshold)
    
    # Step 4: Create cleaned dataset with flagged samples removed
    cleaned_dataset = create_cleaned_dataset(dataset, anomaly_indices)
    
    print(f"[Fusion] Detected {len(anomaly_indices)} anomalies out of {len(dataset)} samples ({len(anomaly_indices)/len(dataset)*100:.2f}%)")
    
    return anomaly_indices, cleaned_dataset, scores
