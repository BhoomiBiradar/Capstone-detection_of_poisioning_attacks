import os
import torch
import numpy as np
from torch.utils.data import TensorDataset
from sklearn.neighbors import NearestNeighbors


def detect_anomalies_knn(dataset: TensorDataset, k: int = 5, percentile: float = 95.0) -> tuple:
    """
    KNN Detector: Density-based - compute mean distance to k nearest neighbors.
    Good for label flipping and outliers.
    
    Args:
        dataset: TensorDataset with (x, y) tensors
        k: Number of nearest neighbors (default 5)
        percentile: Threshold percentile (default 95th)
    
    Returns:
        (anomaly_indices, cleaned_dataset)
    """
    x, y = dataset.tensors
    x_flat = x.view(x.size(0), -1).numpy()
    
    # Fit KNN model
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
    nbrs.fit(x_flat)
    
    # Get distances to k nearest neighbors (excluding self)
    distances, _ = nbrs.kneighbors(x_flat)
    mean_distances = distances[:, 1:].mean(axis=1)  # Exclude self (distance 0)
    
    threshold = np.percentile(mean_distances, percentile)
    
    # Flag anomalies
    anomaly_mask = mean_distances > threshold
    anomaly_indices = np.where(anomaly_mask)[0].tolist()
    
    # Create cleaned dataset
    clean_mask = ~anomaly_mask
    x_clean = x[clean_mask]
    y_clean = y[clean_mask]
    cleaned_dataset = TensorDataset(x_clean, y_clean)
    
    return anomaly_indices, cleaned_dataset


def save_detector_results(anomaly_indices: list, cleaned_dataset: TensorDataset, 
                         base_name: str, output_dir: str = "data"):
    """Save anomaly indices and cleaned dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save anomaly indices
    torch.save(torch.tensor(anomaly_indices), 
               os.path.join(output_dir, f"{base_name}_anomalies.pt"))
    
    # Save cleaned dataset
    x, y = cleaned_dataset.tensors
    torch.save({"x": x, "y": y}, 
               os.path.join(output_dir, f"{base_name}_cleaned.pt"))


