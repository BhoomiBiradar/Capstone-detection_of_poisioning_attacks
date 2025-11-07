import os
import torch
import numpy as np
from torch.utils.data import TensorDataset


def detect_anomalies_centroid(dataset: TensorDataset, percentile: float = 95.0) -> tuple:
    """
    Centroid Detector: Statistical approach - flag samples far from their class centroid.
    Best for corruption and adversarial attacks.
    
    Args:
        dataset: TensorDataset with (x, y) tensors
        percentile: Threshold percentile (default 95th)
    
    Returns:
        (anomaly_indices, cleaned_dataset)
    """
    x, y = dataset.tensors
    x_flat = x.view(x.size(0), -1).numpy()
    y_np = y.numpy()
    
    # Compute centroids for each class
    centroids = {}
    for cls in range(10):  # CIFAR-10 has 10 classes
        mask = y_np == cls
        if mask.sum() > 0:
            centroids[cls] = x_flat[mask].mean(axis=0)
    
    # Compute distances to class centroids
    distances = []
    for i in range(len(x_flat)):
        cls = y_np[i]
        if cls in centroids:
            dist = np.linalg.norm(x_flat[i] - centroids[cls])
            distances.append(dist)
        else:
            distances.append(0.0)
    
    distances = np.array(distances)
    threshold = np.percentile(distances, percentile)
    
    # Flag anomalies
    anomaly_mask = distances > threshold
    anomaly_indices = np.where(anomaly_mask)[0].tolist()
    
    # Create cleaned dataset (remove anomalies)
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

