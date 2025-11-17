import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


def detect_anomalies_autoencoder(dataset: TensorDataset, autoencoder_model, 
                                 device: torch.device, percentile: float = 95.0,
                                 batch_size: int = 256) -> tuple:
    """
    Autoencoder Detector: Deep learning-based - flag samples with high reconstruction error.
    Effective for adversarial and corruption attacks.
    
    Args:
        dataset: TensorDataset with (x, y) tensors
        autoencoder_model: Trained autoencoder model
        device: torch device
        percentile: Threshold percentile (default 95th)
        batch_size: Batch size for inference
    
    Returns:
        (anomaly_indices, cleaned_dataset)
    """
    autoencoder_model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    reconstruction_errors = []
    
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            recon = autoencoder_model(xb)
            # Compute MSE per sample
            errors = F.mse_loss(recon, xb, reduction='none')
            errors = errors.view(errors.size(0), -1).mean(dim=1)
            reconstruction_errors.extend(errors.cpu().numpy())
    
    reconstruction_errors = np.array(reconstruction_errors)
    threshold = np.percentile(reconstruction_errors, percentile)
    
    # Flag anomalies
    anomaly_mask = reconstruction_errors > threshold
    anomaly_indices = np.where(anomaly_mask)[0].tolist()
    
    # Create cleaned dataset
    x, y = dataset.tensors
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


