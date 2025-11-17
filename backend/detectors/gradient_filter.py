import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


def detect_anomalies_gradient(dataset: TensorDataset, model, 
                             device: torch.device, percentile: float = 95.0,
                             batch_size: int = 256) -> tuple:
    """
    Gradient Filter: Adversarial detection - flag samples with high input gradient magnitude.
    Effective against FGSM attacks.
    
    Args:
        dataset: TensorDataset with (x, y) tensors
        model: Trained CNN model
        device: torch device
        percentile: Threshold percentile (default 95th)
        batch_size: Batch size for inference
    
    Returns:
        (anomaly_indices, cleaned_dataset)
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    gradient_magnitudes = []
    
    for xb, yb in loader:
        xb = xb.to(device).requires_grad_(True)
        yb = yb.to(device)
        
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        
        # Compute gradients w.r.t. input
        grad = torch.autograd.grad(loss, xb, create_graph=False, retain_graph=False)[0]
        
        # Compute gradient magnitude per sample
        grad_mag = grad.view(grad.size(0), -1).norm(dim=1)
        gradient_magnitudes.extend(grad_mag.detach().cpu().numpy())
    
    gradient_magnitudes = np.array(gradient_magnitudes)
    threshold = np.percentile(gradient_magnitudes, percentile)
    
    # Flag anomalies
    anomaly_mask = gradient_magnitudes > threshold
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


