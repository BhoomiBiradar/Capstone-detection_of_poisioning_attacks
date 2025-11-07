import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from lime import lime_image
from skimage.segmentation import mark_boundaries


def explain_with_lime(model, dataset: TensorDataset, anomaly_indices: list,
                      device: torch.device, num_samples: int = 5,
                      output_dir: str = "outputs"):
    """
    Generate LIME explanations for detected anomalies.
    
    Args:
        model: Trained CNN model
        dataset: TensorDataset
        anomaly_indices: List of indices flagged as anomalies
        device: torch device
        num_samples: Number of samples to explain
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    x, y = dataset.tensors
    
    # Select a subset of anomalies to explain
    if len(anomaly_indices) > 0:
        sample_indices = anomaly_indices[:min(num_samples, len(anomaly_indices))]
        x_sample = x[sample_indices]
        y_sample = y[sample_indices]
    else:
        # If no anomalies, sample random indices
        sample_indices = np.random.choice(len(x), min(num_samples, len(x)), replace=False)
        x_sample = x[sample_indices]
        y_sample = y[sample_indices]
    
    # Create a prediction function for LIME
    def predict_fn(images):
        """Convert images to tensor and get predictions."""
        if isinstance(images, np.ndarray):
            # LIME provides images in [0, 1] range
            if images.max() > 1.0:
                images = images / 255.0
            
            # Handle different input shapes
            if len(images.shape) == 4:
                # Batch of images
                if images.shape[-1] == 3:  # HWC format
                    images = torch.from_numpy(images).permute(0, 3, 1, 2).float()
                else:  # CHW format
                    images = torch.from_numpy(images).float()
            else:
                # Single image
                if images.shape[-1] == 3:
                    images = torch.from_numpy(images).permute(2, 0, 1).float().unsqueeze(0)
                else:
                    images = torch.from_numpy(images).float().unsqueeze(0)
            
            images = images.to(device)
        else:
            images = images.to(device)
        
        with torch.no_grad():
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()
    
    # Initialize LIME explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Explain each sample
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (img, label) in enumerate(zip(x_sample, y_sample)):
        # Convert to numpy format for LIME (HWC, [0, 1])
        img_np = img.permute(1, 2, 0).numpy()
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
        img_np = np.clip(img_np, 0, 1)
        
        # Get explanation
        try:
            explanation = explainer.explain_instance(
                img_np,
                predict_fn,
                top_labels=1,
                hide_color=0,
                num_samples=1000
            )
            
            # Get explanation for the predicted class
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0],
                positive_only=True,
                num_features=10,
                hide_rest=False
            )
            
            # Plot original and explanation
            axes[idx, 0].imshow(img_np)
            axes[idx, 0].set_title(f'Original (Label: {label.item()})')
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(mark_boundaries(temp, mask))
            axes[idx, 1].set_title(f'LIME Explanation')
            axes[idx, 1].axis('off')
            
        except Exception as e:
            print(f"LIME explanation failed for sample {idx}: {e}")
            # Fallback: show original image
            axes[idx, 0].imshow(img_np)
            axes[idx, 0].set_title(f'Original (Label: {label.item()})')
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(img_np)
            axes[idx, 1].set_title('Explanation unavailable')
            axes[idx, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lime_explanations.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"LIME explanations saved to {output_dir}/lime_explanations.png")


