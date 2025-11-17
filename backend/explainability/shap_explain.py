import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
import shap


def explain_with_shap(
    model,
    dataset: TensorDataset,
    anomaly_indices: list,
    device: torch.device,
    num_samples: int = 100,
    output_dir: str = "outputs",
):
    """
    Generate SHAP explanations for detected anomalies.

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
    if anomaly_indices:
        sample_indices = anomaly_indices[: min(num_samples, len(anomaly_indices))]
        x_sample = x[sample_indices].to(device)
        y_sample = y[sample_indices]
    else:
        # If no anomalies, sample random indices
        sample_indices = np.random.choice(len(x), min(num_samples, len(x)), replace=False)
        x_sample = x[sample_indices].to(device)
        y_sample = y[sample_indices]

    # Create a wrapper function for SHAP
    def model_wrapper(x_input):
        """Wrapper to convert numpy to torch and get predictions."""
        if isinstance(x_input, np.ndarray):
            x_tensor = torch.from_numpy(x_input).float().to(device)
        else:
            x_tensor = x_input.to(device)

        with torch.no_grad():
            logits = model(x_tensor)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    # Use a background dataset (random subset of clean data)
    background_size = min(50, len(x))
    bg_indices = np.random.choice(len(x), background_size, replace=False)
    x_background = x[bg_indices].numpy()

    # For image data, use GradientExplainer
    try:
        explainer = shap.GradientExplainer(
            model_wrapper, torch.from_numpy(x_background).float().to(device)
        )

        # Explain a few samples
        num_explain = min(5, len(x_sample))
        shap_values = explainer.shap_values(x_sample[:num_explain])

        # Visualize
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # Take first class

        # Plot SHAP values
        plt.figure(figsize=(12, 8))
        shap.image_plot(shap_values, x_sample[:num_explain].cpu().numpy(), show=False)
        plt.savefig(os.path.join(output_dir, "shap_explanations.png"), dpi=150, bbox_inches="tight")
        plt.close()

        print(f"SHAP explanations saved to {output_dir}/shap_explanations.png")

    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        print("Creating a simplified visualization instead...")

        # Fallback: simple gradient-based visualization
        x_sample.requires_grad_(True)
        logits = model(x_sample[:num_explain])
        loss = torch.nn.functional.cross_entropy(logits, y_sample[:num_explain].to(device))
        loss.backward()

        gradients = x_sample.grad.abs().mean(dim=1).cpu().numpy()

        fig, axes = plt.subplots(2, num_explain, figsize=(15, 6))
        for i in range(num_explain):
            axes[0, i].imshow(x_sample[i].detach().cpu().permute(1, 2, 0).numpy())
            axes[0, i].set_title(f"Sample {i}")
            axes[0, i].axis("off")

            axes[1, i].imshow(gradients[i], cmap="hot")
            axes[1, i].set_title(f"Gradient {i}")
            axes[1, i].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_explanations.png"), dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Gradient-based explanations saved to {output_dir}/shap_explanations.png")





