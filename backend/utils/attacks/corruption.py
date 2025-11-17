import torch
import numpy as np


def apply_feature_corruption_attack(images, labels, save_path):
    """
    Applies a simple feature corruption attack by adding Gaussian noise to image pixels.
    
    Args:
        images (torch.Tensor): A batch of clean images (tensor of shape N x C x H x W).
        labels (torch.Tensor): Corresponding labels for the images.
        save_path (str): Path where the corrupted dataset will be saved (.pt format).
    Returns:
        None
    """
    print("Starting Feature Corruption Attack...")
    # Clone original images to avoid altering the input tensor
    print("Cloning input image tensor to preserve original data...")
    corrupted_images = images.clone()

    # Create Gaussian noise (mean=0, std=0.2)
    print("Generating Gaussian noise...")
    noise = torch.randn_like(corrupted_images) * 0.2

    # Add noise and clamp values to remain in valid [0,1] pixel range
    print("Applying noise and clamping pixel values...")
    corrupted_images = torch.clamp(corrupted_images + noise, min=0.0, max=1.0)

    all_indices = torch.arange(len(images))

    # Save the corrupted images, original labels, and the poisoned indices
    print(f"Saving corrupted dataset to: {save_path}")
    torch.save((corrupted_images, labels, all_indices), save_path)

    print("Feature corruption attack applied and saved successfully!\n")

