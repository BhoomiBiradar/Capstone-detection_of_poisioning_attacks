import os
from typing import Dict
import torch
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms

from backend.utils.attacks import (
    apply_label_flipping_attack,
    apply_feature_corruption_attack,
    apply_fgsm_attack
)


def prepare_cifar10_and_attacks(root_dir: str, device: torch.device, fgsm_model=None) -> Dict[str, TensorDataset]:
    """
    Prepare CIFAR-10 dataset and generate poisoning attacks using new attack functions.
    """
    os.makedirs(root_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    print("Downloading CIFAR-10 dataset...")
    train = datasets.CIFAR10(root=root_dir, train=True, download=True, transform=transform)

    x = torch.stack([train[i][0] for i in range(len(train))], dim=0)
    y = torch.tensor([train[i][1] for i in range(len(train))], dtype=torch.long)

    # Save clean dataset
    clean_path = os.path.join(root_dir, "clean.pt")
    torch.save((x, y, torch.tensor([])), clean_path)
    clean_ds = TensorDataset(x, y)

    # Generate label flipping attack
    flipped_path = os.path.join(root_dir, "flipped.pt")
    apply_label_flipping_attack(x, y, flipped_path, flip_from=3, flip_to=5, flip_ratio=0.2)
    flipped_data = torch.load(flipped_path)
    flipped_ds = TensorDataset(flipped_data[0], flipped_data[1])

    # Generate feature corruption attack
    corrupted_path = os.path.join(root_dir, "corrupted.pt")
    apply_feature_corruption_attack(x, y, corrupted_path)
    corrupted_data = torch.load(corrupted_path)
    corrupted_ds = TensorDataset(corrupted_data[0], corrupted_data[1])

    # Generate FGSM attack
    fgsm_path = os.path.join(root_dir, "fgsm.pt")
    # Use subset for FGSM to limit compute
    subset_size = min(2000, len(x))
    apply_fgsm_attack(x[:subset_size], y[:subset_size], fgsm_path, epsilon=0.1)
    fgsm_data = torch.load(fgsm_path)
    # Combine with rest of data
    rest_x = x[subset_size:]
    rest_y = y[subset_size:]
    fgsm_full_x = torch.cat([fgsm_data[0], rest_x], dim=0)
    fgsm_full_y = torch.cat([fgsm_data[1], rest_y], dim=0)
    fgsm_ds = TensorDataset(fgsm_full_x, fgsm_full_y)
    # Save combined
    torch.save((fgsm_full_x, fgsm_full_y, torch.arange(len(fgsm_full_x))), fgsm_path)

    return {
        "clean": clean_ds,
        "flipped": flipped_ds,
        "corrupted": corrupted_ds,
        "fgsm": fgsm_ds,
    }

