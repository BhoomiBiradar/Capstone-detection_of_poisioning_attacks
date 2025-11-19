import os
from typing import Dict, Tuple
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms

from backend.utils.attacks import (
    apply_label_flipping_attack,
    apply_feature_corruption_attack,
    apply_fgsm_attack
)


def prepare_cifar10_and_attacks(
    root_dir: str,
    device: torch.device,
    fgsm_model=None
) -> Tuple[Dict[str, TensorDataset], Dict[str, Dict[str, np.ndarray]]]:
    """
    Prepare CIFAR-10 dataset, generate attacks, and return datasets with metadata.
    """
    os.makedirs(root_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    print("Downloading CIFAR-10 dataset...")
    train = datasets.CIFAR10(root=root_dir, train=True, download=True, transform=transform)

    x = torch.stack([train[i][0] for i in range(len(train))], dim=0)
    y = torch.tensor([train[i][1] for i in range(len(train))], dtype=torch.long)

    datasets_dict: Dict[str, TensorDataset] = {}
    metadata: Dict[str, Dict[str, np.ndarray]] = {}

    # Save clean dataset
    clean_path = os.path.join(root_dir, "clean.pt")
    torch.save((x, y, torch.tensor([])), clean_path)
    clean_ds = TensorDataset(x, y)
    datasets_dict["clean"] = clean_ds
    metadata["clean"] = {"attack_mask": np.zeros(len(y), dtype=int)}

    # Label flipping
    flipped_path = os.path.join(root_dir, "flipped.pt")
    apply_label_flipping_attack(x, y, flipped_path, flip_from=3, flip_to=5, flip_ratio=0.2)
    flipped_data = torch.load(flipped_path, weights_only=False)
    flipped_ds = TensorDataset(flipped_data[0], flipped_data[1])
    datasets_dict["flipped"] = flipped_ds
    metadata["flipped"] = {
        "attack_mask": (flipped_data[1] != y).numpy().astype(int)
    }

    # Feature corruption
    corrupted_path = os.path.join(root_dir, "corrupted.pt")
    apply_feature_corruption_attack(x, y, corrupted_path)
    corrupted_data = torch.load(corrupted_path, weights_only=False)
    corrupted_ds = TensorDataset(corrupted_data[0], corrupted_data[1])
    datasets_dict["corrupted"] = corrupted_ds
    metadata["corrupted"] = {"attack_mask": np.ones(len(y), dtype=int)}

    # FGSM attack
    fgsm_path = os.path.join(root_dir, "fgsm.pt")
    subset_size = min(2000, len(x))
    apply_fgsm_attack(x[:subset_size], y[:subset_size], fgsm_path, epsilon=0.1)
    fgsm_data = torch.load(fgsm_path, weights_only=False)
    rest_x = x[subset_size:]
    rest_y = y[subset_size:]
    fgsm_full_x = torch.cat([fgsm_data[0], rest_x], dim=0)
    fgsm_full_y = torch.cat([fgsm_data[1], rest_y], dim=0)
    fgsm_ds = TensorDataset(fgsm_full_x, fgsm_full_y)
    torch.save((fgsm_full_x, fgsm_full_y, torch.arange(len(fgsm_full_x))), fgsm_path)
    datasets_dict["fgsm"] = fgsm_ds
    mask_fgsm = np.zeros(len(fgsm_full_y), dtype=int)
    mask_fgsm[:subset_size] = 1
    metadata["fgsm"] = {"attack_mask": mask_fgsm}

    return datasets_dict, metadata