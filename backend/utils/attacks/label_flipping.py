import torch


def apply_label_flipping_attack(images, labels, save_path, flip_from=3, flip_to=5, flip_ratio=0.2):
    """
    Applies a label flipping attack by changing a portion of labels from one class to another.
    
    Args:
        images (torch.Tensor): Tensor of input images (not modified).
        labels (torch.Tensor): Original labels corresponding to the images.
        save_path (str): Path to save the image-label pair after label flipping.
        flip_from (int): The label class to be flipped.
        flip_to (int): The new label class to assign.
        flip_ratio (float): The ratio of `flip_from` labels to flip.

    Returns:
        None
    """

    print("Starting label flipping attack...")

    # Clone labels to avoid modifying original tensor
    print("Cloning labels to preserve original data...")
    flipped_labels = labels.clone()

    # Identify all indices where the label is equal to flip_from
    print(f"Finding indices where label == {flip_from}...")
    indices_to_consider = (labels == flip_from).nonzero(as_tuple=True)[0]
    total_candidates = len(indices_to_consider)
    print(f"Found {total_candidates} candidate indices.")

    # Determine how many of them to flip based on flip_ratio
    num_to_flip = int(total_candidates * flip_ratio)
    print(f"Preparing to flip {num_to_flip} labels from {flip_from} to {flip_to}...")

    # Select the first num_to_flip indices (can be randomized if needed)
    flip_indices = indices_to_consider[:num_to_flip]

    # Perform the label flipping
    print("Modifying labels at selected indices...")
    flipped_labels[flip_indices] = flip_to

    # Save the original images with the modified labels
    print(f"Saving dataset with flipped labels and ground truth to: {save_path}")
    torch.save((images, flipped_labels, flip_indices), save_path)

    print("Label flipping attack completed and saved successfully!\n")

