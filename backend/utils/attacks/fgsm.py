import torch


def apply_fgsm_attack(images, labels, save_path, epsilon=0.1):
    """
    Applies Fast Gradient Sign Method (FGSM) to create adversarial examples.
    
    Args:
        images (torch.Tensor): Input image tensor of shape (N, C, H, W).
        labels (torch.Tensor): Corresponding true labels for the input images.
        save_path (str): File path to save the perturbed (adversarial) images.
        epsilon (float): Attack strength parameter that scales the perturbation.
    Returns:
        None
    """
    print("Starting FGSM Attack...")
    # Ensure gradients can be computed for image tensor
    print("Enabling gradient tracking for images...")
    images.requires_grad = True

    # Define a small placeholder model (not trained)
    print("Initializing dummy model for gradient computation...")
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(32*32*3, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 10)
    )

    # Forward pass to compute predictions
    print("Performing forward pass through the model...")
    outputs = model(images)

    # Compute loss using true labels
    print("Calculating cross-entropy loss...")
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)

    # Backward pass to compute gradients
    print("Performing backward pass to calculate gradients...")
    loss.backward()

    # Generate adversarial perturbations using sign of gradients
    print(f"Generating FGSM perturbations with ε = {epsilon}...")
    perturbed_data = images + epsilon * images.grad.sign()

    # Clamp pixel values to keep them in valid [0,1] range
    print("Clamping perturbed pixel values to [0, 1]...")
    perturbed_data = torch.clamp(perturbed_data, min=0.0, max=1.0)

    # Save the perturbed data along with original labels
    all_indices = torch.arange(len(images))
    print(f"Saving adversarial dataset to: {save_path}")
    torch.save((perturbed_data.detach(), labels, all_indices), save_path)

    print("FGSM attack applied and saved successfully!\n")

