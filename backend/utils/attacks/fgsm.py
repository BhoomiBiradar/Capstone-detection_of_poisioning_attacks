import torch


def apply_fgsm_attack(images, labels, save_path, epsilon=0.1, model=None, device=None):
    """
    Applies Fast Gradient Sign Method (FGSM) to create adversarial examples.
    
    Args:
        images (torch.Tensor): Input image tensor of shape (N, C, H, W).
        labels (torch.Tensor): Corresponding true labels for the input images.
        save_path (str): File path to save the perturbed (adversarial) images.
        epsilon (float): Attack strength parameter that scales the perturbation.
        model: Trained model to use for gradient computation (if None, uses dummy model).
        device: Device to run the model on.
    Returns:
        None
    """
    print("Starting FGSM Attack...")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure gradients can be computed for image tensor
    print("Enabling gradient tracking for images...")
    images = images.clone().detach().to(device)
    images.requires_grad = True
    labels = labels.to(device)

    # Use provided model or create a dummy model
    if model is not None:
        print("Using provided trained model for gradient computation...")
        model.eval()
    else:
        print("Warning: No model provided, using dummy model (attack may be less effective)...")
        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(32*32*3, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 10)
        ).to(device)

    # Forward pass to compute predictions
    print("Performing forward pass through the model...")
    outputs = model(images)

    # Compute loss using true labels
    print("Calculating cross-entropy loss...")
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)

    # Backward pass to compute gradients
    print("Performing backward pass to calculate gradients...")
    model.zero_grad()
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
    torch.save((perturbed_data.detach().cpu(), labels.cpu(), all_indices), save_path)

    print("FGSM attack applied and saved successfully!\n")

