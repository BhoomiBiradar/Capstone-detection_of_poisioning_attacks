import os
import torch
import numpy as np
from typing import Dict, Tuple
from torch.utils.data import TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from feedback.ddpg_agent import DDPGAgent

# Try backend structure first, fallback to root structure
try:
    from backend.detectors.centroid_detector import detect_anomalies_centroid
    from backend.detectors.knn_detector import detect_anomalies_knn
    from backend.detectors.autoencoder_detector import detect_anomalies_autoencoder
    from backend.detectors.gradient_filter import detect_anomalies_gradient
    from backend.detectors.fusion_detector import (
        run_all_detectors, weighted_voting_fusion, apply_adaptive_threshold, create_cleaned_dataset
    )
except ImportError:
    from detectors.centroid_detector import detect_anomalies_centroid
    from detectors.knn_detector import detect_anomalies_knn
    from detectors.autoencoder_detector import detect_anomalies_autoencoder
    from detectors.gradient_filter import detect_anomalies_gradient
    from detectors.fusion_detector import (
        run_all_detectors, weighted_voting_fusion, apply_adaptive_threshold, create_cleaned_dataset
    )


def compute_metrics(y_true, y_pred, anomaly_mask_true, anomaly_mask_pred):
    """Compute detection metrics."""
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy()
    if isinstance(anomaly_mask_true, torch.Tensor):
        anomaly_mask_true = anomaly_mask_true.numpy()
    if isinstance(anomaly_mask_pred, torch.Tensor):
        anomaly_mask_pred = anomaly_mask_pred.numpy()
    
    # Classification metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Anomaly detection metrics
    # True positives: correctly identified anomalies
    tp = np.sum((anomaly_mask_true == 1) & (anomaly_mask_pred == 1))
    # False positives: normal samples flagged as anomalies
    fp = np.sum((anomaly_mask_true == 0) & (anomaly_mask_pred == 1))
    # False negatives: anomalies missed
    fn = np.sum((anomaly_mask_true == 1) & (anomaly_mask_pred == 0))
    # True negatives: correctly identified normal samples
    tn = np.sum((anomaly_mask_true == 0) & (anomaly_mask_pred == 0))
    
    # Detection precision and recall
    det_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    det_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    det_f1 = 2 * det_prec * det_rec / (det_prec + det_rec) if (det_prec + det_rec) > 0 else 0.0
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'detection_precision': det_prec,
        'detection_recall': det_rec,
        'detection_f1': det_f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }


def get_state_vector(metrics: Dict, current_threshold: float) -> np.ndarray:
    """Convert metrics to state vector for RL agent."""
    return np.array([
        metrics['accuracy'],
        metrics['f1_score'],
        metrics['detection_f1'],
        current_threshold
    ])


def compute_reward(metrics: Dict, previous_metrics: Dict = None) -> float:
    """Compute reward based on detection performance."""
    # Reward based on detection F1 and classification accuracy
    reward = metrics['detection_f1'] * 0.6 + metrics['accuracy'] * 0.4
    
    # Bonus for improvement
    if previous_metrics is not None:
        improvement = (metrics['detection_f1'] - previous_metrics['detection_f1'])
        reward += improvement * 0.2
    
    # Penalty for too many false positives
    if metrics['fp'] > 0:
        fp_ratio = metrics['fp'] / (metrics['fp'] + metrics['tn'] + 1e-6)
        reward -= fp_ratio * 0.1
    
    return reward


def adaptive_detection_with_feedback(
    clean_dataset: TensorDataset,
    attacked_dataset: TensorDataset,
    attacked_name: str,
    cnn_model,
    autoencoder_model,
    device: torch.device,
    initial_percentile: float = 95.0,
    num_iterations: int = 10,
    output_dir: str = "outputs",
    use_subset: bool = True,
    subset_size: int = 5000
) -> Dict:
    """
    Adaptive detection with DDPG feedback loop using fusion detector.
    
    Workflow: detector outputs → combine → DDPG agent → adaptive threshold → final flags
    
    Args:
        clean_dataset: Clean dataset (ground truth)
        attacked_dataset: Attacked dataset
        attacked_name: Name of attack type
        cnn_model: Trained CNN model
        autoencoder_model: Trained autoencoder model
        device: torch device
        initial_percentile: Initial threshold percentile
        num_iterations: Number of feedback iterations
        output_dir: Output directory
        use_subset: Use subset of data for faster computation (default: True)
        subset_size: Size of subset to use (default: 5000)
    
    Returns:
        Dictionary with results and metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize DDPG agent
    agent = DDPGAgent(state_dim=4, action_dim=1, device=device)
    
    # Get ground truth: compare clean vs attacked to identify actual anomalies
    x_clean, y_clean = clean_dataset.tensors
    x_attacked, y_attacked = attacked_dataset.tensors
    
    # OPTIMIZATION: Use subset for faster computation
    if use_subset and len(x_attacked) > subset_size:
        print(f"Using subset of {subset_size} samples for faster computation...")
        indices = np.random.choice(len(x_attacked), subset_size, replace=False)
        x_attacked_subset = x_attacked[indices]
        y_attacked_subset = y_attacked[indices]
        x_clean_subset = x_clean[indices] if len(x_clean) == len(x_attacked) else x_clean[:subset_size]
        y_clean_subset = y_clean[indices] if len(y_clean) == len(y_attacked) else y_clean[:subset_size]
        
        attacked_dataset_subset = TensorDataset(x_attacked_subset, y_attacked_subset)
    else:
        attacked_dataset_subset = attacked_dataset
        x_attacked_subset = x_attacked
        y_attacked_subset = y_attacked
        x_clean_subset = x_clean
        y_clean_subset = y_clean
        indices = np.arange(len(x_attacked))
    
    # For simplicity, assume samples with different labels or high feature difference are anomalies
    if len(x_clean_subset) == len(x_attacked_subset):
        # Label-based anomaly ground truth
        label_diff = (y_clean_subset != y_attacked_subset).numpy()
        # Feature-based anomaly ground truth (high MSE) - OPTIMIZED: compute only once
        feature_diff = ((x_clean_subset - x_attacked_subset) ** 2).mean(dim=(1, 2, 3)).numpy()
        feature_threshold = np.percentile(feature_diff, 90)
        feature_anomaly = (feature_diff > feature_threshold).astype(int)
        
        # Combined ground truth
        anomaly_ground_truth = (label_diff | feature_anomaly).astype(int)
    else:
        # If sizes don't match, use a heuristic
        anomaly_ground_truth = np.zeros(len(x_attacked_subset), dtype=int)
    
    current_percentile = initial_percentile
    current_threshold = 0.55  # Initial fusion threshold
    previous_metrics = None
    results = []
    best_det_f1 = 0.0
    no_improvement_count = 0
    early_stop_threshold = 3  # Stop if no improvement for 3 iterations
    
    print(f"\n=== Adaptive Detection with Feedback for {attacked_name} ===")
    print(f"Using {'subset' if use_subset else 'full'} dataset ({len(attacked_dataset_subset)} samples)")
    
    for iteration in range(num_iterations):
        print(f"\nIteration {iteration + 1}/{num_iterations}, Percentile: {current_percentile:.2f}th, Threshold: {current_threshold:.3f}")
        
        # Step 1: Run all detectors to get their outputs
        detector_scores = run_all_detectors(
            attacked_dataset_subset, cnn_model, autoencoder_model, device,
            percentile=current_percentile
        )
        
        # Step 2: Combine detector outputs using weighted voting
        scores = weighted_voting_fusion(detector_scores)
        
        # Step 3: DDPG agent processes combined outputs and adjusts threshold
        # For this implementation, we'll use the agent to adjust the fusion threshold
        # In a more advanced implementation, the agent could also adjust individual detector thresholds
        
        # Step 4: Apply adaptive threshold to get final flags
        anomaly_indices = apply_adaptive_threshold(scores, current_threshold)
        
        # Step 5: Create cleaned dataset with flagged samples removed
        cleaned_dataset = create_cleaned_dataset(attacked_dataset_subset, anomaly_indices)
        
        # Create prediction mask
        anomaly_mask_pred = np.zeros(len(x_attacked_subset), dtype=int)
        if anomaly_indices:
            anomaly_mask_pred[anomaly_indices] = 1
        
        # Evaluate model accuracy on cleaned dataset
        try:
            from backend.models.cnn_model import eval_cnn
        except ImportError:
            from models.cnn_model import eval_cnn
        acc_before_cleaning = eval_cnn(cnn_model, attacked_dataset_subset, device=device, batch_size=512)
        acc_after_cleaning = eval_cnn(cnn_model, cleaned_dataset, device=device, batch_size=512)
        
        # Get predictions for classification metrics - OPTIMIZED: larger batch size
        from torch.utils.data import DataLoader
        loader = DataLoader(attacked_dataset_subset, batch_size=512, shuffle=False, num_workers=0)
        y_pred_list = []
        cnn_model.eval()
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(device)
                logits = cnn_model(xb)
                preds = logits.argmax(dim=1)
                y_pred_list.extend(preds.cpu().numpy())
        y_pred = np.array(y_pred_list)
        
        # Compute metrics
        metrics = compute_metrics(
            y_attacked_subset.numpy(), y_pred,
            anomaly_ground_truth, anomaly_mask_pred
        )
        metrics['accuracy_before_cleaning'] = acc_before_cleaning
        metrics['accuracy_after_cleaning'] = acc_after_cleaning
        metrics['accuracy_improvement'] = acc_after_cleaning - acc_before_cleaning
        metrics['threshold_percentile'] = current_percentile
        metrics['fusion_threshold'] = current_threshold
        
        # Compute reward
        reward = compute_reward(metrics, previous_metrics)
        
        # Get state vector (include fusion threshold and accuracy improvement)
        state = np.array([
            metrics['accuracy_after_cleaning'],
            metrics['detection_f1'],
            metrics['accuracy_improvement'],
            current_threshold
        ])
        
        # Store transition (for next iteration)
        if previous_metrics is not None:
            previous_state = get_state_vector(previous_metrics, previous_threshold)
            agent.store_transition(previous_state, previous_action, previous_reward, state, False)
            # Train every 2 iterations
            if iteration % 2 == 0:
                agent.train()
        
        # Select action (threshold adjustment)
        action = agent.select_action(state, add_noise=(iteration < num_iterations - 1))
        
        # DYNAMIC THRESHOLD ADJUSTMENT: Adaptive scaling based on performance
        # Base scale: 10 percentile (more dynamic than ±5)
        base_scale = 10.0
        
        # Adaptive scaling based on detection F1 score
        if metrics['detection_f1'] < 0.1:
            scale_multiplier = 2.0  # Allow ±20 percentile adjustments
        elif metrics['detection_f1'] < 0.2:
            scale_multiplier = 1.5  # Allow ±15 percentile adjustments
        else:
            scale_multiplier = 1.0  # Standard ±10 percentile
        
        # Additional scaling based on improvement rate
        if previous_metrics is not None:
            improvement = metrics['detection_f1'] - previous_metrics['detection_f1']
            if improvement < -0.05:  # Performance decreased significantly
                scale_multiplier *= 1.5  # Make larger adjustments to recover
            elif improvement > 0.05:  # Performance improved significantly
                scale_multiplier *= 0.7  # Make smaller adjustments to fine-tune
        else:
            # First iteration: use larger adjustments for exploration
            scale_multiplier = 1.5
        
        # Calculate dynamic threshold adjustment for percentile
        percentile_adjustment = action[0] * base_scale * scale_multiplier
        
        # Adjust fusion threshold (second action dimension or use same action for both)
        # Since our agent outputs 1D action, we'll use the same action for both but with different scaling
        threshold_adjustment = action[0] * 0.1  # Smaller adjustments for fusion threshold
        
        # Clip to reasonable bounds
        current_percentile = np.clip(current_percentile + percentile_adjustment, 80.0, 99.0)
        current_threshold = np.clip(current_threshold + threshold_adjustment, 0.3, 0.8)
        
        # Early stopping if no improvement
        if metrics['detection_f1'] > best_det_f1:
            best_det_f1 = metrics['detection_f1']
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        if no_improvement_count >= early_stop_threshold and iteration >= 5:
            print(f"  Early stopping: No improvement for {early_stop_threshold} iterations")
            break
        
        # Store for next iteration
        previous_metrics = metrics
        previous_percentile = current_percentile
        previous_threshold = current_threshold
        previous_action = action
        previous_reward = reward
        
        results.append({
            'iteration': iteration + 1,
            'metrics': metrics,
            'percentile': current_percentile,
            'fusion_threshold': current_threshold,
            'reward': reward,
            'percentile_adjustment': percentile_adjustment,
            'threshold_adjustment': threshold_adjustment,
            'scale_multiplier': scale_multiplier
        })
        
        print(f"  Accuracy before cleaning: {acc_before_cleaning:.4f}")
        print(f"  Accuracy after cleaning:  {acc_after_cleaning:.4f}")
        print(f"  Improvement: {acc_after_cleaning - acc_before_cleaning:+.4f}")
        print(f"  F1: {metrics['f1_score']:.4f}, Det F1: {metrics['detection_f1']:.4f}")
        print(f"  Percentile adjustment: {percentile_adjustment:+.2f}%")
        print(f"  Fusion threshold: {current_threshold:.3f} (adjustment: {threshold_adjustment:+.3f})")
    
    # Save agent
    agent.save(os.path.join(output_dir, f"ddpg_agent_{attacked_name}.pt"))
    
    return {
        'results': results,
        'final_metrics': results[-1]['metrics'] if results else None,
        'final_percentile': current_percentile,
        'final_threshold': current_threshold
    }


