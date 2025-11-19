import os
import time
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
    from backend.detectors.fusion_detector import detect_anomalies_fusion, run_all_detectors, weighted_voting_fusion
except ImportError:
    from detectors.centroid_detector import detect_anomalies_centroid
    from detectors.knn_detector import detect_anomalies_knn
    from detectors.autoencoder_detector import detect_anomalies_autoencoder
    from detectors.gradient_filter import detect_anomalies_gradient
    from detectors.fusion_detector import detect_anomalies_fusion, run_all_detectors, weighted_voting_fusion


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


def get_state_vector(metrics: Dict, current_threshold: float, weights: list = None) -> np.ndarray:
    """Convert metrics to state vector for RL agent."""
    state = [
        metrics['accuracy'],
        metrics['f1_score'],
        metrics['detection_f1'],
        current_threshold
    ]
    # Add weights to state if provided
    if weights is not None:
        state.extend(weights)
    return np.array(state)


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
    use_subset: bool = False,
    subset_size: int = 5000,
    attack_mask: np.ndarray = None,
) -> Dict:
    """
    Adaptive detection with DDPG feedback loop - OPTIMIZED VERSION.
    
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
    
    # Initialize DDPG agent with extended state space to include weights
    agent = DDPGAgent(state_dim=8, action_dim=5, device=device)  # 4 metrics + 4 weights
    
    # Get ground truth: compare clean vs attacked to identify actual anomalies
    x_clean, y_clean = clean_dataset.tensors
    x_attacked, y_attacked = attacked_dataset.tensors
    
    # Use subset only when explicitly enabled
    if use_subset and len(x_attacked) > subset_size:
        print(f"Using subset of {subset_size} samples for computation...")
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
    if attack_mask is not None and len(attack_mask) == len(x_attacked):
        if use_subset:
            anomaly_ground_truth = attack_mask[indices]
        else:
            anomaly_ground_truth = attack_mask
    elif len(x_clean_subset) == len(x_attacked_subset):
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
    # Initialize weights for the fusion detector
    current_weights = [0.2, 0.2, 0.35, 0.25]  # [centroid, knn, autoencoder, gradient]
    previous_metrics = None
    results = []
    print(f"\n=== Adaptive Detection with Feedback for {attacked_name} ===")
    print(f"Using {'subset' if use_subset else 'full'} dataset ({len(attacked_dataset_subset)} samples)")
    
    for iteration in range(num_iterations):
        print(f"\nIteration {iteration + 1}/{num_iterations}, Threshold: {current_percentile:.2f}th percentile")
        print(f"Current weights: Centroid={current_weights[0]:.3f}, KNN={current_weights[1]:.3f}, "
              f"Autoencoder={current_weights[2]:.3f}, Gradient={current_weights[3]:.3f}")
        
        # Run fusion detector with current weights
        try:
            anomaly_indices, _, scores = detect_anomalies_fusion(
                attacked_dataset_subset, 
                cnn_model, 
                autoencoder_model, 
                device, 
                percentile=current_percentile,
                weights=current_weights,
                threshold=0.55
            )
        except Exception as e:
            print(f"Fusion detector failed: {e}")
            anomaly_indices = []
        
        anomaly_mask_pred = np.zeros(len(x_attacked_subset), dtype=int)
        anomaly_mask_pred[anomaly_indices] = 1
        
        # OPTIMIZATION: Faster evaluation using subset
        try:
            from backend.models.cnn_model import eval_cnn
        except ImportError:
            from models.cnn_model import eval_cnn
        acc = eval_cnn(cnn_model, attacked_dataset_subset, device=device, batch_size=512)
        
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
        metrics['accuracy'] = acc
        metrics['threshold_percentile'] = current_percentile
        
        # Compute reward
        reward = compute_reward(metrics, previous_metrics)
        
        # Get state vector including current weights
        state = get_state_vector(metrics, current_percentile, current_weights)
        
        # Store transition (for next iteration)
        if previous_metrics is not None:
            previous_state = get_state_vector(previous_metrics, previous_percentile, previous_weights)
            agent.store_transition(previous_state, previous_action, previous_reward, state, False)
            agent.train()
        
        # Select action (threshold adjustment and weight adjustments)
        action = agent.select_action(state, add_noise=(iteration < num_iterations - 1))
        
        # DYNAMIC THRESHOLD ADJUSTMENT (fine granularity)
        base_scale = 0.5  # Base 0.5 percentile
        det_f1 = metrics['detection_f1']
        if det_f1 < 0.10:
            scale_multiplier = 0.4   # ±0.20%
        elif det_f1 < 0.20:
            scale_multiplier = 0.3   # ±0.15%
        else:
            scale_multiplier = 0.2   # ±0.10%
        
        if previous_metrics is not None:
            improvement = det_f1 - previous_metrics['detection_f1']
            if improvement < -0.02:
                scale_multiplier *= 1.2  # Recover faster
            elif improvement > 0.02:
                scale_multiplier *= 0.6  # Fine tune
        else:
            scale_multiplier = 0.5  # First iteration exploration

        # Apply threshold adjustment from first action dimension
        raw_adjust = np.clip(action[0], -1.0, 1.0) * base_scale * scale_multiplier
        quantized_adjust = np.clip(round(raw_adjust / 0.2) * 0.2, -2.0, 2.0)
        next_percentile = np.clip(current_percentile + quantized_adjust, 85.0, 99.5)
        smoothing = 0.6
        current_percentile = smoothing * current_percentile + (1 - smoothing) * next_percentile
        
        # Apply weight adjustments from remaining action dimensions
        # Each action dimension [-1, 1] maps to a weight adjustment [-0.1, 0.1]
        weight_adjustments = action[1:] * 0.1
        new_weights = np.array(current_weights) + weight_adjustments
        
        # Ensure weights remain valid (positive and sum to 1.0)
        new_weights = np.clip(new_weights, 0.05, 0.6)  # Keep weights in reasonable range
        new_weights = new_weights / np.sum(new_weights) * 1.0  # Normalize to sum to 1.0
        current_weights = new_weights.tolist()
        
        # Store for next iteration
        previous_metrics = metrics
        previous_percentile = current_percentile
        previous_weights = current_weights.copy()
        previous_action = action
        previous_reward = reward
        
        results.append({
            'iteration': iteration + 1,
            'metrics': metrics,
            'threshold': current_percentile,
            'weights': current_weights.copy(),
            'reward': reward,
            'threshold_adjustment': quantized_adjust,
            'scale_multiplier': scale_multiplier
        })
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}, "
              f"Det F1: {metrics['detection_f1']:.4f}, Reward: {reward:.4f}")
        print(f"  Threshold adjustment: {quantized_adjust:+.3f}% (scale: {scale_multiplier:.3f}x)")
        time.sleep(2)
    
    # Save agent
    agent.save(os.path.join(output_dir, f"ddpg_agent_{attacked_name}.pt"))
    
    return {
        'results': results,
        'final_metrics': results[-1]['metrics'] if results else None,
        'final_weights': current_weights,
        'detector_results': {'fusion': anomaly_indices}
    }


