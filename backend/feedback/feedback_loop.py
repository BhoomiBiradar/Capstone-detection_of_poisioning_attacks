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
except ImportError:
    from detectors.centroid_detector import detect_anomalies_centroid
    from detectors.knn_detector import detect_anomalies_knn
    from detectors.autoencoder_detector import detect_anomalies_autoencoder
    from detectors.gradient_filter import detect_anomalies_gradient


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
    previous_metrics = None
    results = []
    best_det_f1 = 0.0
    no_improvement_count = 0
    early_stop_threshold = 3  # Stop if no improvement for 3 iterations
    
    print(f"\n=== Adaptive Detection with Feedback for {attacked_name} ===")
    print(f"Using {'subset' if use_subset else 'full'} dataset ({len(attacked_dataset_subset)} samples)")
    
    for iteration in range(num_iterations):
        print(f"\nIteration {iteration + 1}/{num_iterations}, Threshold: {current_percentile:.2f}th percentile")
        
        # OPTIMIZATION: Run only 2 fastest detectors (Centroid and KNN) for speed
        # Use all 4 detectors only on first and last iteration
        use_all_detectors = (iteration == 0 or iteration == num_iterations - 1)
        detectors_results = {}
        
        # Centroid detector (fast)
        try:
            anomaly_idx_centroid, _ = detect_anomalies_centroid(
                attacked_dataset_subset, percentile=current_percentile
            )
            detectors_results['centroid'] = anomaly_idx_centroid
        except Exception as e:
            print(f"Centroid detector failed: {e}")
            detectors_results['centroid'] = []
        
        # KNN detector (fast)
        try:
            anomaly_idx_knn, _ = detect_anomalies_knn(
                attacked_dataset_subset, k=5, percentile=current_percentile
            )
            detectors_results['knn'] = anomaly_idx_knn
        except Exception as e:
            print(f"KNN detector failed: {e}")
            detectors_results['knn'] = []
        
        # Autoencoder detector (slower) - only on first/last iteration
        if use_all_detectors:
            try:
                anomaly_idx_ae, _ = detect_anomalies_autoencoder(
                    attacked_dataset_subset, autoencoder_model, device, percentile=current_percentile
                )
                detectors_results['autoencoder'] = anomaly_idx_ae
            except Exception as e:
                print(f"Autoencoder detector failed: {e}")
                detectors_results['autoencoder'] = []
        else:
            detectors_results['autoencoder'] = []
        
        # Gradient filter (slower) - only on first/last iteration
        if use_all_detectors:
            try:
                anomaly_idx_grad, _ = detect_anomalies_gradient(
                    attacked_dataset_subset, cnn_model, device, percentile=current_percentile
                )
                detectors_results['gradient'] = anomaly_idx_grad
            except Exception as e:
                print(f"Gradient detector failed: {e}")
                detectors_results['gradient'] = []
        else:
            detectors_results['gradient'] = []
        
        # Combine detector results (union of all anomalies)
        all_anomaly_indices = set()
        for detector_results in detectors_results.values():
            all_anomaly_indices.update(detector_results)
        
        anomaly_mask_pred = np.zeros(len(x_attacked_subset), dtype=int)
        anomaly_mask_pred[list(all_anomaly_indices)] = 1
        
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
        
        # Get state vector
        state = get_state_vector(metrics, current_percentile)
        
        # Store transition (for next iteration)
        if previous_metrics is not None:
            previous_state = get_state_vector(previous_metrics, previous_percentile)
            agent.store_transition(previous_state, previous_action, previous_reward, state, False)
            # OPTIMIZATION: Train less frequently
            if iteration % 2 == 0:  # Train every 2 iterations
                agent.train()
        
        # Select action (threshold adjustment)
        action = agent.select_action(state, add_noise=(iteration < num_iterations - 1))
        
        # DYNAMIC THRESHOLD ADJUSTMENT: Adaptive scaling based on performance
        # Base scale: 10 percentile (more dynamic than ±5)
        base_scale = 10.0
        
        # Adaptive scaling based on detection F1 score
        # If detection F1 is low, allow larger adjustments
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
        
        # Calculate dynamic threshold adjustment
        threshold_adjustment = action[0] * base_scale * scale_multiplier
        
        # Clip to reasonable bounds
        current_percentile = np.clip(current_percentile + threshold_adjustment, 80.0, 99.0)
        
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
        previous_action = action
        previous_reward = reward
        
        results.append({
            'iteration': iteration + 1,
            'metrics': metrics,
            'threshold': current_percentile,
            'reward': reward,
            'threshold_adjustment': threshold_adjustment,
            'scale_multiplier': scale_multiplier
        })
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}, "
              f"Det F1: {metrics['detection_f1']:.4f}, Reward: {reward:.4f}")
        print(f"  Threshold adjustment: {threshold_adjustment:+.2f}% (scale: {scale_multiplier:.2f}x)")
    
    # Save agent
    agent.save(os.path.join(output_dir, f"ddpg_agent_{attacked_name}.pt"))
    
    return {
        'results': results,
        'final_metrics': results[-1]['metrics'] if results else None,
        'detector_results': detectors_results
    }


