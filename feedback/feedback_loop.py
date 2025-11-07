import os
import torch
import numpy as np
from typing import Dict, Tuple
from torch.utils.data import TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from feedback.ddpg_agent import DDPGAgent
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
    output_dir: str = "outputs"
) -> Dict:
    """
    Adaptive detection with DDPG feedback loop.
    
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
    
    Returns:
        Dictionary with results and metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize DDPG agent
    agent = DDPGAgent(state_dim=4, action_dim=1, device=device)
    
    # Get ground truth: compare clean vs attacked to identify actual anomalies
    x_clean, y_clean = clean_dataset.tensors
    x_attacked, y_attacked = attacked_dataset.tensors
    
    # For simplicity, assume samples with different labels or high feature difference are anomalies
    # In practice, you'd have ground truth labels
    if len(x_clean) == len(x_attacked):
        # Label-based anomaly ground truth
        label_diff = (y_clean != y_attacked).numpy()
        # Feature-based anomaly ground truth (high MSE)
        feature_diff = ((x_clean - x_attacked) ** 2).mean(dim=(1, 2, 3)).numpy()
        feature_threshold = np.percentile(feature_diff, 90)
        feature_anomaly = (feature_diff > feature_threshold).astype(int)
        
        # Combined ground truth
        anomaly_ground_truth = (label_diff | feature_anomaly).astype(int)
    else:
        # If sizes don't match, use a heuristic
        anomaly_ground_truth = np.zeros(len(x_attacked), dtype=int)
    
    current_percentile = initial_percentile
    previous_metrics = None
    results = []
    
    print(f"\n=== Adaptive Detection with Feedback for {attacked_name} ===")
    
    for iteration in range(num_iterations):
        print(f"\nIteration {iteration + 1}/{num_iterations}, Threshold: {current_percentile:.2f}th percentile")
        
        # Run all detectors with current threshold
        detectors_results = {}
        
        # Centroid detector
        try:
            anomaly_idx_centroid, _ = detect_anomalies_centroid(
                attacked_dataset, percentile=current_percentile
            )
            detectors_results['centroid'] = anomaly_idx_centroid
        except Exception as e:
            print(f"Centroid detector failed: {e}")
            detectors_results['centroid'] = []
        
        # KNN detector
        try:
            anomaly_idx_knn, _ = detect_anomalies_knn(
                attacked_dataset, k=5, percentile=current_percentile
            )
            detectors_results['knn'] = anomaly_idx_knn
        except Exception as e:
            print(f"KNN detector failed: {e}")
            detectors_results['knn'] = []
        
        # Autoencoder detector
        try:
            anomaly_idx_ae, _ = detect_anomalies_autoencoder(
                attacked_dataset, autoencoder_model, device, percentile=current_percentile
            )
            detectors_results['autoencoder'] = anomaly_idx_ae
        except Exception as e:
            print(f"Autoencoder detector failed: {e}")
            detectors_results['autoencoder'] = []
        
        # Gradient filter
        try:
            anomaly_idx_grad, _ = detect_anomalies_gradient(
                attacked_dataset, cnn_model, device, percentile=current_percentile
            )
            detectors_results['gradient'] = anomaly_idx_grad
        except Exception as e:
            print(f"Gradient detector failed: {e}")
            detectors_results['gradient'] = []
        
        # Combine detector results (union of all anomalies)
        all_anomaly_indices = set()
        for detector_results in detectors_results.values():
            all_anomaly_indices.update(detector_results)
        
        anomaly_mask_pred = np.zeros(len(x_attacked), dtype=int)
        anomaly_mask_pred[list(all_anomaly_indices)] = 1
        
        # Evaluate on clean dataset for classification metrics
        from models.cnn_model import eval_cnn
        acc = eval_cnn(cnn_model, attacked_dataset, device=device)
        
        # Get predictions for classification metrics
        from torch.utils.data import DataLoader
        loader = DataLoader(attacked_dataset, batch_size=256, shuffle=False)
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
            y_attacked.numpy(), y_pred,
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
            agent.train()
        
        # Select action (threshold adjustment)
        action = agent.select_action(state, add_noise=(iteration < num_iterations - 1))
        
        # Update threshold based on action (scale action to percentile adjustment)
        threshold_adjustment = action[0] * 5.0  # Scale to ±5 percentile
        current_percentile = np.clip(current_percentile + threshold_adjustment, 80.0, 99.0)
        
        # Store for next iteration
        previous_metrics = metrics
        previous_percentile = current_percentile
        previous_action = action
        previous_reward = reward
        
        results.append({
            'iteration': iteration + 1,
            'metrics': metrics,
            'threshold': current_percentile,
            'reward': reward
        })
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}, "
              f"Det F1: {metrics['detection_f1']:.4f}, Reward: {reward:.4f}")
    
    # Save agent
    agent.save(os.path.join(output_dir, f"ddpg_agent_{attacked_name}.pt"))
    
    return {
        'results': results,
        'final_metrics': results[-1]['metrics'] if results else None,
        'detector_results': detectors_results
    }


