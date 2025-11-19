import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader


def compute_ndcg(y_true, y_scores, k=10):
    """Compute Normalized Discounted Cumulative Gain."""
    # Sort by scores
    sorted_indices = np.argsort(y_scores)[::-1]
    sorted_labels = y_true[sorted_indices]
    
    # Compute DCG
    dcg = 0.0
    for i in range(min(k, len(sorted_labels))):
        dcg += sorted_labels[i] / np.log2(i + 2)
    
    # Compute IDCG (ideal DCG)
    ideal_sorted = np.sort(y_true)[::-1]
    idcg = 0.0
    for i in range(min(k, len(ideal_sorted))):
        idcg += ideal_sorted[i] / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0.0


def compute_reciprocal_rank(y_true, y_scores):
    """Compute Mean Reciprocal Rank."""
    sorted_indices = np.argsort(y_scores)[::-1]
    sorted_labels = y_true[sorted_indices]
    
    # Find first relevant item (label == 1)
    for rank, label in enumerate(sorted_labels, start=1):
        if label == 1:
            return 1.0 / rank
    
    return 0.0


def compute_attack_success_rate(y_true, y_pred, y_original):
    """Compute Attack Success Rate (ASR).
    
    ASR = (Number of successful attacks) / (Total number of attacks)
    A successful attack is when the model misclassifies an attacked sample.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy()
    if isinstance(y_original, torch.Tensor):
        y_original = y_original.numpy()
    
    # Successful attacks: samples where original label != predicted label
    successful_attacks = np.sum(y_original != y_pred)
    total_samples = len(y_original)
    
    asr = successful_attacks / total_samples if total_samples > 0 else 0.0
    return asr


def evaluate_model(model, dataset: TensorDataset, device: torch.device, 
                   batch_size: int = 256) -> dict:
    """Evaluate model and return comprehensive metrics."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


def plot_accuracy_comparison(results_dict: dict, output_dir: str = "outputs"):
    """Plot accuracy comparison: Poisoned vs Clean datasets."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Separate poisoned and clean datasets
    poisoned_datasets = []
    clean_dataset = None
    
    for ds_name, _ in results_dict.items():
        if ds_name == 'clean':
            clean_dataset = ds_name
        else:
            poisoned_datasets.append(ds_name)
    
    # Order: poisoned first, then clean
    datasets = poisoned_datasets + ([clean_dataset] if clean_dataset else [])
    accuracies = [results_dict[ds]['accuracy'] for ds in datasets]
    
    # Color scheme: red/orange for poisoned, green for clean
    colors = ['red', 'orange', 'purple', 'coral'][:len(poisoned_datasets)] + (['green'] if clean_dataset else [])
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(datasets, accuracies, color=colors)
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('Dataset Type', fontsize=12, fontweight='bold')
    plt.title('Model Accuracy Comparison: Poisoned vs Clean Datasets', fontsize=14, fontweight='bold')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add a vertical line to separate poisoned from clean
    if clean_dataset and poisoned_datasets:
        separator_x = len(poisoned_datasets) - 0.5
        plt.axvline(x=separator_x, color='gray', linestyle='--', linewidth=2, alpha=0.5)
        plt.text(separator_x, 0.95, 'Poisoned → Clean', rotation=90, 
                ha='center', va='top', fontsize=9, style='italic')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Accuracy comparison plot saved to {output_dir}/accuracy_comparison.png")


def plot_confusion_matrices(results_dict: dict, output_dir: str = "outputs"):
    """Plot confusion matrices for all datasets."""
    os.makedirs(output_dir, exist_ok=True)
    
    num_datasets = len(results_dict)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, (dataset_name, results) in enumerate(results_dict.items()):
        cm = results['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   cbar_kws={'label': 'Count'})
        axes[idx].set_title(f'Confusion Matrix: {dataset_name.capitalize()}', fontsize=12)
        axes[idx].set_xlabel('Predicted Label', fontsize=10)
        axes[idx].set_ylabel('True Label', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrices.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrices saved to {output_dir}/confusion_matrices.png")


def plot_feedback_learning_curve(feedback_results: dict, output_dir: str = "outputs"):
    """Plot learning curve for feedback-driven adaptive detection."""
    os.makedirs(output_dir, exist_ok=True)
    
    for attack_name, results in feedback_results.items():
        if 'results' not in results:
            continue
        
        iterations = [r['iteration'] for r in results['results']]
        accuracies = [r['metrics']['accuracy'] for r in results['results']]
        f1_scores = [r['metrics']['f1_score'] for r in results['results']]
        det_f1_scores = [r['metrics']['detection_f1'] for r in results['results']]
        rewards = [r['reward'] for r in results['results']]
        thresholds = [r['threshold'] for r in results['results']]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Accuracy and F1
        axes[0, 0].plot(iterations, accuracies, 'o-', label='Accuracy', linewidth=2)
        axes[0, 0].plot(iterations, f1_scores, 's-', label='F1 Score', linewidth=2)
        axes[0, 0].set_xlabel('Iteration', fontsize=11)
        axes[0, 0].set_ylabel('Score', fontsize=11)
        axes[0, 0].set_title(f'Classification Performance: {attack_name}', fontsize=12)
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Detection F1
        axes[0, 1].plot(iterations, det_f1_scores, 'o-', color='red', linewidth=2)
        axes[0, 1].set_xlabel('Iteration', fontsize=11)
        axes[0, 1].set_ylabel('Detection F1 Score', fontsize=11)
        axes[0, 1].set_title(f'Anomaly Detection Performance: {attack_name}', fontsize=12)
        axes[0, 1].grid(alpha=0.3)
        
        # Reward
        axes[1, 0].plot(iterations, rewards, 'o-', color='green', linewidth=2)
        axes[1, 0].set_xlabel('Iteration', fontsize=11)
        axes[1, 0].set_ylabel('Reward', fontsize=11)
        axes[1, 0].set_title(f'DDPG Agent Reward: {attack_name}', fontsize=12)
        axes[1, 0].grid(alpha=0.3)
        
        # Threshold
        axes[1, 1].plot(iterations, thresholds, 'o-', color='purple', linewidth=2)
        axes[1, 1].set_xlabel('Iteration', fontsize=11)
        axes[1, 1].set_ylabel('Threshold Percentile', fontsize=11)
        axes[1, 1].set_title(f'Adaptive Threshold: {attack_name}', fontsize=12)
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"feedback_learning_{attack_name}.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Feedback learning curve saved to {output_dir}/feedback_learning_{attack_name}.png")


def compute_all_metrics(clean_dataset: TensorDataset, attacked_datasets: dict,
                        model, device: torch.device, output_dir: str = "outputs") -> dict:
    """Compute all evaluation metrics for all datasets."""
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # Evaluate on clean dataset
    print("\nEvaluating on clean dataset...")
    clean_results = evaluate_model(model, clean_dataset, device)
    results['clean'] = clean_results
    
    # Evaluate on attacked datasets
    for attack_name, attacked_dataset in attacked_datasets.items():
        print(f"Evaluating on {attack_name} dataset...")
        attack_results = evaluate_model(model, attacked_dataset, device)
        
        # Compute ASR (comparing to clean labels)
        x_clean, y_clean = clean_dataset.tensors
        x_attacked, y_attacked = attacked_dataset.tensors
        
        # For ASR, we need original labels (from clean dataset)
        # If sizes match, use clean labels as ground truth
        if len(y_clean) == len(y_attacked):
            asr = compute_attack_success_rate(y_attacked, attack_results['predictions'], y_clean)
        else:
            # Use attacked labels as proxy
            asr = 1.0 - attack_results['accuracy']
        
        attack_results['attack_success_rate'] = asr
        results[attack_name] = attack_results
    
    # Plot results
    plot_accuracy_comparison(results, output_dir)
    plot_confusion_matrices(results, output_dir)
    
    # Print summary
    print("\n=== Evaluation Summary ===")
    for dataset_name, res in results.items():
        print(f"\n{dataset_name.upper()}:")
        print(f"  Accuracy: {res['accuracy']:.4f}")
        print(f"  Precision: {res['precision']:.4f}")
        print(f"  Recall: {res['recall']:.4f}")
        print(f"  F1-Score: {res['f1_score']:.4f}")
        if 'attack_success_rate' in res:
            print(f"  Attack Success Rate: {res['attack_success_rate']:.4f}")
    
    return results


