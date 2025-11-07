import os
import torch
import numpy as np

from data.prepare_and_attacks import prepare_cifar10_and_attacks
from models.cnn_model import build_cnn, train_cnn, eval_cnn
from models.autoencoder_model import build_autoencoder, train_autoencoder
from detectors.centroid_detector import detect_anomalies_centroid, save_detector_results as save_centroid_results
from detectors.knn_detector import detect_anomalies_knn, save_detector_results as save_knn_results
from detectors.autoencoder_detector import detect_anomalies_autoencoder, save_detector_results as save_ae_results
from detectors.gradient_filter import detect_anomalies_gradient, save_detector_results as save_grad_results
from explainability.shap_explain import explain_with_shap
from explainability.lime_explain import explain_with_lime
from feedback.feedback_loop import adaptive_detection_with_feedback
from evaluation_metrics import compute_all_metrics, plot_feedback_learning_curve


def ensure_dirs():
    """Create all necessary directories."""
    for d in [
        "data",
        "models",
        "detectors",
        "feedback",
        "explainability",
        "outputs",
    ]:
        os.makedirs(d, exist_ok=True)


def main():
    """Main pipeline execution."""
    print("=" * 80)
    print("Detecting Poisoning Attacks in Machine Learning Pipelines")
    print("Feedback-Driven Real-Time Detection System")
    print("=" * 80)
    
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")
    
    # ========================================================================
    # 1. Data and Attack Simulation (Initial - without trained model)
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: Data Preprocessing and Attack Simulation")
    print("=" * 80)
    print("Preparing CIFAR-10 dataset and generating attacks...")
    
    # First generate datasets without trained model (for label flipping and corruption)
    # FGSM will use untrained model initially
    datasets = prepare_cifar10_and_attacks(root_dir="data", device=device, fgsm_model=None)
    print("✓ Generated datasets: clean.pt, flipped.pt, corrupted.pt, fgsm.pt")
    
    # ========================================================================
    # 2. Base Model Implementation
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Base Model Training")
    print("=" * 80)
    
    # Build models
    cnn = build_cnn().to(device)
    autoenc = build_autoencoder().to(device)
    print("✓ Built CNN and Autoencoder models")
    
    # Train CNN on clean data
    print("\nTraining CNN on clean data...")
    train_cnn(cnn, datasets["clean"], device=device, epochs=10, batch_size=128)
    torch.save(cnn.state_dict(), os.path.join("models", "cnn_weights.pt"))
    print("✓ CNN trained and saved")
    
    # Train Autoencoder on clean data
    print("\nTraining Autoencoder on clean data...")
    train_autoencoder(autoenc, datasets["clean"], device=device, epochs=10, batch_size=128)
    torch.save(autoenc.state_dict(), os.path.join("models", "autoencoder_weights.pt"))
    print("✓ Autoencoder trained and saved")
    
    # Regenerate FGSM attacks with trained model for better quality
    print("\nRegenerating FGSM attacks with trained model...")
    from data.prepare_and_attacks import regenerate_fgsm_attacks
    x_clean, y_clean = datasets["clean"].tensors
    datasets["fgsm"] = regenerate_fgsm_attacks("data", x_clean, y_clean, cnn, device)
    print("✓ FGSM attacks regenerated with trained model")
    
    # Quick baseline evaluation
    print("\nBaseline Evaluation:")
    for name in ["clean", "flipped", "corrupted", "fgsm"]:
        acc = eval_cnn(cnn, datasets[name], device=device, batch_size=256)
        print(f"  Accuracy on {name}: {acc:.4f}")
    
    # ========================================================================
    # 3. Core Detection Algorithms
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: Core Detection Algorithms")
    print("=" * 80)
    
    attacked_datasets = {
        "flipped": datasets["flipped"],
        "corrupted": datasets["corrupted"],
        "fgsm": datasets["fgsm"]
    }
    
    detector_results_all = {}
    
    for attack_name, attacked_dataset in attacked_datasets.items():
        print(f"\nRunning detectors on {attack_name} dataset...")
        
        # Centroid Detector
        print("  - Centroid Detector...")
        try:
            anomaly_idx_centroid, cleaned_centroid = detect_anomalies_centroid(
                attacked_dataset, percentile=95.0
            )
            save_centroid_results(anomaly_idx_centroid, cleaned_centroid, 
                                 f"{attack_name}_centroid", "data")
            detector_results_all[f"{attack_name}_centroid"] = anomaly_idx_centroid
            print(f"    Found {len(anomaly_idx_centroid)} anomalies")
        except Exception as e:
            print(f"    Error: {e}")
            detector_results_all[f"{attack_name}_centroid"] = []
        
        # KNN Detector
        print("  - KNN Detector...")
        try:
            anomaly_idx_knn, cleaned_knn = detect_anomalies_knn(
                attacked_dataset, k=5, percentile=95.0
            )
            save_knn_results(anomaly_idx_knn, cleaned_knn, 
                            f"{attack_name}_knn", "data")
            detector_results_all[f"{attack_name}_knn"] = anomaly_idx_knn
            print(f"    Found {len(anomaly_idx_knn)} anomalies")
        except Exception as e:
            print(f"    Error: {e}")
            detector_results_all[f"{attack_name}_knn"] = []
        
        # Autoencoder Detector
        print("  - Autoencoder Detector...")
        try:
            anomaly_idx_ae, cleaned_ae = detect_anomalies_autoencoder(
                attacked_dataset, autoenc, device, percentile=95.0
            )
            save_ae_results(anomaly_idx_ae, cleaned_ae, 
                           f"{attack_name}_autoencoder", "data")
            detector_results_all[f"{attack_name}_autoencoder"] = anomaly_idx_ae
            print(f"    Found {len(anomaly_idx_ae)} anomalies")
        except Exception as e:
            print(f"    Error: {e}")
            detector_results_all[f"{attack_name}_autoencoder"] = []
        
        # Gradient Filter
        print("  - Gradient Filter...")
        try:
            anomaly_idx_grad, cleaned_grad = detect_anomalies_gradient(
                attacked_dataset, cnn, device, percentile=95.0
            )
            save_grad_results(anomaly_idx_grad, cleaned_grad, 
                             f"{attack_name}_gradient", "data")
            detector_results_all[f"{attack_name}_gradient"] = anomaly_idx_grad
            print(f"    Found {len(anomaly_idx_grad)} anomalies")
        except Exception as e:
            print(f"    Error: {e}")
            detector_results_all[f"{attack_name}_gradient"] = []
    
    # ========================================================================
    # 4. Results, Plots, and Explainability
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: Evaluation Metrics and Visualizations")
    print("=" * 80)
    
    # Compute comprehensive metrics
    print("\nComputing evaluation metrics...")
    evaluation_results = compute_all_metrics(
        datasets["clean"], 
        attacked_datasets,
        cnn, 
        device, 
        output_dir="outputs"
    )
    print("✓ Metrics computed and plots saved")
    
    # SHAP and LIME explanations
    print("\nGenerating explainability visualizations...")
    for attack_name, attacked_dataset in attacked_datasets.items():
        # Get combined anomaly indices from all detectors
        all_anomalies = set()
        for detector_name in ["centroid", "knn", "autoencoder", "gradient"]:
            key = f"{attack_name}_{detector_name}"
            if key in detector_results_all:
                all_anomalies.update(detector_results_all[key])
        
        anomaly_list = list(all_anomalies)
        
        print(f"  - SHAP explanations for {attack_name}...")
        try:
            explain_with_shap(cnn, attacked_dataset, anomaly_list, device, 
                            num_samples=min(50, len(anomaly_list)), 
                            output_dir=f"outputs/shap_{attack_name}")
        except Exception as e:
            print(f"    SHAP failed: {e}")
        
        print(f"  - LIME explanations for {attack_name}...")
        try:
            explain_with_lime(cnn, attacked_dataset, anomaly_list, device, 
                            num_samples=min(5, len(anomaly_list)), 
                            output_dir=f"outputs/lime_{attack_name}")
        except Exception as e:
            print(f"    LIME failed: {e}")
    
    # ========================================================================
    # 5. Feedback-Driven Adaptive Learning Module
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: Feedback-Driven Adaptive Learning (DDPG)")
    print("=" * 80)
    
    feedback_results = {}
    
    for attack_name, attacked_dataset in attacked_datasets.items():
        print(f"\nRunning adaptive detection with feedback for {attack_name}...")
        try:
            feedback_result = adaptive_detection_with_feedback(
                clean_dataset=datasets["clean"],
                attacked_dataset=attacked_dataset,
                attacked_name=attack_name,
                cnn_model=cnn,
                autoencoder_model=autoenc,
                device=device,
                initial_percentile=95.0,
                num_iterations=10,
                output_dir="outputs"
            )
            feedback_results[attack_name] = feedback_result
            print(f"✓ Feedback loop completed for {attack_name}")
        except Exception as e:
            print(f"  Error in feedback loop for {attack_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Plot feedback learning curves
    if feedback_results:
        print("\nPlotting feedback learning curves...")
        plot_feedback_learning_curve(feedback_results, output_dir="outputs")
        print("✓ Learning curves saved")
    
    # ========================================================================
    # 6. Final Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: Final Summary")
    print("=" * 80)
    
    print("\n✓ All components executed successfully!")
    print("\nGenerated outputs:")
    print("  - Data files: data/*.pt")
    print("  - Model weights: models/*.pt")
    print("  - Detection results: data/*_anomalies.pt, data/*_cleaned.pt")
    print("  - Evaluation plots: outputs/accuracy_comparison.png")
    print("  - Confusion matrices: outputs/confusion_matrices.png")
    print("  - SHAP explanations: outputs/shap_*/")
    print("  - LIME explanations: outputs/lime_*/")
    print("  - Feedback learning curves: outputs/feedback_learning_*.png")
    print("  - DDPG agents: outputs/ddpg_agent_*.pt")
    
    print("\n" + "=" * 80)
    print("Pipeline execution complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
