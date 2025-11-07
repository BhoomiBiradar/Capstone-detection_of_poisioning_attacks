# 📚 COMPLETE PROJECT EXPLANATION

## 🎯 Project Overview

This project implements a **Feedback-Driven Real-Time Detection System** for detecting data poisoning attacks in machine learning pipelines. It uses multiple detection algorithms, explainability techniques (SHAP/LIME), and a reinforcement learning agent (DDPG) to adaptively improve detection over time.

---

## 📁 PROJECT STRUCTURE & FILE-BY-FILE EXPLANATION

### 🔵 **main.py** - The Orchestrator

**Purpose**: Main entry point that coordinates all components sequentially.

**What it does**:

1. **Setup Phase** (Lines 18-40)
   - Creates all necessary directories (`data/`, `models/`, `outputs/`, etc.)
   - Detects available device (CPU/GPU)
   - Prints welcome message

2. **STEP 1: Data Preparation** (Lines 42-53)
   - Calls `prepare_cifar10_and_attacks()` to:
     - Download CIFAR-10 dataset
     - Generate 4 datasets: `clean.pt`, `flipped.pt`, `corrupted.pt`, `fgsm.pt`
   - Initially uses untrained model for FGSM (will regenerate later)

3. **STEP 2: Model Training** (Lines 55-84)
   - Builds CNN and Autoencoder models
   - Trains CNN on clean data (10 epochs)
   - Trains Autoencoder on clean data (10 epochs)
   - Saves model weights to `models/` directory
   - **Regenerates FGSM attacks** with trained CNN for better quality
   - Quick baseline evaluation on all datasets

4. **STEP 3: Detection Algorithms** (Lines 92-164)
   - For each attacked dataset (`flipped`, `corrupted`, `fgsm`):
     - Runs **4 detectors**:
       - Centroid Detector
       - KNN Detector
       - Autoencoder Detector
       - Gradient Filter
     - Each detector outputs:
       - List of anomaly indices
       - Cleaned dataset (with anomalies removed)
     - Saves results to `data/` directory

5. **STEP 4: Evaluation & Explainability** (Lines 166-210)
   - Computes comprehensive metrics (accuracy, F1, confusion matrices)
   - Generates plots: accuracy comparison, confusion matrices
   - For each attack type:
     - Generates **SHAP explanations** (why samples flagged as anomalies)
     - Generates **LIME explanations** (local interpretability)
     - Saves visualizations to `outputs/shap_*/` and `outputs/lime_*/`

6. **STEP 5: Feedback Loop** (Lines 212-246)
   - Runs adaptive detection with DDPG agent for each attack type
   - Agent learns to adjust detection thresholds over 10 iterations
   - Saves learning curves showing improvement over time
   - Saves trained DDPG agents

7. **STEP 6: Final Summary** (Lines 248-269)
   - Prints summary of all generated outputs
   - Lists all saved files

---

### 🟢 **data/prepare_and_attacks.py** - Data Generation

**Purpose**: Prepares CIFAR-10 dataset and simulates three types of poisoning attacks.

**Key Functions**:

1. **`_to_tensor_dataset(x, y)`** (Line 9)
   - Converts tensors to PyTorch TensorDataset format

2. **`_save_tensor_dataset(ds, path)`** (Line 13)
   - Saves dataset as `.pt` file with `{"x": x, "y": y}` structure

3. **`_fgsm_attack(x, y, model, device, eps)`** (Line 18)
   - **Fast Gradient Sign Method (FGSM)** attack
   - Creates adversarial examples by:
     - Computing loss w.r.t. input
     - Taking gradient sign
     - Adding perturbation: `x_adv = x + eps * sign(gradient)`
   - Clamps values to [0, 1] range
   - Returns adversarial images

4. **`_feature_corrupt(x, sigma)`** (Line 32)
   - **Feature Corruption** attack
   - Adds Gaussian noise: `x_noisy = x + N(0, sigma²)`
   - Simulates sensor faults or data corruption
   - Clamps to valid range

5. **`_label_flip(y, src_class, dst_class, flip_ratio)`** (Line 52)
   - **Label Flipping** attack
   - Randomly flips `flip_ratio` (20%) of labels from `src_class` to `dst_class`
   - Example: 20% of class 3 → class 5

6. **`regenerate_fgsm_attacks()`** (Line 38)
   - Regenerates FGSM attacks with a trained model
   - Only processes subset (2000 samples) to limit compute
   - Concatenates with rest of dataset

7. **`prepare_cifar10_and_attacks()`** (Line 63) - **MAIN FUNCTION**
   - Downloads CIFAR-10 training set (50,000 images)
   - Converts to tensors
   - Generates 4 datasets:
     - **clean.pt**: Original clean data
     - **flipped.pt**: 20% labels flipped (class 3→5)
     - **corrupted.pt**: Gaussian noise added (σ=0.15)
     - **fgsm.pt**: FGSM adversarial examples (eps=4/255)
   - Saves all to `data/` directory
   - Returns dictionary of TensorDatasets

---

### 🟡 **models/cnn_model.py** - CNN Classifier

**Purpose**: Defines and trains a CNN for CIFAR-10 classification.

**Components**:

1. **`SimpleCIFAR10CNN` Class** (Line 7)
   - **Architecture**:
     - Conv2D(3→32) + ReLU + MaxPool → 16×16
     - Conv2D(32→64) + ReLU + MaxPool → 8×8
     - Conv2D(64→128) + ReLU + MaxPool → 4×4
     - Flatten → Linear(128×4×4 → 256) → ReLU
     - Linear(256 → 10) → Output (10 classes)
   - Input: 32×32×3 images
   - Output: 10 logits (one per class)

2. **`build_cnn()`** (Line 27)
   - Factory function to create CNN instance

3. **`train_cnn(model, dataset, device, epochs, batch_size, lr)`** (Line 31)
   - **Training Loop**:
     - Creates DataLoader with shuffling
     - Uses Adam optimizer (lr=1e-3)
     - For each epoch:
       - Forward pass → logits
       - Compute CrossEntropy loss
       - Backward pass → update weights
   - Trains model in-place

4. **`eval_cnn(model, dataset, device, batch_size)`** (Line 47)
   - **Evaluation**:
     - Sets model to eval mode
     - Computes predictions (argmax of logits)
     - Calculates accuracy: `correct / total`
   - Returns accuracy (float 0-1)

---

### 🟡 **models/autoencoder_model.py** - Autoencoder

**Purpose**: Defines a convolutional autoencoder for reconstruction-based anomaly detection.

**Components**:

1. **`_AE` Class** (Line 40) - **Inner class for autoencoder**
   - **Encoder** (compression):
     - Conv2D(3→32) + ReLU
     - MaxPool → Conv2D(32→64) + ReLU
     - MaxPool → Conv2D(64→128) + ReLU
     - Output: 8×8×128 feature map
   
   - **Decoder** (reconstruction):
     - ConvTranspose2d(128→64) → 16×16
     - ConvTranspose2d(64→32) → 32×32
     - Conv2D(32→3) + Sigmoid → Original size
   
   - Input: 32×32×3
   - Output: 32×32×3 (reconstructed image)

2. **`build_autoencoder()`** (Line 38)
   - Returns autoencoder instance

3. **`train_autoencoder(model, dataset, device, epochs, batch_size, lr)`** (Line 68)
   - **Training**:
     - Uses MSE loss: `loss = ||reconstructed - original||²`
     - Trains to minimize reconstruction error
     - Normal samples should have low error
     - Anomalies should have high error

---

### 🔴 **detectors/centroid_detector.py** - Statistical Detector

**Purpose**: Detects anomalies by measuring distance from class centroids.

**How it works**:

1. **`detect_anomalies_centroid(dataset, percentile=95.0)`** (Line 7)
   - **Algorithm**:
     - Flattens images to vectors
     - For each class (0-9):
       - Computes centroid: `mean of all samples in class`
     - For each sample:
       - Computes Euclidean distance to its class centroid
     - Sets threshold = 95th percentile of distances
     - Flags samples with distance > threshold as anomalies
   
   - **Why it works**:
     - Normal samples cluster around class centroid
     - Corrupted/adversarial samples are far from centroid
   
   - Returns: `(anomaly_indices, cleaned_dataset)`

2. **`save_detector_results()`** (Line 56)
   - Saves anomaly indices and cleaned dataset to files

---

### 🔴 **detectors/knn_detector.py** - Density-Based Detector

**Purpose**: Detects anomalies using k-nearest neighbors density.

**How it works**:

1. **`detect_anomalies_knn(dataset, k=5, percentile=95.0)`** (Line 8)
   - **Algorithm**:
     - Fits KNN model (k=5) on all samples
     - For each sample:
       - Finds k nearest neighbors
       - Computes mean distance to neighbors
     - Sets threshold = 95th percentile of mean distances
     - Flags samples with high mean distance as anomalies
   
   - **Why it works**:
     - Normal samples have nearby neighbors (low distance)
     - Anomalies are isolated (high distance to neighbors)
     - Good for label flipping (mislabeled samples are isolated)
   
   - Returns: `(anomaly_indices, cleaned_dataset)`

---

### 🔴 **detectors/autoencoder_detector.py** - Deep Learning Detector

**Purpose**: Detects anomalies using reconstruction error from autoencoder.

**How it works**:

1. **`detect_anomalies_autoencoder(dataset, autoencoder_model, device, percentile=95.0)`** (Line 8)
   - **Algorithm**:
     - Loads trained autoencoder
     - For each sample:
       - Reconstructs image: `recon = autoencoder(x)`
       - Computes MSE: `error = ||recon - x||²`
     - Sets threshold = 95th percentile of errors
     - Flags samples with high reconstruction error as anomalies
   
   - **Why it works**:
     - Autoencoder trained on clean data
     - Reconstructs normal samples well (low error)
     - Struggles with anomalies (high error)
     - Effective for adversarial and corruption attacks
   
   - Returns: `(anomaly_indices, cleaned_dataset)`

---

### 🔴 **detectors/gradient_filter.py** - Adversarial Detector

**Purpose**: Detects adversarial examples using input gradient magnitude.

**How it works**:

1. **`detect_anomalies_gradient(dataset, model, device, percentile=95.0)`** (Line 8)
   - **Algorithm**:
     - For each sample:
       - Sets `x.requires_grad = True`
       - Forward pass: `logits = model(x)`
       - Computes loss: `loss = CrossEntropy(logits, y)`
       - Backward pass: `grad = ∂loss/∂x`
       - Computes gradient magnitude: `||grad||`
     - Sets threshold = 95th percentile of magnitudes
     - Flags samples with high gradient magnitude as anomalies
   
   - **Why it works**:
     - Adversarial examples (FGSM) have high input gradients
     - Normal samples have lower gradients
     - Specifically effective against FGSM attacks
   
   - Returns: `(anomaly_indices, cleaned_dataset)`

---

### 🟣 **feedback/ddpg_agent.py** - Reinforcement Learning Agent

**Purpose**: Implements DDPG (Deep Deterministic Policy Gradient) agent for adaptive threshold adjustment.

**Components**:

1. **`Actor` Network** (Line 9)
   - **Purpose**: Policy network that outputs threshold adjustment
   - **Architecture**:
     - Input: State vector [accuracy, f1_score, detection_f1, threshold]
     - FC(4→128) → ReLU
     - FC(128→128) → ReLU
     - FC(128→1) → Tanh → Action [-1, 1]
   - Output: Action (threshold adjustment)

2. **`Critic` Network** (Line 26)
   - **Purpose**: Q-value estimator (how good is state-action pair)
   - **Architecture**:
     - Input: State + Action (concatenated)
     - FC(5→128) → ReLU
     - FC(128→128) → ReLU
     - FC(128→1) → Q-value
   - Output: Q-value (expected reward)

3. **`DDPGAgent` Class** (Line 42)
   - **Components**:
     - Actor + Actor_target (target network for stability)
     - Critic + Critic_target
     - Replay buffer (stores past experiences)
     - Optimizers (Adam)
   
   - **Key Methods**:
     - **`select_action(state, add_noise)`** (Line 73):
       - Actor predicts action
       - Adds noise for exploration
       - Returns action [-1, 1]
     
     - **`store_transition(state, action, reward, next_state, done)`** (Line 88):
       - Stores experience in replay buffer
     
     - **`train()`** (Line 92):
       - Samples batch from replay buffer
       - Updates Critic: minimize MSE between Q and target Q
       - Updates Actor: maximize Q-value
       - Soft updates target networks (τ=0.001)

---

### 🟣 **feedback/feedback_loop.py** - Adaptive Detection Loop

**Purpose**: Implements feedback-driven adaptive detection using DDPG agent.

**Key Functions**:

1. **`compute_metrics(y_true, y_pred, anomaly_mask_true, anomaly_mask_pred)`** (Line 15)
   - Computes:
     - Classification metrics: accuracy, precision, recall, F1
     - Detection metrics: TP, FP, FN, TN, detection F1
   - Returns dictionary of metrics

2. **`get_state_vector(metrics, current_threshold)`** (Line 63)
   - Converts metrics to 4D state vector:
     - `[accuracy, f1_score, detection_f1, threshold]`
   - Used as input to DDPG agent

3. **`compute_reward(metrics, previous_metrics)`** (Line 73)
   - **Reward Function**:
     - Base: `0.6 * detection_f1 + 0.4 * accuracy`
     - Bonus: `+0.2 * improvement_in_detection_f1`
     - Penalty: `-0.1 * false_positive_ratio`
   - Encourages high detection F1 and accuracy
   - Penalizes false positives

4. **`adaptive_detection_with_feedback()`** (Line 91) - **MAIN FUNCTION**
   - **Algorithm**:
     - Initializes DDPG agent
     - Creates ground truth anomalies (compare clean vs attacked)
     - For 10 iterations:
       1. Run all 4 detectors with current threshold
       2. Combine results (union of all anomalies)
       3. Compute metrics (accuracy, F1, detection F1)
       4. Compute reward
       5. Store transition in replay buffer
       6. Train DDPG agent
       7. Agent selects new threshold adjustment
       8. Update threshold: `new = old + action * 5.0`
       9. Clip threshold to [80, 99] percentile
     - Saves agent and returns results
   
   - **Adaptive Learning**:
     - Agent learns which threshold works best
     - Adjusts threshold based on detection performance
     - Improves over iterations

---

### 🟠 **explainability/shap_explain.py** - SHAP Explanations

**Purpose**: Generates SHAP (SHapley Additive exPlanations) visualizations.

**How it works**:

1. **`explain_with_shap(model, dataset, anomaly_indices, device, num_samples, output_dir)`** (Line 9)
   - **Algorithm**:
     - Selects subset of anomaly samples
     - Creates model wrapper function
     - Uses `GradientExplainer` (for images)
     - Computes SHAP values (feature importance)
     - Visualizes with `shap.image_plot()`
     - Saves to PNG file
   
   - **Fallback** (if SHAP fails):
     - Uses gradient-based visualization
     - Shows input gradients as heatmap
   
   - **Output**: `shap_explanations.png` showing which pixels contributed to anomaly detection

---

### 🟠 **explainability/lime_explain.py** - LIME Explanations

**Purpose**: Generates LIME (Local Interpretable Model-agnostic Explanations) visualizations.

**How it works**:

1. **`explain_with_lime(model, dataset, anomaly_indices, device, num_samples, output_dir)`** (Line 10)
   - **Algorithm**:
     - Selects subset of anomaly samples
     - Creates prediction function wrapper
     - Uses `LimeImageExplainer`
     - For each sample:
       - Generates perturbed versions
       - Trains local linear model
       - Identifies important image segments
     - Visualizes with `mark_boundaries()`
     - Saves to PNG file
   
   - **Output**: `lime_explanations.png` showing which image regions contributed to detection

---

### 🟦 **evaluation_metrics.py** - Metrics & Plotting

**Purpose**: Computes evaluation metrics and generates plots.

**Key Functions**:

1. **`compute_ndcg(y_true, y_scores, k=10)`** (Line 10)
   - Normalized Discounted Cumulative Gain
   - Measures ranking quality
   - Higher = better ranking of anomalies

2. **`compute_reciprocal_rank(y_true, y_scores)`** (Line 30)
   - Mean Reciprocal Rank
   - Position of first relevant item
   - Higher = anomalies found earlier

3. **`compute_attack_success_rate(y_true, y_pred, y_original)`** (Line 43)
   - Attack Success Rate (ASR)
   - `ASR = (misclassified samples) / (total samples)`
   - Measures attack effectiveness

4. **`evaluate_model(model, dataset, device, batch_size)`** (Line 64)
   - Evaluates model on dataset
   - Computes: accuracy, precision, recall, F1, confusion matrix
   - Returns dictionary of metrics

5. **`plot_accuracy_comparison(results_dict, output_dir)`** (Line 111)
   - Creates bar chart comparing accuracy across datasets
   - Saves: `accuracy_comparison.png`

6. **`plot_confusion_matrices(results_dict, output_dir)`** (Line 137)
   - Creates 2×2 grid of confusion matrices
   - One for each dataset (clean, flipped, corrupted, fgsm)
   - Saves: `confusion_matrices.png`

7. **`plot_feedback_learning_curve(feedback_results, output_dir)`** (Line 160)
   - Creates 2×2 subplot showing:
     - Classification performance (accuracy, F1)
     - Detection F1 over iterations
     - DDPG reward over iterations
     - Adaptive threshold over iterations
   - Saves: `feedback_learning_{attack_name}.png`

8. **`compute_all_metrics(clean_dataset, attacked_datasets, model, device, output_dir)`** (Line 214)
   - **Main evaluation function**
   - Evaluates on all datasets
   - Computes ASR
   - Generates all plots
   - Prints summary
   - Returns dictionary of results

---

## 🔄 DATA FLOW DIAGRAM

```
1. main.py
   ↓
2. prepare_cifar10_and_attacks()
   → Downloads CIFAR-10
   → Generates: clean.pt, flipped.pt, corrupted.pt, fgsm.pt
   ↓
3. Train CNN & Autoencoder
   → cnn_weights.pt, autoencoder_weights.pt
   ↓
4. Run 4 Detectors (for each attack)
   → Centroid, KNN, Autoencoder, Gradient
   → Output: anomaly_indices, cleaned_datasets
   ↓
5. Evaluation & Explainability
   → compute_all_metrics() → plots
   → SHAP/LIME → visualizations
   ↓
6. Feedback Loop (DDPG)
   → adaptive_detection_with_feedback()
   → Agent learns optimal threshold
   → Learning curves
   ↓
7. Final Outputs
   → All files saved to outputs/
```

---

## 🎯 KEY CONCEPTS

### **95th Percentile Threshold**
- All detectors use 95th percentile as cutoff
- Meaning: Top 5% of samples (by distance/error/magnitude) are flagged as anomalies
- Adaptive: DDPG agent adjusts this threshold (80-99%)

### **Anomaly Detection**
- **Goal**: Identify poisoned/attacked samples
- **Method**: Multiple detectors vote (union of results)
- **Output**: List of indices flagged as anomalies

### **Feedback Loop**
- **State**: [accuracy, f1_score, detection_f1, threshold]
- **Action**: Threshold adjustment (-1 to +1, scaled to ±5 percentile)
- **Reward**: Based on detection F1 and accuracy
- **Learning**: Agent improves threshold selection over iterations

### **Explainability**
- **SHAP**: Global feature importance (which pixels matter)
- **LIME**: Local explanations (which regions matter for specific samples)
- **Purpose**: Understand WHY samples were flagged

---

## 📊 OUTPUT FILES

### **Data Files** (`data/`)
- `clean.pt`, `flipped.pt`, `corrupted.pt`, `fgsm.pt`
- `*_anomalies.pt` (anomaly indices)
- `*_cleaned.pt` (datasets with anomalies removed)

### **Model Files** (`models/`)
- `cnn_weights.pt`
- `autoencoder_weights.pt`

### **Output Files** (`outputs/`)
- `accuracy_comparison.png`
- `confusion_matrices.png`
- `shap_explanations.png` (in `shap_*/` folders)
- `lime_explanations.png` (in `lime_*/` folders)
- `feedback_learning_{attack}.png`
- `ddpg_agent_{attack}.pt`

---

## 🚀 EXECUTION SUMMARY

When you run `python main.py`:

1. **Downloads** CIFAR-10 (if not present)
2. **Generates** 4 datasets with attacks
3. **Trains** CNN (10 epochs) and Autoencoder (10 epochs)
4. **Runs** 4 detectors on 3 attack types (12 detector runs total)
5. **Evaluates** model performance on all datasets
6. **Generates** SHAP and LIME explanations
7. **Runs** DDPG feedback loop (10 iterations × 3 attacks = 30 iterations)
8. **Saves** all outputs to respective directories

**Total Time**: ~30-60 minutes (depending on CPU/GPU)

---

This completes the detailed explanation of every file and component in the project! 🎉


