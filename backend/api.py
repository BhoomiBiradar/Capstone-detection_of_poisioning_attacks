"""
FastAPI Backend for Poisoning Attack Detection System
"""
import os
import sys
import base64
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
# Suppress torch.load FutureWarnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, message='.*torch.load.*weights_only.*')

# Ensure project root in sys.path when run directly
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.utils.data_preparation import prepare_cifar10_and_attacks
from backend.models.cnn_model import build_cnn, train_cnn, eval_cnn
from backend.models.autoencoder_model import build_autoencoder, train_autoencoder
from backend.detectors.centroid_detector import detect_anomalies_centroid
from backend.detectors.knn_detector import detect_anomalies_knn
from backend.detectors.autoencoder_detector import detect_anomalies_autoencoder
from backend.detectors.gradient_filter import detect_anomalies_gradient
from backend.detectors.fusion_detector import detect_anomalies_fusion
from backend.feedback.feedback_loop import adaptive_detection_with_feedback
from backend.evaluation_metrics import compute_all_metrics
from backend.explainability.shap_explain import explain_with_shap
from backend.explainability.lime_explain import explain_with_lime

app = FastAPI(title="Poisoning Attack Detection API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
datasets_cache: Dict[str, TensorDataset] = {}
dataset_metadata: Dict[str, Dict[str, np.ndarray]] = {}
models_cache: Dict[str, torch.nn.Module] = {}
detection_results: Dict[str, Dict] = {}
threshold_cache: Dict[str, float] = {}


# Request/Response Models
class DatasetRequest(BaseModel):
    dataset_type: str  # "clean", "flipped", "corrupted", "fgsm", "mixed"
    mix_flipped: Optional[int] = 0
    mix_corrupted: Optional[int] = 0
    mix_fgsm: Optional[int] = 0


class DetectionRequest(BaseModel):
    dataset_type: str
    threshold: float = 95.0


class FeedbackRequest(BaseModel):
    dataset_type: str
    num_iterations: int = 10


# Helper Functions
def tensor_to_base64(tensor: torch.Tensor) -> str:
    """Convert tensor image to base64 string."""
    img = tensor.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    pil_img = Image.fromarray(img)
    buff = io.BytesIO()
    pil_img.save(buff, format="PNG")
    img_str = base64.b64encode(buff.getvalue()).decode()
    return img_str


def get_sample_images(dataset, num_samples: int = 10) -> List[Dict]:
    """Get sample images from dataset."""
    x, y = dataset.tensors
    indices = np.random.choice(len(x), min(num_samples, len(x)), replace=False)
    samples = []
    for idx in indices:
        samples.append({
            "image": tensor_to_base64(x[idx]),
            "label": int(y[idx].item()),
            "index": int(idx)
        })
    return samples


def sanitize_for_json(obj):
    """Recursively convert numpy/tensor types to native Python for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return sanitize_for_json(obj.tolist())
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [sanitize_for_json(item) for item in obj]
    if isinstance(obj, dict):
        return {key: sanitize_for_json(value) for key, value in obj.items()}
    return obj


def build_mixed_dataset(
    mix_flipped: int,
    mix_corrupted: int,
    mix_fgsm: int,
) -> Tuple[TensorDataset, np.ndarray]:
    """Create mixed dataset according to requested percentages."""    
    total_mix = mix_flipped + mix_corrupted + mix_fgsm
    if total_mix > 100:
        raise ValueError("Mix percentages cannot exceed 100%")

    if "clean" not in datasets_cache:
        raise ValueError("'clean' dataset not found in cache. Please initialize datasets first.")
    
    base = datasets_cache["clean"]
    x_base, y_base = base.tensors
    total_samples = len(x_base)

    def sample_from(name: str, count: int):
        if count <= 0:
            return None
        if name not in datasets_cache:
            raise ValueError(f"Dataset '{name}' not found in cache")
        
        ds = datasets_cache[name]
        
        mask = dataset_metadata.get(name, {}).get("attack_mask")
        
        x, y = ds.tensors
        
        available = len(x)
        actual_count = min(count, available)
        
        idx = torch.randperm(available)[:actual_count]
        x_sel = x[idx]
        y_sel = y[idx]
        
        if mask is not None:
            mask_sel = mask[idx.cpu().numpy()]
        else:
            mask_sel = np.zeros(actual_count, dtype=int)
        
        return x_sel, y_sel, mask_sel

    mix_counts = {
        "flipped": int(total_samples * mix_flipped / 100),
        "corrupted": int(total_samples * mix_corrupted / 100),
        "fgsm": int(total_samples * mix_fgsm / 100),
    }
    clean_count = total_samples - sum(mix_counts.values())

    x_parts = []
    y_parts = []
    mask_parts = []

    if clean_count > 0:
        clean_sample = sample_from("clean", clean_count)
        if clean_sample:
            x_clean, y_clean, mask_clean = clean_sample
            x_parts.append(x_clean)
            y_parts.append(y_clean)
            mask_parts.append(mask_clean)
        else:
            print(f" ")

    for name, count in mix_counts.items():
        if count <= 0:
            continue
        try:
            sample = sample_from(name, count)
            if not sample:
                continue
            x_part, y_part, mask_part = sample
            x_parts.append(x_part)
            y_parts.append(y_part)
            mask_parts.append(mask_part)
        except Exception as e:
            import traceback
            raise

    if not x_parts:
        raise ValueError("No samples collected. All counts may be zero or datasets unavailable.")

    
    try:
        x_mix = torch.cat(x_parts, dim=0)
        y_mix = torch.cat(y_parts, dim=0)
        mask_mix = np.concatenate(mask_parts, axis=0)
    except Exception as e:
        import traceback
        raise

    shuffled_idx = torch.randperm(len(x_mix))
    x_mix = x_mix[shuffled_idx]
    y_mix = y_mix[shuffled_idx]
    mask_mix = mask_mix[shuffled_idx.cpu().numpy()]

    result = TensorDataset(x_mix, y_mix), mask_mix
    return result


def run_all_detectors(dataset_subset, threshold, models_cache, device):
    """Run all detectors on the provided dataset subset."""
    public_results: Dict[str, Dict] = {}
    internal_indices: Dict[str, List[int]] = {}

    def _record_result(name: str, indices: List[int]):
        public_results[name] = {
            "anomalies_count": len(indices),
            "anomaly_indices": [int(idx) for idx in indices[:100]],
        }
        internal_indices[name] = [int(idx) for idx in indices]

    try:
        anomaly_idx, _ = detect_anomalies_centroid(dataset_subset, percentile=threshold)
        _record_result("centroid", anomaly_idx)
        print(f"[API] Centroid detector found {len(anomaly_idx)} anomalies.")
    except Exception as e:
        public_results["centroid"] = {"error": str(e)}
        internal_indices["centroid"] = []
        print(f"[API] Centroid detector error: {e}")

    try:
        if len(dataset_subset) == 0:
            raise ValueError("Dataset is empty")
        anomaly_idx, _ = detect_anomalies_knn(dataset_subset, k=5, percentile=threshold)
        _record_result("knn", anomaly_idx)
        print(f"[API] KNN detector found {len(anomaly_idx)} anomalies.")
    except Exception as e:
        error_msg = str(e) if e is not None else "Unknown error"
        public_results["knn"] = {"error": error_msg}
        internal_indices["knn"] = []
        print(f"[API] KNN detector error: {error_msg}")

    try:
        anomaly_idx, _ = detect_anomalies_autoencoder(
            dataset_subset, models_cache["autoencoder"], device, percentile=threshold
        )
        _record_result("autoencoder", anomaly_idx)
        print(f"[API] Autoencoder detector found {len(anomaly_idx)} anomalies.")
    except Exception as e:
        public_results["autoencoder"] = {"error": str(e)}
        internal_indices["autoencoder"] = []
        print(f"[API] Autoencoder detector error: {e}")

    try:
        anomaly_idx, _ = detect_anomalies_gradient(
            dataset_subset, models_cache["cnn"], device, percentile=threshold
        )
        _record_result("gradient", anomaly_idx)
        print(f"[API] Gradient detector found {len(anomaly_idx)} anomalies.")
    except Exception as e:
        public_results["gradient"] = {"error": str(e)}
        internal_indices["gradient"] = []
        print(f"[API] Gradient detector error: {e}")

    # NEW: Run fusion detector
    try:
        anomaly_idx, _, _ = detect_anomalies_fusion(
            dataset_subset, models_cache["cnn"], models_cache["autoencoder"], device, percentile=threshold
        )
        _record_result("fusion", anomaly_idx)
        print(f"[API] Fusion detector found {len(anomaly_idx)} anomalies.")
    except Exception as e:
        public_results["fusion"] = {"error": str(e)}
        internal_indices["fusion"] = []
        print(f"[API] Fusion detector error: {e}")

    combined_full = []
    for indices in internal_indices.values():
        combined_full.extend(indices)

    return public_results, internal_indices, combined_full


def adaptive_threshold_refinement(
    dataset_subset,
    initial_threshold,
    models_cache,
    device,
    base_results,
    base_internal_indices,
    max_cycles=3,
):
    """Lightweight adaptive threshold refinement during detection."""
    # If we're using the full dataset, skip adaptive refinement to avoid multiple runs
    # The full dataset is large enough that we don't need multiple refinement cycles
    if len(dataset_subset) > 10000:  # If dataset is large (full dataset)
        print(f"[API] Skipping adaptive threshold refinement for large dataset ({len(dataset_subset)} samples)")
        return float(initial_threshold), base_results, base_internal_indices, [], []
    
    current_threshold = float(initial_threshold)
    results = base_results
    internal_indices = base_internal_indices
    adaptive_log = []

    for cycle in range(1, max_cycles + 1):
        combined = []
        for indices in internal_indices.values():
            combined.extend(indices)
        ratio = len(combined) / max(1, len(dataset_subset))
        log_entry = {
            "cycle": cycle,
            "threshold": round(current_threshold, 3),
            "anomaly_ratio": round(ratio, 4),
        }

        adjustment = 0.0
        if ratio > 0.08:
            adjustment = +0.2
            log_entry["action"] = "increase"
        elif ratio < 0.02:
            adjustment = -0.2
            log_entry["action"] = "decrease"
        else:
            log_entry["action"] = "stable"
            adaptive_log.append(log_entry)
            break

        new_threshold = float(np.clip(current_threshold + adjustment, 80.0, 99.5))
        log_entry["adjustment"] = round(adjustment, 3)
        log_entry["new_threshold"] = round(new_threshold, 3)
        adaptive_log.append(log_entry)

        current_threshold = new_threshold
        results, internal_indices, _ = run_all_detectors(
            dataset_subset, current_threshold, models_cache, device
        )

    combined_final = []
    for indices in internal_indices.values():
        combined_final.extend(indices)
    return current_threshold, results, internal_indices, combined_final, adaptive_log


def generate_visualizations(
    dataset_type: str,
    dataset_subset: TensorDataset,
    anomaly_indices_map: Dict[str, List[int]],
    cnn_model,
    autoencoder_model,
    device: torch.device,
):
    """Generate SHAP, LIME, and evaluation metrics visualizations."""
    try:
        outputs_dir = Path("backend/static/outputs")
        outputs_dir.mkdir(parents=True, exist_ok=True)

        # Combine anomaly indices from all detectors
        combined_anomalies = set()
        for indices in anomaly_indices_map.values():
            combined_anomalies.update(int(idx) for idx in indices)
        combined_anomalies = list(combined_anomalies)

        # SHAP / LIME explanations
        if combined_anomalies:
            print(f"[API] Generating SHAP explanations...")
            try:
                shap_dir = outputs_dir / f"shap_{dataset_type}"
                shap_dir.mkdir(parents=True, exist_ok=True)
                explain_with_shap(
                    cnn_model,
                    dataset_subset,
                    combined_anomalies,
                    device,
                    num_samples=min(50, len(combined_anomalies)),
                    output_dir=str(shap_dir),
                )
                print(f"[API] SHAP explanations saved to {shap_dir}")
            except Exception as shap_error:
                print(f"[API] SHAP generation failed: {shap_error}")
                import traceback
                traceback.print_exc()

            print(f"[API] Generating LIME explanations...")
            try:
                lime_dir = outputs_dir / f"lime_{dataset_type}"
                lime_dir.mkdir(parents=True, exist_ok=True)
                explain_with_lime(
                    cnn_model,
                    dataset_subset,
                    combined_anomalies,
                    device,
                    num_samples=min(5, len(combined_anomalies)),
                    output_dir=str(lime_dir),
                )
                print(f"[API] LIME explanations saved to {lime_dir}")
            except Exception as lime_error:
                print(f"[API] LIME generation failed: {lime_error}")
                import traceback
                traceback.print_exc()
        else:
            print("[API] No anomalies detected; skipping SHAP/LIME.")

        # Accuracy / confusion matrix plots
        print(f"[API] Generating evaluation metrics...")
        try:
            if "clean" in datasets_cache and cnn_model is not None:
                attacked_datasets = {dataset_type: dataset_subset}
                compute_all_metrics(
                    clean_dataset=datasets_cache["clean"],
                    attacked_datasets=attacked_datasets,
                    model=cnn_model,
                    device=device,
                    output_dir=str(outputs_dir),
                )
                print(f"[API] Evaluation metrics saved to {outputs_dir}")
            else:
                print("[API] Skipping compute_all_metrics; clean dataset or model missing.")
        except Exception as metrics_error:
            print(f"[API] Metrics generation failed: {metrics_error}")
            import traceback
            traceback.print_exc()

    except Exception as viz_error:
        print(f"[API] Visualization generation failed: {viz_error}")
        import traceback
        traceback.print_exc()


# API Endpoints
@app.get("/")
async def root():
    return {"message": "Poisoning Attack Detection API", "status": "running"}


@app.post("/send_data")
async def send_data(request: DatasetRequest):
    """Load dataset and return sample images."""
    try:
        print(f"[DEBUG] /send_data called with dataset_type={request.dataset_type}, mix_flipped={request.mix_flipped}, mix_corrupted={request.mix_corrupted}, mix_fgsm={request.mix_fgsm}")
        
        valid_types = ["clean", "flipped", "corrupted", "fgsm", "mixed"]
        if request.dataset_type not in valid_types:
            print(f"[DEBUG] Invalid dataset type: {request.dataset_type}")
            raise HTTPException(status_code=400, detail="Invalid dataset type")
        
        # Special handling for "mixed" - build on-demand
        if request.dataset_type == "mixed":
            print(f"[DEBUG] Building mixed dataset with percentages: flipped={request.mix_flipped}, corrupted={request.mix_corrupted}, fgsm={request.mix_fgsm}")
            print(f"[DEBUG] Available datasets in cache: {list(datasets_cache.keys())}")
            print(f"[DEBUG] Available metadata keys: {list(dataset_metadata.keys())}")
            
            # Check that required datasets exist
            required = ["clean", "flipped", "corrupted", "fgsm"]
            missing = [d for d in required if d not in datasets_cache]
            if missing:
                print(f"[DEBUG] Missing required datasets: {missing}")
                raise HTTPException(status_code=400, detail=f"Missing required datasets: {missing}. Please initialize first.")
            
            try:
                dataset, mask = build_mixed_dataset(
                    request.mix_flipped or 0,
                    request.mix_corrupted or 0,
                    request.mix_fgsm or 0,
                )
                print(f"[DEBUG] Mixed dataset built successfully. Size: {len(dataset)}, mask shape: {mask.shape}")
            except ValueError as mix_err:
                print(f"[DEBUG] ValueError in build_mixed_dataset: {mix_err}")
                raise HTTPException(status_code=400, detail=str(mix_err))
            except Exception as mix_err:
                print(f"[DEBUG] Unexpected error in build_mixed_dataset: {type(mix_err).__name__}: {mix_err}")
                import traceback
                print(f"[DEBUG] Traceback:\n{traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=f"Error building mixed dataset: {str(mix_err)}")
            
            datasets_cache["mixed"] = dataset
            dataset_metadata["mixed"] = {"attack_mask": mask}
            print(f"[DEBUG] Mixed dataset cached successfully")
        else:
            # Load regular dataset (clean, flipped, corrupted, fgsm)
            if request.dataset_type not in datasets_cache:
                print(f"[DEBUG] Dataset '{request.dataset_type}' not in cache. Cache keys: {list(datasets_cache.keys())}")
                if not datasets_cache:
                    print("[DEBUG] Cache is empty, initializing all datasets...")
                    # Initialize all datasets
                    all_datasets, all_metadata = prepare_cifar10_and_attacks("backend/data", device)
                    datasets_cache.update(all_datasets)
                    dataset_metadata.update(all_metadata)
                    print(f"[DEBUG] Initialized datasets: {list(datasets_cache.keys())}")
                    print(f"[DEBUG] Initialized metadata keys: {list(dataset_metadata.keys())}")
                else:
                    print(f"[DEBUG] Cache has datasets but not '{request.dataset_type}'. Available: {list(datasets_cache.keys())}")
                    raise HTTPException(status_code=404, detail="Dataset not found. Please initialize first.")
            
            dataset = datasets_cache[request.dataset_type]
            print(f"[DEBUG] Loaded dataset '{request.dataset_type}' from cache. Size: {len(dataset)}")
        
        print(f"[DEBUG] Getting sample images from dataset of size {len(dataset)}")
        samples = get_sample_images(dataset, num_samples=10)
        print(f"[DEBUG] Got {len(samples)} sample images")
        
        response = {
            "status": "success",
            "dataset_type": request.dataset_type,
            "total_samples": len(dataset),
            "samples": samples
        }
        if request.dataset_type == "mixed":
            response["mix_config"] = {
                "flipped": request.mix_flipped or 0,
                "corrupted": request.mix_corrupted or 0,
                "fgsm": request.mix_fgsm or 0,
                "clean": max(
                    0,
                    100 - ((request.mix_flipped or 0) + (request.mix_corrupted or 0) + (request.mix_fgsm or 0)),
                ),
            }
        print(f"[DEBUG] /send_data returning success response")
        return response
    except HTTPException:
        raise
    except Exception as e:
        print(f"[DEBUG] Unexpected error in /send_data: {type(e).__name__}: {e}")
        import traceback
        print(f"[DEBUG] Traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_sample_images")
async def get_sample_images_endpoint(dataset_type: str, num_samples: int = 10):
    """Get random sample images from dataset."""
    try:
        if dataset_type not in datasets_cache:
            raise HTTPException(status_code=404, detail="Dataset not loaded")
        
        dataset = datasets_cache[dataset_type]
        samples = get_sample_images(dataset, num_samples)
        
        return {
            "status": "success",
            "samples": samples
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run_detection")
async def run_detection(request: DetectionRequest):
    """Run all 4 detectors on the specified dataset."""
    try:
        print(f"[API] Starting /run_detection for dataset: {request.dataset_type}")
        if request.dataset_type not in datasets_cache:
            raise HTTPException(status_code=404, detail="Dataset not loaded")
        
        dataset = datasets_cache[request.dataset_type]
        threshold = threshold_cache.get(request.dataset_type, request.threshold)
        print(f"[API] /run_detection | dataset={request.dataset_type} | threshold={threshold:.2f}")
        
        # Initialize models if needed
        print("[API] Initializing models...")
        if "cnn" not in models_cache:
            cnn = build_cnn().to(device)
            if os.path.exists("backend/models/cnn_weights.pt"):
                cnn.load_state_dict(torch.load("backend/models/cnn_weights.pt", map_location=device, weights_only=False))
            models_cache["cnn"] = cnn
        
        if "autoencoder" not in models_cache:
            autoenc = build_autoencoder().to(device)
            if os.path.exists("backend/models/autoencoder_weights.pt"):
                autoenc.load_state_dict(torch.load("backend/models/autoencoder_weights.pt", map_location=device, weights_only=False))
            models_cache["autoencoder"] = autoenc

        # Use full dataset for better accuracy (comment out the subset code)
        # For very large datasets, you might want to keep the subset, but for better results, use full dataset
        dataset_subset = dataset
        print(f"[API] Using full dataset with {len(dataset_subset)} samples.")
        
        # If you want to use subset for faster computation, uncomment the following:
        # subset_size = min(5000, len(dataset))
        # if len(dataset) > subset_size:
        #     indices = torch.randperm(len(dataset))[:subset_size]
        #     x_subset = dataset.tensors[0][indices]
        #     y_subset = dataset.tensors[1][indices]
        #     dataset_subset = TensorDataset(x_subset, y_subset)
        #     print(f"[API] Using subset of {subset_size} samples (original {len(dataset)}).")
        # else:
        #     dataset_subset = dataset

        print("[API] Running all detectors...")
        public_results, internal_indices_map, combined_full_indices = run_all_detectors(
            dataset_subset, threshold, models_cache, device
        )

        print("[API] Running adaptive threshold refinement...")
        (
            final_threshold,
            final_public_results,
            final_internal_indices,
            final_combined_indices,
            adaptive_log,
        ) = adaptive_threshold_refinement(
            dataset_subset,
            threshold,
            models_cache,
            device,
            base_results=public_results,
            base_internal_indices=internal_indices_map,
        )

        detection_results[request.dataset_type] = final_public_results
        threshold_cache[request.dataset_type] = final_threshold

        # Generate visualizations including SHAP and LIME
        print("[API] Generating visualizations...")
        generate_visualizations(
            dataset_type=request.dataset_type,
            dataset_subset=dataset_subset,
            anomaly_indices_map=final_internal_indices,
            cnn_model=models_cache["cnn"],
            autoencoder_model=models_cache["autoencoder"],
            device=device,
        )
        print("[API] Visualizations generated.")

        response = {
            "status": "success",
            "dataset_type": request.dataset_type,
            "threshold": final_threshold,
            "detectors": final_public_results,
            "total_anomalies": len(final_combined_indices),
            "subset_size": len(dataset_subset),
            "adaptive_log": adaptive_log,
        }
        print(f"[API] /run_detection completed | total_anomalies={len(final_combined_indices)}")
        return sanitize_for_json(response)
    except Exception as e:
        print(f"[API] /run_detection error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback_loop")
async def feedback_loop(request: FeedbackRequest):
    """Run adaptive feedback loop with DDPG."""
    try:
        if request.dataset_type not in datasets_cache:
            raise HTTPException(status_code=404, detail="Dataset not loaded")
        
        if "clean" not in datasets_cache:
            raise HTTPException(status_code=404, detail="Clean dataset not loaded")
        
        # Initialize models if needed
        if "cnn" not in models_cache:
            cnn = build_cnn().to(device)
            if os.path.exists("backend/models/cnn_weights.pt"):
                cnn.load_state_dict(torch.load("backend/models/cnn_weights.pt", map_location=device, weights_only=False))
            models_cache["cnn"] = cnn
        
        if "autoencoder" not in models_cache:
            autoenc = build_autoencoder().to(device)
            if os.path.exists("backend/models/autoencoder_weights.pt"):
                autoenc.load_state_dict(torch.load("backend/models/autoencoder_weights.pt", map_location=device, weights_only=False))
            models_cache["autoencoder"] = autoenc

        attack_mask = dataset_metadata.get(request.dataset_type, {}).get("attack_mask")
        
        feedback_result = adaptive_detection_with_feedback(
            clean_dataset=datasets_cache["clean"],
            attacked_dataset=datasets_cache[request.dataset_type],
            attacked_name=request.dataset_type,
            cnn_model=models_cache["cnn"],
            autoencoder_model=models_cache["autoencoder"],
            device=device,
            initial_percentile=95.0,
            num_iterations=request.num_iterations,
            output_dir="backend/static/outputs",
            use_subset=False,
            attack_mask=attack_mask
        )
        
        response = {
            "status": "success",
            "dataset_type": request.dataset_type,
            "iterations": request.num_iterations,
            "results": feedback_result["results"],
            "final_metrics": feedback_result["final_metrics"]
        }
        return sanitize_for_json(response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_visuals")
async def get_visuals(dataset_type: str):
    """Get visualization images (SHAP, LIME, confusion matrix, etc.)."""
    try:
        print(f"[API] get_visuals called with dataset_type: {dataset_type}")
        visuals_dir = f"backend/static/outputs"
        files = {
            "accuracy_comparison": f"{visuals_dir}/accuracy_comparison.png",
            "confusion_matrix": f"{visuals_dir}/confusion_matrices.png",
            "feedback_learning": f"{visuals_dir}/feedback_learning_{dataset_type}.png",
            "shap": f"{visuals_dir}/shap_{dataset_type}/shap_explanations.png",
            "lime": f"{visuals_dir}/lime_{dataset_type}/lime_explanations.png"
        }
        
        print(f"[API] Checking for files: {files}")
        available_files = {}
        base_static = Path("backend/static")
        for name, path in files.items():
            print(f"[API] Checking if {path} exists: {os.path.exists(path)}")
            if os.path.exists(path):
                try:
                    rel_path = Path(path).relative_to(base_static)
                except ValueError:
                    rel_path = Path(path)
                available_files[name] = f"/static/{rel_path.as_posix()}"
                print(f"[API] Found {name} at {available_files[name]}")
        
        print(f"[API] Available files: {available_files}")
        return {
            "status": "success",
            "available_visuals": available_files
        }
    except Exception as e:
        print(f"[API] Error in get_visuals: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics(dataset_type: str):
    """Get evaluation metrics for a dataset."""
    try:
        if dataset_type not in datasets_cache:
            raise HTTPException(status_code=404, detail="Dataset not loaded")
        
        if "cnn" not in models_cache:
            raise HTTPException(status_code=404, detail="Model not loaded")
        
        dataset = datasets_cache[dataset_type]
        cnn = models_cache["cnn"]
        
        accuracy = eval_cnn(cnn, dataset, device=device)
        
        return {
            "status": "success",
            "dataset_type": dataset_type,
            "accuracy": accuracy
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# NEW: Add endpoint for getting metrics for cleaned dataset
@app.post("/metrics_cleaned")
async def get_metrics_cleaned(request: DetectionRequest):
    """Get evaluation metrics for a cleaned dataset."""
    try:
        print(f"[API] get_metrics_cleaned called with request: {request}")
        if request.dataset_type not in datasets_cache:
            raise HTTPException(status_code=404, detail="Dataset not loaded")
        
        if "cnn" not in models_cache:
            raise HTTPException(status_code=404, detail="Model not loaded")
        
        # Get original dataset
        dataset = datasets_cache[request.dataset_type]
        cnn = models_cache["cnn"]
        threshold = request.threshold
        print(f"[API] Dataset size: {len(dataset)}, Threshold: {threshold}")
        
        # Run fusion detector to get cleaned dataset
        try:
            anomaly_indices, cleaned_dataset, scores = detect_anomalies_fusion(
                dataset, cnn, models_cache["autoencoder"], device, percentile=threshold
            )
            print(f"[API] Detected {len(anomaly_indices)} anomalies, Cleaned dataset size: {len(cleaned_dataset)}")
            
            # Check if we have enough samples to retrain
            if len(cleaned_dataset) < 10:
                print("[API] Warning: Very few samples in cleaned dataset, using original dataset for evaluation")
                original_accuracy = eval_cnn(cnn, dataset, device=device)
                cleaned_accuracy = eval_cnn(cnn, dataset, device=device)  # Same as original since too few samples
                improvement = 0.0
                return {
                    "status": "warning",
                    "message": "Very few samples in cleaned dataset",
                    "dataset_type": request.dataset_type,
                    "original_accuracy": original_accuracy,
                    "cleaned_accuracy": cleaned_accuracy,
                    "accuracy_improvement": improvement
                }
            
            # Create a copy of the model for retraining
            from backend.models.cnn_model import build_cnn
            retrained_cnn = build_cnn().to(device)
            retrained_cnn.load_state_dict(cnn.state_dict())  # Start with same weights
            
            # Retrain model on cleaned dataset
            print(f"[API] Retraining model on cleaned dataset with {len(cleaned_dataset)} samples...")
            if len(cleaned_dataset) > 0:
                # Use a smaller number of epochs for faster retraining
                from backend.models.cnn_model import train_cnn
                # Use fewer epochs but with a validation split to prevent overfitting
                train_cnn(retrained_cnn, cleaned_dataset, device=device, epochs=3, batch_size=64)
            
            # Evaluate accuracy on both datasets
            original_accuracy = eval_cnn(cnn, dataset, device=device)
            cleaned_accuracy = eval_cnn(retrained_cnn, cleaned_dataset, device=device)
            improvement = cleaned_accuracy - original_accuracy
            
            print(f"[API] Original accuracy: {original_accuracy:.4f}, Cleaned accuracy: {cleaned_accuracy:.4f}, Improvement: {improvement:+.4f}")
            
            return {
                "status": "success",
                "dataset_type": request.dataset_type,
                "original_accuracy": original_accuracy,
                "cleaned_accuracy": cleaned_accuracy,
                "accuracy_improvement": improvement
            }
        except Exception as e:
            print(f"[API] Error in fusion detection: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Error in fusion detection: {str(e)}")
            
    except Exception as e:
        print(f"[API] Error in get_metrics_cleaned: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/initialize")
async def initialize():
    """Initialize all datasets and models."""
    try:
        # Prepare datasets (FGSM will use dummy model initially)
        all_datasets, all_metadata = prepare_cifar10_and_attacks("backend/data", device)
        datasets_cache.update(all_datasets)
        dataset_metadata.update(all_metadata)
        
        # Build and train models
        cnn = build_cnn().to(device)
        autoenc = build_autoencoder().to(device)
        
        # Train if weights don't exist
        if not os.path.exists("backend/models/cnn_weights.pt"):
            train_cnn(cnn, datasets_cache["clean"], device=device, epochs=10, batch_size=128)
            os.makedirs("backend/models", exist_ok=True)
            torch.save(cnn.state_dict(), "backend/models/cnn_weights.pt")
        
        if not os.path.exists("backend/models/autoencoder_weights.pt"):
            train_autoencoder(autoenc, datasets_cache["clean"], device=device, epochs=10, batch_size=128)
            torch.save(autoenc.state_dict(), "backend/models/autoencoder_weights.pt")
        
        # Load weights
        cnn.load_state_dict(torch.load("backend/models/cnn_weights.pt", map_location=device, weights_only=False))
        autoenc.load_state_dict(torch.load("backend/models/autoencoder_weights.pt", map_location=device, weights_only=False))
        
        models_cache["cnn"] = cnn
        models_cache["autoencoder"] = autoenc
        
        # Regenerate FGSM attacks with trained model for better effectiveness
        print("[API] Regenerating FGSM attacks with trained model...")
        from backend.utils.attacks.fgsm import apply_fgsm_attack
        x_clean, y_clean = datasets_cache["clean"].tensors
        subset_size = min(2000, len(x_clean))
        fgsm_path = "backend/data/fgsm.pt"
        apply_fgsm_attack(
            x_clean[:subset_size], 
            y_clean[:subset_size], 
            fgsm_path, 
            epsilon=0.15,  # Increased epsilon for more effective attack
            model=cnn, 
            device=device
        )
        fgsm_data = torch.load(fgsm_path, weights_only=False)
        rest_x = x_clean[subset_size:]
        rest_y = y_clean[subset_size:]
        fgsm_full_x = torch.cat([fgsm_data[0], rest_x], dim=0)
        fgsm_full_y = torch.cat([fgsm_data[1], rest_y], dim=0)
        fgsm_ds = TensorDataset(fgsm_full_x, fgsm_full_y)
        torch.save((fgsm_full_x, fgsm_full_y, torch.arange(len(fgsm_full_x))), fgsm_path)
        datasets_cache["fgsm"] = fgsm_ds
        mask_fgsm = np.zeros(len(fgsm_full_y), dtype=int)
        mask_fgsm[:subset_size] = 1
        dataset_metadata["fgsm"] = {"attack_mask": mask_fgsm}
        print("[API] FGSM attacks regenerated with trained model.")
        
        return {
            "status": "success",
            "message": "All datasets and models initialized",
            "datasets": list(datasets_cache.keys()),
            "models": list(models_cache.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Serve static files
@app.get("/static/{file_path:path}")
async def serve_static(file_path: str):
    """Serve static files."""
    file_location = f"backend/static/{file_path}"
    if os.path.exists(file_location):
        return FileResponse(file_location)
    raise HTTPException(status_code=404, detail="File not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8081)
