"""
FastAPI Backend for Poisoning Attack Detection System
"""
import os
import sys
import base64
import io
from pathlib import Path
from typing import Dict, List, Optional
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset

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
datasets_cache = {}
models_cache = {}
detection_results = {}


# Request/Response Models
class DatasetRequest(BaseModel):
    dataset_type: str  # "clean", "flipped", "corrupted", "fgsm"


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
        else:
            print("[API] No anomalies detected; skipping SHAP/LIME.")

        # Accuracy / confusion matrix plots
        if "clean" in datasets_cache and cnn_model is not None:
            attacked_datasets = {dataset_type: dataset_subset}
            compute_all_metrics(
                clean_dataset=datasets_cache["clean"],
                attacked_datasets=attacked_datasets,
                model=cnn_model,
                device=device,
                output_dir=str(outputs_dir),
            )
        else:
            print("[API] Skipping compute_all_metrics; clean dataset or model missing.")

    except Exception as viz_error:
        print(f"[API] Visualization generation failed: {viz_error}")


# API Endpoints
@app.get("/")
async def root():
    return {"message": "Poisoning Attack Detection API", "status": "running"}


@app.post("/send_data")
async def send_data(request: DatasetRequest):
    """Load dataset and return sample images."""
    try:
        if request.dataset_type not in ["clean", "flipped", "corrupted", "fgsm"]:
            raise HTTPException(status_code=400, detail="Invalid dataset type")
        
        # Load dataset if not cached
        if request.dataset_type not in datasets_cache:
            if not datasets_cache:
                # Initialize all datasets
                all_datasets = prepare_cifar10_and_attacks("backend/data", device)
                datasets_cache.update(all_datasets)
            else:
                raise HTTPException(status_code=404, detail="Dataset not found. Please initialize first.")
        
        dataset = datasets_cache[request.dataset_type]
        samples = get_sample_images(dataset, num_samples=10)
        
        return {
            "status": "success",
            "dataset_type": request.dataset_type,
            "total_samples": len(dataset),
            "samples": samples
        }
    except Exception as e:
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
        if request.dataset_type not in datasets_cache:
            raise HTTPException(status_code=404, detail="Dataset not loaded")
        
        dataset = datasets_cache[request.dataset_type]
        threshold = request.threshold
        print(f"[API] /run_detection | dataset={request.dataset_type} | threshold={threshold}")
        
        # Initialize models if needed
        if "cnn" not in models_cache:
            cnn = build_cnn().to(device)
            if os.path.exists("backend/models/cnn_weights.pt"):
                cnn.load_state_dict(torch.load("backend/models/cnn_weights.pt", map_location=device))
            models_cache["cnn"] = cnn
        
        if "autoencoder" not in models_cache:
            autoenc = build_autoencoder().to(device)
            if os.path.exists("backend/models/autoencoder_weights.pt"):
                autoenc.load_state_dict(torch.load("backend/models/autoencoder_weights.pt", map_location=device))
            models_cache["autoencoder"] = autoenc

        # Use subset for speed
        subset_size = min(5000, len(dataset))
        if len(dataset) > subset_size:
            indices = torch.randperm(len(dataset))[:subset_size]
            x_subset = dataset.tensors[0][indices]
            y_subset = dataset.tensors[1][indices]
            dataset_subset = TensorDataset(x_subset, y_subset)
            print(f"[API] Using subset of {subset_size} samples (original {len(dataset)}).")
        else:
            dataset_subset = dataset

        results = {}
        
        # Run all detectors
        try:
            anomaly_idx_centroid, cleaned_centroid = detect_anomalies_centroid(
                dataset_subset, percentile=threshold
            )
            results["centroid"] = {
                "anomalies_count": len(anomaly_idx_centroid),
                "anomaly_indices": anomaly_idx_centroid[:100]  # Limit for response
            }
            print(f"[API] Centroid detector found {len(anomaly_idx_centroid)} anomalies.")
        except Exception as e:
            results["centroid"] = {"error": str(e)}
            print(f"[API] Centroid detector error: {e}")
        
        try:
            anomaly_idx_knn, cleaned_knn = detect_anomalies_knn(
                dataset_subset, k=5, percentile=threshold
            )
            results["knn"] = {
                "anomalies_count": len(anomaly_idx_knn),
                "anomaly_indices": anomaly_idx_knn[:100]
            }
            print(f"[API] KNN detector found {len(anomaly_idx_knn)} anomalies.")
        except Exception as e:
            results["knn"] = {"error": str(e)}
            print(f"[API] KNN detector error: {e}")
        
        try:
            anomaly_idx_ae, cleaned_ae = detect_anomalies_autoencoder(
                dataset_subset, models_cache["autoencoder"], device, percentile=threshold
            )
            results["autoencoder"] = {
                "anomalies_count": len(anomaly_idx_ae),
                "anomaly_indices": anomaly_idx_ae[:100]
            }
            print(f"[API] Autoencoder detector found {len(anomaly_idx_ae)} anomalies.")
        except Exception as e:
            results["autoencoder"] = {"error": str(e)}
            print(f"[API] Autoencoder detector error: {e}")
        
        try:
            anomaly_idx_grad, cleaned_grad = detect_anomalies_gradient(
                dataset_subset, models_cache["cnn"], device, percentile=threshold
            )
            results["gradient"] = {
                "anomalies_count": len(anomaly_idx_grad),
                "anomaly_indices": anomaly_idx_grad[:100]
            }
            print(f"[API] Gradient detector found {len(anomaly_idx_grad)} anomalies.")
        except Exception as e:
            results["gradient"] = {"error": str(e)}
            print(f"[API] Gradient detector error: {e}")
        
        # Combine all anomalies
        all_anomalies = set()
        for detector_result in results.values():
            if "anomaly_indices" in detector_result:
                all_anomalies.update(detector_result["anomaly_indices"])
        
        detection_results[request.dataset_type] = results

        # Generate visualizations
        anomaly_indices_map = {
            name: detector_result.get("anomaly_indices", [])
            for name, detector_result in results.items()
            if "anomaly_indices" in detector_result
        }
        generate_visualizations(
            dataset_type=request.dataset_type,
            dataset_subset=dataset_subset,
            anomaly_indices_map=anomaly_indices_map,
            cnn_model=models_cache["cnn"],
            autoencoder_model=models_cache["autoencoder"],
            device=device,
        )

        response = {
            "status": "success",
            "dataset_type": request.dataset_type,
            "threshold": threshold,
            "detectors": results,
            "total_anomalies": len(all_anomalies),
            "subset_size": len(dataset_subset),
        }
        print(f"[API] /run_detection completed | total_anomalies={len(all_anomalies)}")
        return sanitize_for_json(response)
    except Exception as e:
        print(f"[API] /run_detection error: {e}")
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
                cnn.load_state_dict(torch.load("backend/models/cnn_weights.pt", map_location=device))
            models_cache["cnn"] = cnn
        
        if "autoencoder" not in models_cache:
            autoenc = build_autoencoder().to(device)
            if os.path.exists("backend/models/autoencoder_weights.pt"):
                autoenc.load_state_dict(torch.load("backend/models/autoencoder_weights.pt", map_location=device))
            models_cache["autoencoder"] = autoenc
        
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
            use_subset=True,  # Use subset for faster computation
            subset_size=5000  # Use 5000 samples instead of full dataset
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
        visuals_dir = f"backend/static/outputs"
        files = {
            "accuracy_comparison": f"{visuals_dir}/accuracy_comparison.png",
            "confusion_matrix": f"{visuals_dir}/confusion_matrices.png",
            "feedback_learning": f"{visuals_dir}/feedback_learning_{dataset_type}.png",
            "shap": f"{visuals_dir}/shap_{dataset_type}/shap_explanations.png",
            "lime": f"{visuals_dir}/lime_{dataset_type}/lime_explanations.png"
        }
        
        available_files = {}
        for name, path in files.items():
            if os.path.exists(path):
                available_files[name] = f"/static/outputs/{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}" if "/" in os.path.basename(os.path.dirname(path)) else f"/static/outputs/{os.path.basename(path)}"
        
        return {
            "status": "success",
            "available_visuals": available_files
        }
    except Exception as e:
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


@app.post("/initialize")
async def initialize():
    """Initialize all datasets and models."""
    try:
        # Prepare datasets
        all_datasets = prepare_cifar10_and_attacks("backend/data", device)
        datasets_cache.update(all_datasets)
        
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
        cnn.load_state_dict(torch.load("backend/models/cnn_weights.pt", map_location=device))
        autoenc.load_state_dict(torch.load("backend/models/autoencoder_weights.pt", map_location=device))
        
        models_cache["cnn"] = cnn
        models_cache["autoencoder"] = autoenc
        
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
    uvicorn.run(app, host="0.0.0.0", port=8000)

