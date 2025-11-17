# 🚀 Setup Guide - Frontend Integration

## Project Structure

```
project_root/
│
├── backend/
│   ├── api.py                      # FastAPI backend server
│   ├── models/                     # CNN, Autoencoder models
│   ├── detectors/                  # Detection algorithms
│   ├── feedback/                   # DDPG feedback loop
│   ├── utils/
│   │   ├── attacks/                # NEW: Attack functions
│   │   │   ├── label_flipping.py
│   │   │   ├── corruption.py
│   │   │   └── fgsm.py
│   │   └── data_preparation.py     # Updated data prep
│   ├── data/                       # Dataset files
│   └── static/outputs/             # Generated visualizations
│
├── frontend/
│   └── streamlit_app.py            # Streamlit dashboard
│
└── main.py                         # Original CLI version (still works)
```

## 📦 Installation

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt
```

### 2. Move Existing Files to Backend

The existing files need to be moved to the `backend/` directory:

```bash
# Move model files
mv models/ backend/models/

# Move detector files
mv detectors/ backend/detectors/

# Move feedback files
mv feedback/ backend/feedback/

# Move evaluation_metrics.py
mv evaluation_metrics.py backend/
```

**Note**: The new attack functions are already in `backend/utils/attacks/`

## 🎯 Running the System

### Streamlit Frontend (Recommended)

1. **Start Backend API:**
```bash
cd backend
python api.py
# Or: uvicorn api:app --reload --port 8000
```

2. **Start Streamlit Frontend:**
```bash
cd frontend
streamlit run streamlit_app.py
```

3. **Open Browser:**
   - Streamlit Dashboard: http://localhost:8501
   - API Documentation: http://localhost:8000/docs

### Original CLI (Still Works)

```bash
python main.py
```

## 🔄 Migration Steps

### Step 1: Update Imports

Update imports in existing files to use new structure:

**Old:**
```python
from data.prepare_and_attacks import prepare_cifar10_and_attacks
from models.cnn_model import build_cnn
```

**New:**
```python
from backend.utils.data_preparation import prepare_cifar10_and_attacks
from backend.models.cnn_model import build_cnn
```

### Step 2: Update Data Preparation

The new data preparation uses the new attack functions:

```python
from backend.utils.attacks import (
    apply_label_flipping_attack,
    apply_feature_corruption_attack,
    apply_fgsm_attack
)
```

### Step 3: Update File Paths

Update all file paths to use `backend/` prefix:

- `data/` → `backend/data/`
- `models/` → `backend/models/`
- `outputs/` → `backend/static/outputs/`

## 🧪 Testing the API

### Using curl:

```bash
# Initialize system
curl -X POST http://localhost:8000/initialize

# Load dataset
curl -X POST http://localhost:8000/send_data \
  -H "Content-Type: application/json" \
  -d '{"dataset_type": "flipped"}'

# Run detection
curl -X POST http://localhost:8000/run_detection \
  -H "Content-Type: application/json" \
  -d '{"dataset_type": "flipped", "threshold": 95.0}'
```

### Using Python:

```python
import requests

# Initialize
response = requests.post("http://localhost:8000/initialize")
print(response.json())

# Load dataset
response = requests.post("http://localhost:8000/send_data", 
                        json={"dataset_type": "flipped"})
print(response.json())
```

## 📝 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API status |
| `/initialize` | POST | Initialize datasets and models |
| `/send_data` | POST | Load dataset and get samples |
| `/get_sample_images` | GET | Get random sample images |
| `/run_detection` | POST | Run all 4 detectors |
| `/feedback_loop` | POST | Run adaptive feedback loop |
| `/get_visuals` | GET | Get visualization files |
| `/metrics` | GET | Get evaluation metrics |
| `/static/{path}` | GET | Serve static files |

## 🐛 Troubleshooting

### Issue: Module not found errors

**Solution:** Make sure you're running from project root and paths are correct:
```bash
# From project root
python -m backend.api
```

### Issue: Port already in use

**Solution:** Change port in `api.py`:
```python
uvicorn.run(app, host="0.0.0.0", port=8001)  # Use different port
```

### Issue: Streamlit cannot connect to backend

**Solution:** 
- Check backend is running: `curl http://localhost:8000/`
- Check API_BASE_URL in `streamlit_app.py` matches backend URL
- Ensure CORS is enabled in backend (already done)

## ✅ Verification

1. Backend API is running: http://localhost:8000/docs
2. Frontend can connect: Open Streamlit and check for errors
3. Datasets load: Click "Initialize System" in Streamlit sidebar
4. Detection works: Run detection and see results

## 🎉 Next Steps

1. Customize the Streamlit UI
2. Add more visualization options
3. Implement real-time updates (WebSocket)
4. Add authentication if needed
5. Deploy to cloud (Heroku, AWS, Streamlit Cloud, etc.)
