# 🎨 Frontend Integration - Complete Guide

## Overview

This project includes a **Streamlit Dashboard** for interacting with the poisoning attack detection system, connected to a **FastAPI backend** that provides REST API endpoints.

---

## 🏗️ Architecture

```
┌─────────────────┐
│   Streamlit     │
│   Dashboard     │
└────────┬────────┘
         │ HTTP/REST API
         ▼
┌─────────────────┐
│   FastAPI       │
│   Backend       │  (api.py)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ML Models     │
│   Detectors     │
│   DDPG Agent    │
└─────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install Python dependencies
pip install -r requirements.txt
```

### Step 1: Start Backend

```bash
cd backend
python api.py
# Backend runs on http://localhost:8000
```

### Step 2: Start Streamlit Frontend

```bash
cd frontend
streamlit run streamlit_app.py
# Opens at http://localhost:8501
```

---

## 📡 API Endpoints

### Core Endpoints

| Endpoint | Method | Description | Request Body |
|----------|--------|-------------|--------------|
| `/initialize` | POST | Initialize datasets & models | None |
| `/send_data` | POST | Load dataset | `{"dataset_type": "flipped"}` |
| `/get_sample_images` | GET | Get sample images | Query: `dataset_type`, `num_samples` |
| `/run_detection` | POST | Run detectors | `{"dataset_type": "flipped", "threshold": 95.0}` |
| `/feedback_loop` | POST | Run DDPG feedback | `{"dataset_type": "flipped", "num_iterations": 10}` |
| `/get_visuals` | GET | Get visualization files | Query: `dataset_type` |
| `/metrics` | GET | Get evaluation metrics | Query: `dataset_type` |

### Example API Calls

```python
import requests

BASE_URL = "http://localhost:8000"

# Initialize
response = requests.post(f"{BASE_URL}/initialize")
print(response.json())

# Load dataset
response = requests.post(
    f"{BASE_URL}/send_data",
    json={"dataset_type": "flipped"}
)
data = response.json()
print(f"Loaded {data['total_samples']} samples")

# Run detection
response = requests.post(
    f"{BASE_URL}/run_detection",
    json={"dataset_type": "flipped", "threshold": 95.0}
)
results = response.json()
print(f"Found {results['total_anomalies']} anomalies")
```

---

## 🎨 Streamlit Dashboard Features

**Features:**
- ✅ One-click initialization
- ✅ Dataset selection and preview
- ✅ Interactive detection with threshold slider
- ✅ Real-time feedback loop visualization
- ✅ Metrics display
- ✅ Sample image gallery
- ✅ Detection results table
- ✅ Learning curves (4 subplots)

**Sections:**
- **Control Panel** (Sidebar): Initialize system, select dataset
- **Dataset Selection**: Load and preview datasets
- **Sample Images**: Gallery of 10 random samples
- **Detection Algorithms**: Run 4 detectors with adjustable threshold
- **Feedback Loop**: Adaptive learning visualization
- **Evaluation Metrics**: Real-time accuracy and F1 scores
- **Visualizations**: SHAP, LIME, confusion matrices

---

## 🔧 Configuration

### Backend Configuration

Edit `backend/api.py`:

```python
# Change port
uvicorn.run(app, host="0.0.0.0", port=8001)

# Change CORS origins (if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
)
```

### Frontend Configuration

**Streamlit** (`frontend/streamlit_app.py`):
```python
API_BASE_URL = "http://localhost:8000"  # Change if needed
```

---

## 📊 Workflow

### Typical User Flow

1. **Initialize System**
   - Click "🚀 Initialize System" button in sidebar
   - Backend downloads CIFAR-10 and trains models
   - Takes ~5-10 minutes first time

2. **Load Dataset**
   - Select dataset type (flipped/corrupted/fgsm) from dropdown
   - Click "📥 Load Dataset"
   - View sample images in gallery
   - See dataset statistics

3. **Run Detection**
   - Adjust threshold slider (80-99 percentile)
   - Click "▶️ Run Detection"
   - View results from 4 detectors in table
   - See anomaly counts and bar chart

4. **Run Feedback Loop**
   - Set number of iterations (1-20)
   - Click "▶️ Run Feedback Loop"
   - Watch threshold adapt over iterations
   - View learning curves (4 subplots)

5. **View Visualizations**
   - Click "🖼️ Get Visualizations"
   - Get SHAP/LIME explanations
   - View confusion matrices
   - See accuracy comparisons

---

## 🐛 Troubleshooting

### Backend Issues

**Problem:** Module import errors
```bash
# Solution: Run from project root
cd /path/to/project
python -m backend.api
```

**Problem:** Port already in use
```bash
# Solution: Change port in api.py
uvicorn.run(app, port=8001)
```

### Frontend Issues

**Problem:** Cannot connect to backend
```bash
# Solution: Check backend is running
curl http://localhost:8000/

# Check API_BASE_URL in streamlit_app.py matches backend URL
```

**Problem:** Images not loading
```bash
# Solution: Check static file paths
# Ensure backend/static/outputs/ exists
```

**Problem:** Streamlit not starting
```bash
# Solution: Check streamlit is installed
pip install streamlit

# Run from frontend directory
cd frontend
streamlit run streamlit_app.py
```

---

## 🚀 Deployment

### Backend Deployment

**Heroku:**
```bash
# Create Procfile
echo "web: uvicorn backend.api:app --host 0.0.0.0 --port \$PORT" > Procfile

# Deploy
git push heroku main
```

**Docker:**
```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Frontend Deployment

**Streamlit Cloud:**
- Push to GitHub
- Connect to Streamlit Cloud
- Set API_BASE_URL to deployed backend URL in `streamlit_app.py`

---

## 📝 Next Steps

1. **Customize UI**: Modify Streamlit components in `streamlit_app.py`
2. **Add Features**: Real-time updates, WebSocket support
3. **Authentication**: Add user login if needed
4. **Database**: Store results in database
5. **Monitoring**: Add logging and monitoring

---

## 🎉 Success!

You now have a fully functional Streamlit frontend connected to the FastAPI backend for poisoning attack detection!

**Questions?** Check the API docs at http://localhost:8000/docs
