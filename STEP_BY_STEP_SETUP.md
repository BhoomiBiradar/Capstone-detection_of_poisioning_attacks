# 🚀 Step-by-Step Setup Guide

## Complete Setup from Scratch

Follow these steps in order to set up the project with frontend integration.

---

## **STEP 1: Run Migration Script** ✅

```bash
python MIGRATION_SCRIPT.py
```

**What it does:**
- Creates `backend/` directory structure
- Moves `models/` → `backend/models/`
- Moves `detectors/` → `backend/detectors/`
- Moves `feedback/` → `backend/feedback/`
- Moves `evaluation_metrics.py` → `backend/evaluation_metrics.py`
- Creates `__init__.py` files

**Expected output:**
```
✓ Created backend/models
✓ Created backend/detectors
...
✓ Moved evaluation_metrics.py → backend/evaluation_metrics.py
✅ Migration complete!
```

---

## **STEP 2: Verify File Structure** ✅

Check that your structure looks like this:

```
project/
├── backend/
│   ├── models/
│   │   ├── cnn_model.py
│   │   ├── autoencoder_model.py
│   │   └── __init__.py
│   ├── detectors/
│   │   ├── centroid_detector.py
│   │   ├── knn_detector.py
│   │   ├── autoencoder_detector.py
│   │   ├── gradient_filter.py
│   │   └── __init__.py
│   ├── feedback/
│   │   ├── ddpg_agent.py
│   │   ├── feedback_loop.py
│   │   └── __init__.py
│   ├── utils/
│   │   ├── attacks/
│   │   │   ├── label_flipping.py
│   │   │   ├── corruption.py
│   │   │   ├── fgsm.py
│   │   │   └── __init__.py
│   │   ├── data_preparation.py
│   │   └── __init__.py
│   ├── api.py
│   ├── evaluation_metrics.py
│   └── __init__.py
│
├── frontend/
│   └── streamlit_app.py
│
├── data/                    # Old location (can delete after migration)
├── models/                  # Old location (can delete after migration)
├── detectors/               # Old location (can delete after migration)
├── feedback/                # Old location (can delete after migration)
│
├── main.py
├── requirements.txt
└── MIGRATION_SCRIPT.py
```

---

## **STEP 3: Update Imports in Files** ✅

### 3.1 Update `main.py`

**Find and replace:**
```python
# OLD
from data.prepare_and_attacks import prepare_cifar10_and_attacks
from models.cnn_model import build_cnn, train_cnn, eval_cnn
from models.autoencoder_model import build_autoencoder, train_autoencoder
from detectors.centroid_detector import detect_anomalies_centroid
from evaluation_metrics import compute_all_metrics

# NEW
from backend.utils.data_preparation import prepare_cifar10_and_attacks
from backend.models.cnn_model import build_cnn, train_cnn, eval_cnn
from backend.models.autoencoder_model import build_autoencoder, train_autoencoder
from backend.detectors.centroid_detector import detect_anomalies_centroid
from backend.evaluation_metrics import compute_all_metrics
```

### 3.2 Update `evaluation_metrics.py` (if it has imports)

Check if `backend/evaluation_metrics.py` has any imports that need updating.

---

## **STEP 4: Install Dependencies** ✅

```bash
pip install -r requirements.txt
```

**This installs:**
- PyTorch, torchvision
- FastAPI, uvicorn
- Streamlit
- scikit-learn, matplotlib, etc.

---

## **STEP 5: Test Backend API** ✅

### 5.1 Start Backend

```bash
cd backend
python api.py
```

**Expected output:**
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 5.2 Test API (in another terminal)

```bash
# Test if API is running
curl http://localhost:8000/

# Or open in browser:
# http://localhost:8000/docs
```

**You should see:** API documentation page (Swagger UI)

---

## **STEP 6: Test Streamlit Frontend** ✅

### 6.1 Start Frontend (in new terminal)

```bash
cd frontend
streamlit run streamlit_app.py
```

**Expected output:**
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

### 6.2 Open Browser

Go to: **http://localhost:8501**

**You should see:**
- Streamlit dashboard
- "Initialize System" button in sidebar
- Dataset selection dropdown

---

## **STEP 7: Run Full Pipeline** ✅

### 7.1 In Streamlit Dashboard:

1. **Click "🚀 Initialize System"** (in sidebar)
   - This downloads CIFAR-10 and trains models
   - Takes ~5-10 minutes first time
   - Shows progress messages

2. **Select Dataset** (e.g., "flipped")
   - Click "📥 Load Dataset"
   - See sample images appear

3. **Run Detection**
   - Adjust threshold slider
   - Click "▶️ Run Detection"
   - See detection results table

4. **Run Feedback Loop**
   - Set iterations (e.g., 10)
   - Click "▶️ Run Feedback Loop"
   - Watch learning curves appear

---

## **STEP 8: Clean Up Old Files (Optional)** ✅

After verifying everything works, you can delete old directories:

```bash
# Only do this AFTER everything works!
rmdir /s data        # Windows
rmdir /s models      # Windows
rmdir /s detectors   # Windows
rmdir /s feedback    # Windows

# Or on Linux/Mac:
# rm -rf data models detectors feedback
```

**⚠️ Warning:** Only delete if files are successfully in `backend/`!

---

## **Troubleshooting** 🔧

### Issue: Import errors

**Solution:**
```bash
# Make sure you're in project root
cd C:\Users\91961\OneDrive\Desktop\cap

# Check if files exist
dir backend\models\cnn_model.py
dir backend\detectors\centroid_detector.py
```

### Issue: ModuleNotFoundError: No module named 'backend'

**Solution:**
```python
# Add project root to Python path
import sys
sys.path.insert(0, '.')

# Or run from project root:
python -m backend.api
```

### Issue: Backend won't start

**Solution:**
```bash
# Check if port 8000 is in use
netstat -ano | findstr :8000

# Change port in backend/api.py:
uvicorn.run(app, port=8001)
```

### Issue: Streamlit can't connect to backend

**Solution:**
1. Check backend is running: `curl http://localhost:8000/`
2. Check `API_BASE_URL` in `frontend/streamlit_app.py`
3. Check CORS settings in `backend/api.py`

---

## **Quick Verification Checklist** ✅

- [ ] Migration script ran successfully
- [ ] Files are in `backend/` directories
- [ ] `backend/api.py` starts without errors
- [ ] API docs accessible at http://localhost:8000/docs
- [ ] Streamlit app starts without errors
- [ ] Dashboard loads at http://localhost:8501
- [ ] "Initialize System" button works
- [ ] Datasets load successfully
- [ ] Detection runs successfully
- [ ] Feedback loop runs successfully

---

## **What's Next?** 🎯

Once everything is working:

1. **Customize UI**: Edit `frontend/streamlit_app.py`
2. **Add Features**: Modify `backend/api.py` endpoints
3. **Deploy**: Follow deployment guide in `README_FRONTEND.md`

---

## **Summary of Changes Made** 📋

✅ **New Attack Functions**: `backend/utils/attacks/` (3 files)
✅ **FastAPI Backend**: `backend/api.py` (REST API)
✅ **Streamlit Frontend**: `frontend/streamlit_app.py` (Dashboard)
✅ **Optimized Feedback Loop**: Faster execution, dynamic thresholds
✅ **Updated Structure**: All files moved to `backend/`

**You're all set!** 🎉










