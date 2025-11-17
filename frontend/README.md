# Frontend - Poisoning Attack Detection Dashboard

## Streamlit Frontend

### Quick Start

1. **Start the backend API:**
```bash
cd backend
python api.py
# Or: uvicorn api:app --reload --port 8000
```

2. **Start the Streamlit frontend:**
```bash
cd frontend
streamlit run streamlit_app.py
```

3. **Open browser:**
   - Streamlit Dashboard: http://localhost:8501
   - API docs: http://localhost:8000/docs

## Features

- **Dataset Selection**: Load and preview different attack datasets
- **Detection Panel**: Run 4 detection algorithms with adjustable threshold
- **Feedback Loop**: Adaptive learning with DDPG agent
- **Visualizations**: View detection results and learning curves
- **Metrics**: Real-time evaluation metrics
- **Sample Images**: Gallery of dataset samples

## Usage

1. Click "Initialize System" in the sidebar
2. Select a dataset type (clean, flipped, corrupted, fgsm)
3. Click "Load Dataset" to view sample images
4. Adjust threshold and click "Run Detection"
5. Click "Run Feedback Loop" to see adaptive learning
6. View metrics and visualizations
