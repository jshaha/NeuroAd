# NeuroAd

Neuroscience-powered ad analysis tool that uses brain encoding to predict visual impact and optimize advertising creative.

## How It Works

1. **ResNet-50** extracts multi-layer visual features from uploaded ads
2. **Content Signal Detectors** compute face, scene, text, color, contrast, reward, and spatial signals
3. **Brain Region Probes** (derived from BOLD5000 + img2fmri methodology) map signals to predicted fMRI activations across 20 HCP MMP1.0 parcels
4. **AI Interpretation** analyzes activation patterns and provides creative recommendations

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export ANTHROPIC_API_KEY=your_key_here
export PERPLEXITY_API_KEY=your_key_here  # Optional

# Run the server
uvicorn backend:app --reload
```

Open `index.html` in your browser or visit `http://localhost:8000` to use the app.

## Environment Variables

- `ANTHROPIC_API_KEY` - Required for AI analysis
- `PERPLEXITY_API_KEY` - Optional, used as primary AI with Claude as fallback

## Deployment

### Render

1. Create a new Web Service on [Render](https://render.com)
2. Connect your GitHub repo
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `uvicorn backend:app --host 0.0.0.0 --port $PORT`
5. Add environment variables in the Render dashboard

### Railway

1. Create new project on [Railway](https://railway.app)
2. Deploy from GitHub
3. Add environment variables
4. Railway auto-detects Python and deploys

## Tech Stack

- FastAPI backend
- PyTorch for neural network inference
- Anthropic Claude / Perplexity for AI analysis
- Vanilla HTML/CSS/JS frontend
