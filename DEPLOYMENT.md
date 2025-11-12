# Streamlit Cloud Deployment Guide

## Prerequisites

1. A GitHub account
2. Your code pushed to a GitHub repository
3. A Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))

## Deployment Steps

### 1. Prepare Your Repository

Ensure these files are in your repository:
- ✅ `app.py` - Main Streamlit application
- ✅ `requirements.txt` - Python dependencies
- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ `.gitignore` - Excludes unnecessary files
- ✅ `sta_pruning/` - Your Python package

### 2. Push to GitHub

```bash
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main
```

### 3. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your repository
4. Set the following:
   - **Main file path**: `app.py`
   - **Python version**: 3.11 (or your version)
   - **Branch**: main (or your default branch)
5. Click "Deploy"

### 4. Monitor Deployment

- The app will take 2-5 minutes to deploy
- You can view logs in real-time
- Once deployed, you'll get a public URL like: `https://your-app-name.streamlit.app`

## Configuration

### Streamlit Settings

The `.streamlit/config.toml` file contains:
- Server configuration (headless mode, CORS)
- Theme customization (colors, fonts)

### Python Dependencies

All dependencies are listed in `requirements.txt`:
- streamlit
- numpy, pandas
- scikit-learn, xgboost
- networkx
- matplotlib, plotly

## Troubleshooting

### Common Issues

**1. Import Errors**
- Ensure all dependencies are in `requirements.txt`
- Check Python version compatibility

**2. Memory Issues**
- Streamlit Cloud free tier has 1GB RAM limit
- Consider reducing model size or batch processing limits
- Use `@st.cache_resource` for expensive operations

**3. Slow Loading**
- The initial model training may take time
- Consider pre-training and loading saved models
- Use progress indicators for user feedback

### Performance Optimization

```python
# Already implemented in app.py:
@st.cache_resource
def initialize_system():
    # Cached initialization
    pass
```

## Environment Variables & Secrets

If you need to add secrets (API keys, credentials):

1. In Streamlit Cloud dashboard, go to your app settings
2. Click "Secrets" in the left sidebar
3. Add secrets in TOML format:

```toml
[database]
host = "your-host"
password = "your-password"
```

4. Access in code:
```python
import streamlit as st
db_host = st.secrets["database"]["host"]
```

## Custom Domain (Optional)

Streamlit Cloud allows custom domains on paid plans:
1. Go to app settings
2. Click "Custom domain"
3. Follow DNS configuration instructions

## Monitoring & Analytics

- View app analytics in Streamlit Cloud dashboard
- Monitor resource usage (CPU, memory)
- Check error logs for debugging

## GitHub Repository Setup

### Initialize Git (if not already done)

```bash
git init
git add .
git commit -m "Initial commit: STA Path Pruning System"
```

### Create GitHub Repository

1. Go to [github.com](https://github.com) and create a new repository
2. **Do NOT** initialize with README (you already have one)
3. Copy the repository URL

### Connect and Push

```bash
# Add remote
git remote add origin https://github.com/your-username/sta-pathpruning.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Updating Your App

Any push to your GitHub repository will automatically redeploy:

```bash
git add .
git commit -m "Update feature"
git push origin main
```

The app will r