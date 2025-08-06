# Market Scout 📈

A powerful tool for discovering market patterns and relationships using technical analysis and AI insights.

## Deployment Instructions

### Option 1: Streamlit Cloud (Recommended)

1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Add your environment variables:
   - `ANTHROPIC_API_KEY`: Your Anthropic API key for AI analysis

### Option 2: Railway

1. Visit [Railway.app](https://railway.app)
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Configure environment variables:
   - `ANTHROPIC_API_KEY`: Your Anthropic API key
5. Railway will automatically detect the Python environment and deploy

### Option 3: Render

1. Visit [Render.com](https://render.com)
2. Create a new Web Service
3. Connect your GitHub repository
4. Set the following:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run market-app.py`
5. Add environment variables:
   - `ANTHROPIC_API_KEY`: Your Anthropic API key

## Local Development

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set your environment variables:
   ```bash
   export ANTHROPIC_API_KEY='your-api-key-here'
   ```
5. Run the app:
   ```bash
   streamlit run market-app.py
   ```

## Features

- Market pattern analysis with multiple timeframes
- Interactive charts and visualizations
- AI-powered insights using Claude API
- Preset patterns and custom configurations
- Forward returns analysis
- Risk metrics and benchmarking 