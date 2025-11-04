# Anemia Detection Project

This project implements an AI-based system for anemia detection using deep learning models (RegNet and U-Net).

## Project Structure

```
├── app/                          # Applications
│   └── webui/                    # Gradio web interface
├── data/                         # Data directory
│   └── raw/                      # Raw data including sample images
├── models/                       # Model files
│   └── trained_models/          # Trained model weights
├── src/                         # Source code
│   ├── anemia_explainer/       # Anemia detection and explanation module
│   └── models/                 # Model architecture definitions
└── config/                      # Configuration files
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the web interface:
```bash
python app/webui/app.py
```

## Models

- RegNet: Anemia classification model
- U-Net: Image segmentation model

## License

[Your license information here]