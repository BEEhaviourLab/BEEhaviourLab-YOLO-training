# BeeYOLO: Bee Detection using YOLOv8

Authors: Rachel Parkinson & Cait Newport

## Overview
This repository contains code for training and evaluating YOLOv8 models for bee detection in video footage. The project includes tools for:
- Training YOLOv8 models on bee detection datasets
- Evaluating model performance
- Processing videos for bee detection
- Comparing different model configurations

## Project Structure

beeYOLO/
├── beeyolo/ # Main package
│ ├── load_run_metrics.py # Model evaluation tools
│ └── video_inference.py # Video processing utilities
├── notebooks/ # Jupyter notebooks for training and analysis
├── datasets/ # Training and validation data
│ ├── train/
│ └── val/
├── runs/ # Model checkpoints and training logs
└── requirements.txt # Project dependencies


## Installation

### 1. Set up Virtual Environment

Create a new virtual environment
On Mac/Linux
``` bash
python -m venv yolo_training_enviro
```



On Windows
``` bash
python -m venv yolo_training_enviro
```

### 2. Activate Virtual Environment

On Linux/Mac
``` bash
source yolo_training_enviro/bin/activate
```

On Windows
``` bash
yolo_training_enviro\Scripts\activate
```

### 3. Install Package

Install the package in development mode
``` bash
pip install -e .
```

## Weights & Biases (wandb) Setup

This project uses Weights & Biases (wandb) for experiment tracking and model monitoring. While wandb is optional, it provides valuable insights into training progress and model performance.

### 1. Create a wandb Account

1. Visit [https://wandb.ai/site](https://wandb.ai/site) to create a free account
2. Sign up with your email address
3. Verify your email and complete the setup process

### 2. Get Your API Key

1. Log in to your wandb account
2. Go to your profile settings (click your profile picture → Settings)
3. Navigate to the "API Keys" section
4. Copy your API key (it will look like: `abc123def456ghi789...`)

### 3. Configure Environment Variables

Create a `.env` file in the root directory of this project:

```bash
# Create .env file in the project root
echo "WANDB_API_KEY=your_api_key_here" > .env
```

**Important**: Replace `your_api_key_here` with your actual wandb API key.

**Note**: The `.env` file should be in the same directory as this README.md file.

### 4. Alternative: Set Environment Variable Directly

If you prefer not to use a `.env` file, you can set the environment variable directly in your terminal:

**On Mac/Linux:**
```bash
export WANDB_API_KEY="your_api_key_here"
```

**On Windows:**
```bash
set WANDB_API_KEY=your_api_key_here
```

### 5. Disable wandb (Optional)

If you don't want to use wandb for tracking, you can disable it by adding the `--disable_wandb` flag when running the training script:

```bash
python beeyolo/YOLO_model_training.py \
--project_name "insect-detection" \
--run_name "yolov8n-full-training" \
--data_yaml "../datasets/insect-data/data.yaml" \
--train_path "../datasets/insect-data/train/images" \
--val_path "../datasets/insect-data/val/images" \
--disable_wandb
```

## Usage

### Training Models
Use the training notebook in `notebooks/02_YOLO_model_training.ipynb` to:
- Configure training parameters
- Train YOLOv8 models
- Monitor training progress

### Evaluating Models
The evaluation tools in `beeyolo.load_run_metrics` allow you to:
- Compare model performance
- Visualize training metrics
- Analyze detection results

### Processing Videos
Use `beeyolo.video_inference` to:
- Process videos with trained models
- Save detection results
- Generate annotated videos

## Dependencies
- ultralytics
- pandas
- matplotlib
- seaborn
- opencv-python
- wandb (optional, for experiment tracking)

## Contributing
Feel free to open issues or submit pull requests for improvements.

## License
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License