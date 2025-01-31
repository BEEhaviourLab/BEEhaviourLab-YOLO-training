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
├── dataset/ # Training and validation data
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

## Contributing
Feel free to open issues or submit pull requests for improvements.

## License
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License