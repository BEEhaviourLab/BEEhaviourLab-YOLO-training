# YOLO Model Benchmarking

This directory contains scripts for benchmarking YOLO models on different datasets to assess their performance.

## Files

- `benchmark_YOLO.py` - Main benchmarking script with comprehensive functionality
- `example_benchmark.py` - Example script showing how to use the benchmark programmatically
- `README_benchmark.md` - This documentation file

## Features

The benchmark script provides:

- **Flexible data selection**: Choose between validation-only (`val`) or all data (`all`) modes
- **Comprehensive metrics**: Precision, recall, F1-score, and per-class performance
- **Multiple model support**: Test different YOLO models on the same dataset
- **Automatic dataset detection**: Automatically finds available models and datasets
- **Results export**: Save results to JSON files for further analysis
- **Command-line interface**: Easy-to-use CLI for quick benchmarking

## Data Selection Modes

### Validation Mode (`val`)
- Uses only the validation data from `datasets/{dataset}/val/`
- Suitable for standard evaluation when you want to test on unseen data

### All Data Mode (`all`)
- Combines both training and validation data from `datasets/{dataset}/train/` and `datasets/{dataset}/val/`
- Useful when the model wasn't trained on any of the data and you want to test on the entire dataset

## Usage

### Command Line Interface

#### List available models and datasets:
```bash
python benchmark_YOLO.py --list-models
python benchmark_YOLO.py --list-datasets
```

#### Basic benchmarking:
```bash
# Benchmark on validation data only
python benchmark_YOLO.py --model beeYOLO --dataset hoverfly-data --mode val

# Benchmark on all data (train + validation)
python benchmark_YOLO.py --model beeYOLO --dataset hoverfly-data --mode all

# Custom confidence and IoU thresholds
python benchmark_YOLO.py --model beeYOLO --dataset hoverfly-data --mode val --conf-threshold 0.3 --iou-threshold 0.4

# Save results to specific file
python benchmark_YOLO.py --model beeYOLO --dataset hoverfly-data --mode val --output my_results.json
```

### Programmatic Usage

```python
from benchmark_YOLO import YOLOBenchmark

# Initialize benchmark
benchmark = YOLOBenchmark()

# Run benchmark on validation data
results_val = benchmark.run_benchmark(
    model_name="beeYOLO",
    dataset_name="hoverfly-data",
    mode="val",
    conf_threshold=0.5,
    iou_threshold=0.5
)

# Run benchmark on all data
results_all = benchmark.run_benchmark(
    model_name="beeYOLO",
    dataset_name="hoverfly-data",
    mode="all",
    conf_threshold=0.5,
    iou_threshold=0.5
)

# Print results
benchmark.print_results(results_val)
benchmark.print_results(results_all)

# Generate summary report
benchmark.generate_summary_report()

# Save results
benchmark.save_results("my_benchmark_results.json")
```

## Example Script

Run the example script to see the benchmark in action:

```bash
python example_benchmark.py
```

This will:
1. List available models and datasets
2. Benchmark a model on validation data
3. Benchmark the same model on all data
4. Compare multiple models on the same dataset
5. Generate a summary report
6. Save results to a JSON file

## Output

The benchmark provides:

- **Overall metrics**: Precision, recall, F1-score across all classes
- **Per-class metrics**: Individual performance for each class
- **Detection counts**: True positives, false positives, false negatives
- **Dataset statistics**: Total images, ground truth annotations, detections
- **Configuration**: Confidence and IoU thresholds used

## Requirements

Make sure you have the following dependencies installed:

```bash
pip install -r requirements.txt
```

Required packages:
- `ultralytics` - YOLO model loading and inference
- `opencv-python` - Image processing
- `numpy` - Numerical operations
- `tqdm` - Progress bars
- `pyyaml` - YAML file parsing
- `matplotlib` and `seaborn` - Visualization (optional)

## Dataset Structure

Your datasets should follow this structure:
```
datasets/
├── dataset-name/
│   ├── data.yaml          # Dataset configuration
│   ├── train/
│   │   ├── images/        # Training images
│   │   └── labels/        # Training labels (YOLO format)
│   └── val/
│       ├── images/        # Validation images
│       └── labels/        # Validation labels (YOLO format)
```

The `data.yaml` file should contain:
- `nc`: Number of classes
- `names`: List of class names
- `train`: Path to training images
- `val`: Path to validation images

## Model Files

Place your YOLO model files (`.pt` format) in the `models/` directory. The script will automatically detect and list available models.

## Tips

1. **Start with validation mode** to get a quick assessment of model performance
2. **Use all-data mode** when you want to test on the complete dataset
3. **Adjust confidence thresholds** based on your application requirements
4. **Lower IoU thresholds** for more lenient detection matching
5. **Save results** to JSON files for later analysis and comparison
6. **Use the summary report** to quickly compare multiple model-dataset combinations

## Troubleshooting

- **No models found**: Ensure your `.pt` files are in the `models/` directory
- **No datasets found**: Check that your dataset folders contain `data.yaml` files
- **Import errors**: Install required dependencies with `pip install -r requirements.txt`
- **Memory issues**: Process datasets in smaller batches or use GPU acceleration
