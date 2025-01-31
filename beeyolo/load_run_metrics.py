import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
from ultralytics import YOLO

def load_run_metrics(run_dir):
    """Load metrics from a training run directory"""
    results_file = Path(run_dir) / "results.csv"
    if not results_file.exists():
        raise FileNotFoundError(f"No results.csv found in {run_dir}")
    return pd.read_csv(results_file)

def compare_models(run_dirs, run_names):
    """Compare metrics across different model runs"""
    metrics_dfs = []
    for run_dir, name in zip(run_dirs, run_names):
        try:
            df = load_run_metrics(run_dir)
            df['model'] = name
            metrics_dfs.append(df)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue

    if not metrics_dfs:
        raise ValueError("No valid results found to compare")

    combined_df = pd.concat(metrics_dfs)

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison')

    # Plot mAP50
    sns.lineplot(data=combined_df, x='epoch', y='metrics/mAP50(B)',
                hue='model', ax=axes[0,0])
    axes[0,0].set_title('mAP50 vs Epoch')
    axes[0,0].set_ylabel('mAP50')

    # Plot mAP50-95
    sns.lineplot(data=combined_df, x='epoch', y='metrics/mAP50-95(B)',
                hue='model', ax=axes[0,1])
    axes[0,1].set_title('mAP50-95 vs Epoch')
    axes[0,1].set_ylabel('mAP50-95')

    # Plot training loss
    sns.lineplot(data=combined_df, x='epoch', y='train/box_loss',
                hue='model', ax=axes[1,0])
    axes[1,0].set_title('Training Box Loss vs Epoch')
    axes[1,0].set_ylabel('Box Loss')

    # Plot validation loss
    sns.lineplot(data=combined_df, x='epoch', y='val/box_loss',
                hue='model', ax=axes[1,1])
    axes[1,1].set_title('Validation Box Loss vs Epoch')
    axes[1,1].set_ylabel('Box Loss')

    plt.tight_layout()
    return combined_df

def compare_model_predictions(model_paths, test_image):
    """
    Compare predictions from multiple models on a single test image
    
    Args:
        model_paths (list): List of paths to model weights
        test_image (str): Path to test image
    """
    # Create a subplot for each model plus the original image
    n_models = len(model_paths)
    fig, axes = plt.subplots(1, n_models + 1, figsize=(5 * (n_models + 1), 5))
    
    # Plot original image
    img = plt.imread(test_image)
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot predictions from each model
    for idx, model_path in enumerate(model_paths):
        try:
            # Load model and make prediction
            model = YOLO(model_path)
            results = model.predict(test_image, conf=0.25)
            
            # Plot result
            result_plot = results[0].plot()
            axes[idx + 1].imshow(result_plot)
            axes[idx + 1].set_title(f'Model {idx + 1}')
            axes[idx + 1].axis('off')
            
        except Exception as e:
            print(f"Error with model {idx + 1}: {e}")
            axes[idx + 1].text(0.5, 0.5, f'Error: {str(e)}', 
                             ha='center', va='center')
            axes[idx + 1].axis('off')
    
    plt.tight_layout()
    plt.show()