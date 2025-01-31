#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=medium
#SBATCH --time=00:72:00 # 72 hours
#SBATCH --job-name=triggerfish_detection_YOLO_training
#SBATCH --mail-type=START,BEGIN,END
#SBATCH --mail-user=caitlin.newport@biology.ox.ac.uk


# Build the environment
module load CUDA/11.8
# Activate the Conda environment
module load Anaconda3
source activate base
conda activate $DATA/venvs/torch_env

# Move files to the scratch folder
mkdir -p /scratch/triggerfish-detection
cp -R $DATA/triggerfish-detection/* /scratch/triggerfish-detection/

# Change into the scratch directory
cd /scratch/triggerfish-detection

# show start time
echo "Starting at $(date)"

# Run the training script
python YOLO_model_training.py \
    --project_name "triggerfish-detection" \
    --run_name "yolov8-training-run-1" \
    --data_yaml "data.yaml" \
    --train_path "datasets/dataset_v1/images/train" \
    --val_path "datasets/dataset_v1/images/val" \
    --model_size "n" \
    --epochs 1 \
    --batch_size 4 \
    --image_size 640

# show end time
echo "Ending at $(date)"

# Move files back to the data folder, excluding raw_data and datasets directories
rsync -av --exclude='raw_data' --exclude='datasets' /scratch/triggerfish-detection/* $DATA/triggerfish-detection/

# Move files from the ARC/HTC to your local machine:
# IT should be a folder called runs/detect/train/
#scp zool2073@htc-login.arc.ox.ac.uk:/data/dtc-schmidt/zool2073/triggerfish_detection/runs/detect/train/ /Users/user/projects/VideoUtilies/runs/detect/train/

echo "All Done!"