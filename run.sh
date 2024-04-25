#!/bin/bash
#SBATCH --partition=SCSEGPU_M2
#SBATCH --qos=q_dmsai
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --job-name=face
#SBATCH --output=output/output_%j.out
#SBATCH --error=output/error_%j.err

module load anaconda3/23.5.2
eval "$(conda shell.bash hook)"
conda activate face_resolution
cd Face-Super-Resolution

python train.py \
--gpu_ids 0 \
--batch_size 8
