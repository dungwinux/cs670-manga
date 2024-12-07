#!/bin/bash -l

#SBATCH --job-name=inpainting
#SBATCH -o /work/pi_miyyer_umass_edu/ctpham/cs670-manga/.logs/%x-%A-%a.out
#SBATCH --partition=gypsum-rtx8000
#SBATCH --mem=100G
#SBATCH --time=0-24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ctpham@umass.edu


module load conda/latest
conda activate /scratch/workspace/ctpham_umass_edu-llama/envs/inpainting/

python3 model_torch.py