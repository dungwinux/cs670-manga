#!/bin/bash -l
#SBATCH -J cv_7
#SBATCH --partition=gpu-preempt
#SBATCH --ntasks=1                # Total of 8 tasks (1 task per GPU)
#SBATCH --gres=gpu:1              # Request 1 GPU per task (distributed across nodes)
#SBATCH --mem=400GB
#SBATCH --constraint=a100-80g
#SBATCH --time=0-24:00:00
#SBATCH --output=/work/pi_miyyer_umass_edu/ctpham/cs670-manga/.logs/%x_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ctpham@cs.umass.edu

module load conda/latest
conda activate /scratch/workspace/ctpham_umass_edu-llama/envs/manga
python3 qwen.py