#!/bin/bash
#BSUB -q c02516
#BSUB -J video_training_all
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=20GB]"
#BSUB -M 20GB
#BSUB -W 04:00
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

echo "Starting job $LSB_JOBNAME with Job ID $LSB_JOBID"
source /zhome/a0/c/223983/deep_learning_video_classification/4.1/venv/bin/activate
echo "Activated virtual environment."
echo "Running complete workflow script..."
source /zhome/a0/c/223983/deep_learning_video_classification/4.1/run_complete_workflow.sh
