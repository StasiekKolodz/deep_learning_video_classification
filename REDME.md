# Video Classification Project

## Setup Instructions

I created `requirements-cpu.txt` file so we can all setup exact same environment on HPC.

You can use `setup_env_cuda.sh` script to automatically create python venv and install packages from requirements file.

```bash
chmod +x setup_env_cuda.sh
./setup_env_cuda.sh
```

---

## How to Run Training

In 4.1 directory there are 3 python scripts used for training, evaluation, and creating some plots. (All parameters are set as command-line arguments because it's required for GPU jobs)

### Python Scripts

- `4_1_training.py` - runs training
- `4_1_evaluation.py` - runs evaluation
- `visualise_results.py` - creates some more plots

### Automation Scripts

I created bash scripts to automate whole process and run it in GPU job.

- `run_complete_workflow.sh` - runs training, evaluation and plotting for all models (task 4.1)
- `run_complete_workflow_no_leakage.sh` - does the same thing but on no leakage dataset (task 4.2.1)

### Job Scripts

If you want to run training as jobs you can use job scripts I created but probably paths have to be adjusted there.

Results

Results for 4.1 are in this markdown file results.md (click)

Results for 4.2.1 are in this markdown file results_no_leakage.md (click)


## Results

- **Task 4.1 Results:** See [results.md](results.md)
- **Task 4.2.1 Results (No Data Leakage):** See [results_no_leakage.md](results_no_leakage.md)
