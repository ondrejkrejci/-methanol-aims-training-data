#!/bin/bash
#SBATCH --job-name=mace_test # myTest is the job name in the queue
#SBATCH --account=project_2006995 # CSC project name
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100:2
#SBATCH --time=00-00:15:00
#SBATCH -o job.out # job output data will be written here
#SBATCH -e job.err # any errors will be written here

## execute job
module load tykky
export PATH="/projappl/project_2006995/MACE_gpu/bin:$PATH"
python3 run_mace_dyn.py
#python3 run_mace_opt.py

