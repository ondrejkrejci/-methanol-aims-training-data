#!/bin/bash

## execute job
module load tykky
export PATH="/projappl/project_2006995/MACE_gpu/bin:$PATH"
#python3 run_mace_dyn.py
python3 run_mace_opt.py

