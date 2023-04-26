#!/bin/bash
#
#$ -cwd
#$ -e /lfs/l1/cta/nieves/image_pars/runner.e$JOB_ID.$TASK_ID
#$ -V
#$ -t 1

conda activate fc_analysis

python3 ff_calibration.py
