#!/bin/bash
#SBATCH -J gemm
#SBATCH -o %x.log # <JOB_NAME>.<JOB_ID>.log
#SBATCH -p batch -c60 --mem=64GB -A suhailb

work_dir=$(pwd)
scratch_dir=/scratch/$USER/${SLURM_JOB_NAME}.${SLURM_JOB_ID}

mkdir $scratch_dir && cd $scratch_dir

rsync -rav --exclude='*.log' --exclude='output_*' --exclude='tmp' $work_dir/ $scratch_dir/

eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

pyenv deactivate
pyenv activate evolve-hls

ml load xilinx/vivado/2023.2

python /home/suhailb/Projects/EvolveHLS/extern/openevolve/openevolve-run.py \
  ./initial_prompt.md \
  ./evaluator.py \
  --config ./config.yaml \
  --iterations 10

rsync -rav $scratch_dir/ $work_dir/

