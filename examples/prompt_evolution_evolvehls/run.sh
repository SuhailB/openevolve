srun -c16 --mem=32GB python /home/suhailb/Projects/EvolveHLS/extern/openevolve/openevolve-run.py \
  ./initial_prompt.md \
  ./evaluator.py \
  --config ./config.yaml \
  --iterations 1 &> out.log