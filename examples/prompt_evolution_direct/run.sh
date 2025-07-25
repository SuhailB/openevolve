srun -c16 --mem=32GB python ../../openevolve-run.py ./initial_prompt.md \
  ./evaluator.py \
  --config ./config.yaml \
  --iterations 100 &> out.log