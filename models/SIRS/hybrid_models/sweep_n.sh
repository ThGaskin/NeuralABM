#!/bin/bash
for seed in {0..9}; do
  for n in 2 4 6 8 10 12 14; do
    for key in params hybrid_1 hybrid_2 bb; do
      for recursive in true false; do
        if [ "$key" = "params" ] && [ "$recursive" = true ]; then
          continue
        fi
        if [ "$recursive" = true ]; then
          rec_flag="--recursive"
        else
          rec_flag=""
        fi
        sbatch \
          -p <partition_name> \
          -N 1 \
          --ntasks=1 \
          --output=logs/slurm-%A.out \
          --job-name="${key}__n_${n}__rec_${recursive}__seed_${seed}" \
          --wrap="python sweep.py --key ${key} --n ${n} --L 101 ${rec_flag} --training_data 'data/SIRS/hybrid_training_data.pt' --seed ${seed} --out_dir='~/SIRS_sweep/'"
      done
    done
  done
done