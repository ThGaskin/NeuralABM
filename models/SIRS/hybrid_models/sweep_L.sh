#!/bin/bash
for seed in {0..9}; do
  for L in 20 30 40 50 60 70 80; do
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
          --job-name="${key}__L_${L}__rec_${recursive}__seed_${seed}" \
          --wrap="python sweep.py --key ${key} --n 1 --L ${L} ${rec_flag} --training_data 'data/SIRS/hybrid_training_data.pt' --seed ${seed} --out_dir='~/SIRS_sweep/'"
      done
    done
  done
done