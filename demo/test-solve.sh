#!/bin/zsh
python3 ../cansrmapp/cmsolver.py \
--lambda_selection 3.0  \
--lambda_gb 1.0 \
--alpha_partition 2.25 \
--indir nest \
--outdir model \
--n_cycles 1 \
--n_chains 1  
