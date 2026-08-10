#!/bin/bash

if [[ -a model ]]
then
    echo "cleaned up old model with identical name." 
    rm -rf model
fi

python3 ../cansrmapp/cmsolver.py \
--lambda_selection 2.75  \
--lambda_gb 1.0 \
--alpha_partition 2.5 \
--indir nest \
--outdir model \
--n_cycles 5 \
--n_chains 4
