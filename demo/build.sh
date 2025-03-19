#!/bin/zsh
python3 ../cansrmapp/cmbuilder.py \
--omics_path ../data/omics.csv.gz \
--signature_path ../data/signatures.csv.gz \
--sm_path ../systems_maps/nest.pickle \
--blacklist_path ../data/lowly_expressed_blacklist.pickle \
--length_timing_path ../data/length_and_timing.hdf \
--output_path nest \
--signature_sparsity 0.05 \
--critical_synteny_quantile 0.8 \
--no_arm_pcs  \
--spoof_seed orig \
--spoof_smsize 2296
                                     
