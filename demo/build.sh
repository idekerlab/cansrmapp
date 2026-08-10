#!/bin/bash
python3 ../cansrmapp/cmbuilder.py \
--omics_path ../data/tcga_luad/omics_full.csv.gz \
--signature_path ../data/tcga_luad/signatures.csv.gz \
--sm_path ../module_maps/nest.pickle \
--blacklist_path ../data/lowly_expressed_blacklist.pickle \
--length_timing_path ../data/length_and_timing.hdf \
--output_path nest \
--spoof_seed orig \
#--critical_synteny_quantile 0.8 \
#--signature_sparsity 0.05 \
#--spoof_smsize 2296 
