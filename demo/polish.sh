#!/bin/bash

OUTDIR="summary"

mkdir -p ${OUTDIR}

python3 << fin
import cansrmapp as cm
import cansrmapp.cmpresentation as cmp
import numpy as np
cma=cmp.loadcma('nest','model')
fsf=cma.feature_summary_frame()
dfmap=fsf.query('map_estimate > 0')
dfmap.to_csv("${OUTDIR}/feature_summary.csv")
dfus=cma.underselection(cutoff=np.log(4))
dfus.index.name='Gene'
dfus.columns.name='Alteration Type'
dfus.to_csv("${OUTDIR}/selected_events_boolean.csv")
fin

