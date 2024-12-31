# Overview

CanSRMaPP relies on a number of third-party files for reference and reconciling
multiple data sources. This document describes the provenance of all such files,
and hosts frozen copies since some may be updated in-place by the maintainers.

# Systems Maps

This section describes the provenance of all systems maps used in the 
analysis.  Where possible, links will be given.  Scripts implementing the 
CanSRMaPP algorithm use `pickle` files derived from running 
`scripts/assemble_systems_maps.py`, which reads these files.


## ActivePathways(ActPwy)
Source : [Paczkowska et al, 2020](https://doi.org/10.1038/s41467-019-13983-9) 
During development, ICGC retired its data distribution site. Instructions for 
accessing ICGC legacy data can be found at 
[https://docs.icgc-argo.org/docs/data-access/icgc-25k-data], but we have not 
been able to verify that the relevant data is hosted there.

Assuming that you are able to retreive the file 
`pathway_and_network_method_results.tar.gz`, you can find a file at this 
path: 
`PCAWG_consensus/cds_noncds_ActiveDriverPW_PCAWGintegrated.Pancan-no-skin-melanoma-lymph_noEnh__results_300817.txt`
which is included here. Per 
[https://www.icgc-argo.org/page/77/e3-publication-policy], no restrictions 
are placed on these data.

## CORUM

During development, Helmholtz-Munich was hit by a cyberattack and CORUM could 
not longer be hosted digitally.  The included file `CORUM download 
2022_09_12.xlsx` was graciously  communicated by Andreas Ruepp. This release is 
described in [Tsitsiridis et al, 2022](https://doi.org/10.1093/nar/gkac1015), 
and is distributed under CC-BY-SA 4.0.

##  Gene Ontology (GO)

The OBO file `go_basic.obo` was downloaded on September 28, 2024 from 
[https://geneontology.org/docs/download-ontology/go_basic.obo], distributed 
under CC-BY 4.0. The file `gene2go_human` is a modified version of a file 
downloaded March 12, 2024 from [https://ftp.ncbi.nlm.nih.gov/gene/DATA/], 
with annotations from nonhuman taxa removed; no restrictions have been placed 
by NCBI on this file's use. **NOTE** that the given link is updated 
**DAILY**, and discrepancies between this file and the one at the link are 
extremely likely.

## Kuenzi's Pathway Map Consensus (Kuenzi)

Supplementary Table 3 was downloaded from [Kuenzi and Ideker, 
2020](https://doi.org/10.1038/s41568-020-0240-7), from its publicly available 
form at (https://pmc.ncbi.nlm.nih.gov/articles/PMC7224610/).  Permission has 
been granted from the authors to redistribute this in the file 
`NIHMS1571677-supplement-Supp_Table_3.xlsx`.

## NeSTv0

"NeSTv0" is a hitherto unpublished precursor of the interaction map found in
[Zheng, Kelly, et al., 2021](https://doi.org/10.1126/science.abf3067), prior
to filtering for mutation-enriched systems. It is distributed here as
`IAS_clixo_hidef_Nov17.edges` with permission from the authors, and is 
subject to the license governing this repository. Because systems in this
file are named `Clusterx-y`, an additional file, `NeST_map_1.5_default_node_Nov20.csv`,
is incorporated to map these to their NEST IDs as published.

## reactome

`NCBI2Reactome_All_Levels.txt` was downloaded from [https://reactome.org/download/current/]
on July 6, 2023. It is redistributed here under CC0 Creative Commons Public Domain license.

## WEBSTER

WEBSTER [Pan et al, 2022](https://doi.org/10.1016/j.cels.2021.12.005)system 
assignments cannot be redistributed under current licenses. To access WEBSTER 
for use in CanSRMaPP, download Supplementary Table S3 from the link and save
it to this folder on your local machine.

## WikiPathways

GPML files from WikiPathways were downloaded on September 27, 2024 and are included in the
directory `wikipathways_gpml`. 
They are redistributed here under the Creative Commons CC0 waiver.

## Wainberg-Kamber coessential modules (WK)

Coessential modules from [Wainberg, Kamber, et al](https://doi.org/10.1038/s41588-021-00840-z)
are cannot be redistributed under current licenses. To access the coessential modules
for use in CanSRMapp, download Supplementary Data 2 from the article at the link
and save it to this folder on your local machine.




