=========
cansrmapp
=========


|b| 

.. |b| image:: https://app.travis-ci.com/idekerlab/cansrmapp.svg
        :target: https://app.travis-ci.com/idekerlab/cansrmapp

.. .. |a| |b| |c|
   .. |a| image:: https://img.shields.io/pypi/v/cansrmapp.svg
        :target: https://pypi.python.org/pypi/cansrmapp
   .. |b| image:: https://app.travis-ci.com/idekerlab/cansrmapp.svg
        :target: https://app.travis-ci.com/idekerlab/cansrmapp
   .. |c| image:: https://readthedocs.org/projects/cansrmapp/badge/?version=latest
        :target: https://cansrmapp.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

CanSRMaPP is a modeling tool for identifying a minimal feature set describing the
metagenome of a cancer cohort.

* Free software: BSD license
* Source code: https://github.com/idekerlab/cansrmapp

Dependencies
------------

* `Pytorch 2.5+ with torchaudio, torchvision <https://pytorch.org>`__ (tested on 2.5.0)0
* tables
* matplotlib
* numpy
* pandas
* scikit-learn
* scikit-image
* scipy


Compatibility
-------------

* Python 3.11+
* CUDA 12.1 if using GPU. Download the appropriate CUDA toolkit for your system `here <https://developer.nvidia.com/cuda-12-1-1-download-archive>` :

**Note**
   CUDA is only required for implementations using NVIDIA GPUs;
   feel free to ignore otherwise.

   The root CanSRMaPP module automatically detects whether CUDA is set up;
   `cmbuilder` and in particular `cmsolver` will configure themselves to use
   the GPU if available.


Installation
------------

Anaconda environment
~~~~~~~~~~~~~~~~~~~~

.. code-block::

    conda create -n cansrmapp python=3.11 -y
    conda activate cansrmapp

Building and installing cansrmapp package

.. code-block::

   git clone https://github.com/idekerlab/cansrmapp
   cd cansrmapp
   pip install -r requirements_dev.txt
   make dist
   pip install dist/cansrmapp*whl

Usage
----------

Basic usage / code test
~~~~~~~~~~~~~~~~~~~~~~~

To fit CanSRMaPP models, scripts are provided in `demo/`.
A simple test invocation (<5 minutes) is : 

.. code-block:: bash

    cd demo
    ./build.sh
    ./test-solve.sh
    ./polish.sh

`build.sh`
    creates the CanSRMaPP input matrices in ``demo/nest`` (where ``nest`` is the model name).

`test-solve.sh`
    Finds the maximum-posterior solution for the input matrices. In the
    interest of low runtime and debugging, some parameters in `test-solve.sh` have been
    set such that they may not converge on optimal solutions; those in `full-solve.sh`
    **are** set to produce an optimal solution.

`polish.sh` 
    Puts the results in a more interpretable format; work will continue on improving
    presentation. The key files are stored in ``demo/summary`` : 

    `feature_summary.csv`
        contains the Maximum a Posteriori (MAP) estimate of 
        each input feature along with that feature's type (gene, signature, or genomic background),
        and its name.

    `selected_events_boolean.csv`
        contains true/false values for a simple selection test on
        each alteration type (column) and each gene (row).

To reproduce the core CanSRMaPP workflow (~30 minutes): 

.. code-block:: bash

    cd demo
    ./build.sh
    ./full-solve.sh
    ./polish.sh
    ./validate.sh

Output for the **final** command should resemble : 

.. code-block:: 

    Feature weight agreement with publication (pearson)
    PearsonRResult(statistic=0.9999972289807557, pvalue=0.0)
    Feature identification agreement with publication (jaccard,differences)
              Local run           |         Publication          
    -------------------------------------------------------------
           only        |       common       |       only        
             0         |         122        |         0         

    ============================================================
    Detected GPU.
    TCGA-LUAD [training] frequency agreement (pearson) :
    PearsonRResult(statistic=0.9750266, pvalue=0.0)
    TCGA-CPTAC [evaluation] frequency agreement (pearson) :
    PearsonRResult(statistic=0.89953285, pvalue=0.0)

Indicating that the 122 CanSRMaPP features are those recovered by the authors,
and that their deviation from the authors' values is less than one part in
10\ :sup:`5`. 

**Note**
  Anecdotally, you can expect a single cycle of `cmsolver` to take
  about 70 seconds on a GPU and up to 20 minutes when parallelized
  over multiple CPUs; GPU runtime may be slower on WSL. ``test-solve.sh``
  runs for one cycle, while ``full-solve.sh`` runs for twenty.

  Parallelization largely takes place from
  backends handled by `numpy`, `scipy`, and `pytorch`, so if
  you wish to limit parallelization, follow procedures
  relevant to those modules for setting environment variables.

==========================
Redistributed data sources
==========================

CanSRMaPP relies on a number of third-party files for reference and reconciling
multiple data sources. This document describes the provenance of all such files,
and hosts frozen copies since some may be updated in-place by the maintainers.

Cancer Genomic Data
-------------------

Cancer genomic data was downloaded from the `Genomic Data Commons`_ on
February 2, 2024. Because this data is subject to varying degrees of
controlled access, it cannot be redistributed here in its original form.
Binarized alteration states and signature activities, which constitute
a de-identified data derivative under the NIH universal Data Use Certification,
are hosted here and on `zenodo`. Gene level alteration states for the
TCGA LUAD cohort are located in ``data/tcga_[cohort]/omics_full.csv.gz``;
for the CPTAC LUAD cohort, ``data/omics_cptac_luad.csv.gz``.
Signature activities for the TCGA LUAD cohort are in ``data/tcga_[cohort]/signatures.csv.gz``.

.. _zenodo: https://doi.org/10.5281/zenodo.17995310.
.. _Genomic Data Commons: https://gdc.cancer.gov/ 


Gene Info
---------

``Homo_sapiens.gene_info`` was downloaded from
`<https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz>`_ on
November 3, 2024. This file is unrestricted as described `here <https://ftp.ncbi.nlm.nih.gov/README.ftp>`_

GRCh38 genomic annotation
-------------------------

``GCF_000001405.40_GRCh38.p14_genomic.gff.gz`` was downloaded from `this FTP directory`_ on November 12, 2024.
This file is unrestricted as described `according to these terms`_
The reduced file `gff_reduced.gff.gz` derived from this one is the result of running the command  ::

        gunzip -c GCF_000001405.40_GRCh38.p14_genomic.gff.gz | awk -F'     ' '$0 !~ /^#/ && $3 == "gene" && $9 ~/GeneID/ ' | gzip -c > gff_reduced.gff.gz

.. _this ftp directory: https://ftp.ncbi.nlm.nih.gov/genomes/refseq/vertebrate_mammalian/Homo_sapiens/latest_assembly_versions/GCF_000001405.40_GRCh38.p14/
.. _according to these terms: https://ftp.ncbi.nlm.nih.gov/README.ftp

NeSTv0
------

"NeSTv0" is a precursor of the interaction map found in
`Zheng, Kelly, et al., 2021`_, prior to filtering for mutation-enriched modules.
It is distributed here as ``nest.pickle`` with permission from the authors, and is
subject to the license governing this repository. The file contains a `dict` object
mapping each module to a `set` of member gene Entrez IDs. Because module in this
file are named ``Clusterx-y``, an additional file, ``NeST_map_1.5_default_node_Nov20.csv``,
is incorporated to map these to their NEST IDs as published.

.. _Zheng, Kelly, et al., 2021: https://doi.org/10.1126/science.abf3067

reactome
--------

`NCBI2Reactome_All_Levels.txt` was downloaded from https://reactome.org/download/current/
on November 3, 2024, and refactored into the file `module_maps/reactome.pickle`

Gene Ontology (GO)
------------------

The OBO file `go_basic.obo` was downloaded on September 28, 2024 from 
https://geneontology.org/docs/download-ontology/go_basic.obo, distributed
under CC-BY 4.0. The file `gene2go_human` is a modified version of a file 
downloaded March 12, 2024 from https://ftp.ncbi.nlm.nih.gov/gene/DATA/, 
with annotations from nonhuman taxa removed; no restrictions have been placed 
by NCBI on these files' use. They were in turn used to generate 
`module_maps/go_bpcc.pickle`, which omits the "molecular function" namespace.
**NOTE** that the given link is updated **DAILY**, and discrepancies between
this file and the one at the link are 
extremely likely.

CORUM
-----

During development, Helmholtz-Munich was hit by a cyberattack and CORUM could 
not longer be hosted digitally.  Source data for the included file
`module_maps/corum.pickle` was graciously communicated by Andreas Ruepp.
This release is described in `Tsitsiridis et al, 2022`_ , 
and is distributed under CC-BY-SA 4.0.

.. _Tsitsiridis et al, 2022: https://doi.org/10.1093/nar/gkac1015, 


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

