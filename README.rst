=========
cansrmapp
=========


|a| |b| |c|

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

see `requirements.txt`


Compatibility
-------------

* Python 3.11+
* CUDA 12.1 _only_ 

.. note::
   CUDA is only required for implementations using GPUs;
   feel free to ignore if not using GPU.

   The root CanSRMaPP module automatically detects whether CUDA is set up;
   `cmbuilder` and in particular `cmsolver` will configure themselves to use
   the GPU if available.

Installation
------------

.. code-block::

   git clone https://github.com/idekerlab/cansrmapp
   cd cansrmapp
   make dist
   pip install dist/cansrmapp*whl


Usage
----------

Basic usage / code test
~~~~~~~

To fit CanSRMaPP models, two scripts are provided in `demo/`; the simplest invocation is
.. code-block::

    cd demo
    ./build.sh
    ./test-solve.sh

`build.sh` creates the CanSRMaPP input matrices; `test-solve.sh` solves them. In the
interest of low runtime and debugging, some parameters in `test-solve.sh` have been
set such that they may not converge on optimal solutions; those in `full-solve.sh`
are set to produce an optimal solution.

.. note::
   Anecdotally, you can expect a single cycle of `cmsolver` to take
   about 1 minute on a GPU and up to 20 minutes when parallelized
   over multiple CPUs. Parallelization largely takes place from
   backends handled by `numpy`, `scipy`, and `pytorch`, so if 
   you wish to limit parallelization, follow their advice for 
   setting environment variables.


=======
Redistributed data sources
=======

CanSRMaPP relies on a number of third-party files for reference and reconciling
multiple data sources. This document describes the provenance of all such files,
and hosts frozen copies since some may be updated in-place by the maintainers.

NCBI Files
-----------

Gene Info
~~~~~~~~~~~
``Homo_sapiens.gene_info`` was downloaded from
`<https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz>`_ on
November 3, 2024. This file is unrestricted as described `here`_

.. _here: https://ftp.ncbi.nlm.nih.gov/README.ftp>

Genbank Flat File
~~~~~~~~~~~
``GCF_000001405.40_GRCh38.p14_genomic.gff.gz`` was downloaded from `this FTP directory`_ on November 12, 2024.
This file is unrestricted as described `according to these terms`_
The reduced file `gff_reduced.gff.gz` derived from this one is the result of running the command  ::
        gunzip -c GCF_000001405.40_GRCh38.p14_genomic.gff.gz | awk -F'     ' '$0 !~ /^#/ && $3 == "gene" && $9 ~/GeneID/ ' | gzip -c > gff_reduced.gff.gz

.. _this ftp directory: https://ftp.ncbi.nlm.nih.gov/genomes/refseq/vertebrate_mammalian/Homo_sapiens/latest_assembly_versions/GCF_000001405.40_GRCh38.p14/
.. _according to these terms: https://ftp.ncbi.nlm.nih.gov/README.ftp


NeSTv0
~~~~~~~~~

"NeSTv0" is a precursor of the interaction map found in
`Zheng, Kelly, et al., 2021`_, prior to filtering for mutation-enriched systems.
It is distributed here as ``nest.pickle`` with permission from the authors, and is 
subject to the license governing this repository. The file contains a `dict` object
mapping each system to a `set` of member gene Entrez IDs. Because systems in this
file are named ``Clusterx-y``, an additional file, ``NeST_map_1.5_default_node_Nov20.csv``,
is incorporated to map these to their NEST IDs as published. 

.. _Zheng, Kelly, et al., 2021: https://doi.org/10.1126/science.abf3067

 


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage


..      Run **make** command with no arguments to see other build/deploy options including creation of Docker image 

..      .. code-block::

..         make

..      Output:

..      .. code-block::

..         clean                remove all build, test, coverage and Python artifacts
..         clean-build          remove build artifacts
..         clean-pyc            remove Python file artifacts
..         clean-test           remove test and coverage artifacts
..         lint                 check style with flake8
..         test                 run tests quickly with the default Python
..         test-all             run tests on every Python version with tox
..         coverage             check code coverage quickly with the default Python
..         docs                 generate Sphinx HTML documentation, including API docs
..         servedocs            compile the docs watching for changes
..         testrelease          package and upload a TEST release
..         release              package and upload a release
..         dist                 builds source and wheel package
..         install              install the package to the active Python's site-packages
..         dockerbuild          build docker image and store in local repository
..         dockerpush           push image to dockerhub


..      For developers
..      -------------------------------------------

..      To deploy development versions of this package
..      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

..      Below are steps to make changes to this code base, deploy, and then run
..      against those changes.

..      #. Make changes

..         Modify code in this repo as desired

..      #. Build and deploy

..      .. code-block::

..          # From base directory of this repo cansrmapp
..          pip uninstall cansrmapp -y ; make clean dist; pip install dist/cansrmapp*whl



..      Needed files
..      ------------

..      **TODO:** Add description of needed files


..      Usage
..      -----

..      For information invoke :code:`cansrmappcmd.py -h`

..      **Example usage**

..      **TODO:** Add information about example usage

..      .. code-block::

..         cansrmappcmd.py # TODO Add other needed arguments here


..      Via Docker
..      ~~~~~~~~~~~~~~~~~~~~~~

..      **Example usage**

..      **TODO:** Add information about example usage


..      .. code-block::

..         Coming soon ...
