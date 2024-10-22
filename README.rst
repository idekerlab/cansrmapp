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




Mark please fill this out


* Free software: BSD license
* Documentation: https://cansrmapp.readthedocs.io.
* Source code: https://github.com/idekerlab/cansrmapp



Dependencies
------------

* `cellmaps_utils <https://pypi.org/project/cellmaps-utils>`__

Compatibility
-------------

* Python 3.8+

Installation
------------

.. code-block::

   git clone https://github.com/idekerlab/cansrmapp
   cd cansrmapp
   make dist
   pip install dist/cansrmapp*whl


Run **make** command with no arguments to see other build/deploy options including creation of Docker image 

.. code-block::

   make

Output:

.. code-block::

   clean                remove all build, test, coverage and Python artifacts
   clean-build          remove build artifacts
   clean-pyc            remove Python file artifacts
   clean-test           remove test and coverage artifacts
   lint                 check style with flake8
   test                 run tests quickly with the default Python
   test-all             run tests on every Python version with tox
   coverage             check code coverage quickly with the default Python
   docs                 generate Sphinx HTML documentation, including API docs
   servedocs            compile the docs watching for changes
   testrelease          package and upload a TEST release
   release              package and upload a release
   dist                 builds source and wheel package
   install              install the package to the active Python's site-packages
   dockerbuild          build docker image and store in local repository
   dockerpush           push image to dockerhub

For developers
-------------------------------------------

To deploy development versions of this package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Below are steps to make changes to this code base, deploy, and then run
against those changes.

#. Make changes

   Modify code in this repo as desired

#. Build and deploy

.. code-block::

    # From base directory of this repo cansrmapp
    pip uninstall cansrmapp -y ; make clean dist; pip install dist/cansrmapp*whl



Needed files
------------

**TODO:** Add description of needed files


Usage
-----

For information invoke :code:`cansrmappcmd.py -h`

**Example usage**

**TODO:** Add information about example usage

.. code-block::

   cansrmappcmd.py # TODO Add other needed arguments here


Via Docker
~~~~~~~~~~~~~~~~~~~~~~

**Example usage**

**TODO:** Add information about example usage


.. code-block::

   Coming soon ...

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
.. _NDEx: http://www.ndexbio.org
