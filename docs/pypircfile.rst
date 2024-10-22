Expected configuration for .pypirc file
============================================

To run ``make testrelease`` or ``make release`` to deploy cansrmapp to TestPyPi_ & PyPi_ you must have credentials set up.

#. If not already, create account on PyPi_
#. If not already, create account on TestPyPi_
#. Create a configuration file in your home directory ``~/.pypirc`` and fill it with the following information replacing **<XXX>**
   values with correct information:

.. code-block::

    [distutils]
    index-servers=
    pypi
    testpypi

    [testpypi]
    repository:https://test.pypi.org/legacy/
    username = __token__
    password = <API TOKEN FROM PYPI>

    [pypi]
    username = __token__
    password = <API TOKEN FROM PYPI>

.. _TestPyPi: https://test.pypi.org
.. _PyPi: https://pypi.org
