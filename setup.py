#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
import os
import re
from setuptools import setup, find_packages
import sys

if sys.version_info < (3,11) :
    print("""
        CanSRMaPP requires Python 3.11
        Isntall Python 3.11 (and set up your environment such that
        /usr/bin/env python points to that version)
        before proceeding.
        """)
    exit(1)


with open(os.path.join('cansrmapp', '__init__.py')) as ver_file:
    for line in ver_file:
        line = line.rstrip()
        if line.startswith('__version__'):
            version = re.sub("'", "", line[line.index("'"):])
        elif line.startswith('__description__'):
            desc = re.sub("'", "", line[line.index("'"):])
        elif line.startswith('__repo_url__'):
            repo_url = re.sub("'", "", line[line.index("'"):])
        elif line.startswith('__author__'):
            author = re.sub("'", "", line[line.index("'"):])
        elif line.startswith('__email__'):
            email = re.sub("'", "", line[line.index("'"):])

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()


requirements = ['torch>=2.5.0,<2.7.0',
                'torchvision>=0.20.0',
                'matplotlib>=3.9.2',
                'numpy>=1.26.4',
                'pandas>=2.2.2',
                'tables>=3.8.0',
                'scikit-image>=0.24.0',
                'scikit-learn>=1.5.2',
                'scipy>=1.13.1']

setup_requirements = ['wheel']

setup(
    author=author,
    author_email=email,
    python_requires='>=3.11.0',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.11'
    ],
    description=desc,
    #requirements=requirements,
    install_requires=requirements,
    license="BSD license",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/x-rst',
    package_data={'':['bioinfo_redistributed/*.*']},
    include_package_data=True,
    keywords='cansrmapp',
    name='cansrmapp',
    packages=find_packages(include=['cansrmapp','cansrmapp.bioinfo_redistributed']),
    package_dir={'cansrmapp': 'cansrmapp'},
    scripts=[],
    setup_requires=setup_requirements,

    url=repo_url,
    version=version,
    zip_safe=False)
