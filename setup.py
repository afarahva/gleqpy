"""Install GLEqPy.

This script (setup.py) will install the GLEPy package.
"""

import os
from setuptools import setup

# Get description from README
root = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(root, 'README.md'), 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='gleqpy',
    version='1.0.0',
    author="Ardavan Farahvash",
    author_email="ardavanf95@gmail.com",
    maintainer="Ardavan Farahvash",
    maintainer_email="ardavanf95@gmail.com",
    description="Tools for generalized Langevin equation simulation and analysis",
    long_description=long_description,
    long_description_content_type='text/markdown',
    license="MIT",
    url="https://github.com/afarahva/gleqpy/",

    install_requires=['numpy','scipy'],
    packages=["md","memory","ase"],
    
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ],
    keywords=[
        "generalized-langevin-equation",
        "langevin",
        "molecular-dynamics",
        "atomic-simulation-environment",
        "python"
    ],
)