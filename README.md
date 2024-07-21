# NMA Macrocircuits

Modelling the social behaviors of multiple worms with and without architectural pirors

# Setup

- The scripts works in Python==3.10. Other packages can be install via `packages.txt`. 

## Tonic initialisation

This project requires [tonic](https://github.com/neuromatch/tonic/tree/master) as wrappers to achieve better modularity and readabililty of diverse RL agents and tasks. To install, first clone the library into the root project:

```git clone https://github.com/fabiopardo/tonic.git```

Then navigate to the setup.py file, and modified into the following:

```
import setuptools
from setuptools import find_packages


setuptools.setup(
    name='tonic',
    description='Tonic RL Library',
    url='https://github.com/fabiopardo/tonic',
    version='0.3.0',
    author='Fabio Pardo',
    author_email='f.pardo@imperial.ac.uk',
    install_requires=[
        'gym', 'matplotlib', 'numpy', 'pandas', 'pyyaml', 'termcolor'],
    license='MIT',
    python_requires='>=3.6',
    keywords=['tonic', 'deep learning', 'reinforcement learning'],
    packages=find_packages(include=['tonic']) # the line being added
    )

```
Then install tonic properly using

```
pip install -e tonic/
```