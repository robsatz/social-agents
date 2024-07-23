# NMA Macrocircuits

Modelling the social behaviors of multiple worms with and without architectural pirors

# Setup

- The scripts works in Python==3.10. Other packages can be install via `packages.txt`. 

- Clone this repository
- Within the project root folder, create a folder called 'lib'
- Download tonic library:

```
cd lib
git clone https://github.com/fabiopardo/tonic.git
```

- Create conda environment:

```
cd ..
conda env create -f environment.yaml
conda activate social-agents
```

- Install the tonic dependency and the social_agents package from this repo into your conda environment:
```
python setup.py install
```
