# NMA Macrocircuits

Modelling the social behaviors of multiple worms with and without architectural pirors

## Setup

- Clone this repository
- Navigate to root folder
- Create conda environment:

```
conda env create -f environment.yaml
conda activate social-agents
```
The environment setup will automatically 
install the `tonic` and `social_agents` packages as separate
editable pip installs in your `social-agents` conda environment (with python=3.10.14). 
This means you can dynamically develop each package without having
to 'refresh' any imports.