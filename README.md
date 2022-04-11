# Network diffusion examples

Chosen examples for network_diffusion package

## How to use

This code can be executed using CodeOcean capsule (step 1a), that suppress 
requirement of the runtime environment preparation from the user. Otherwise, 
the configuration of the workspace configuration will be essential (step 1b). 

### Step 1. option A
Navigate to this [page](https://codeocean.com/capsule/8807709)

### Step 1. option B
Create new python environment using `requirements.txt` file:  
```
conda create --name nd_examples python=3.7
conda activate nd_examples
pip install -r requirements.txt
```
Create new ipython kernel from environment created above:  
```
pip install ipykernel
ipython kernel install --user --name=nd_examples
```
Modify the `output_dir` variable in the `config.ini` file in order to point to
the accessible directory.

### Step 2.
Run one of following files in order to see results of experiments:
  - `epidemic.py` (example of epidemic propagation with auxiliary processes:
    vaccinations and awareness),
  - `market_competition.ipynb` (marketing campaign of two competitive products
    ),
  - `gossip.ipynb` (example of gossip spreading on two different social 
    networks),
  - `efficiency_tests/tests.ipynb` (comparison of the network_diffusion 
    time-efficiency)

## Remarks

Please note that networks available in this directory have a source
[here](http://multilayer.it.uu.se/datasets.html).

This code is published under GNU General Public License v3 (see `LICENSE` file).
Authors: Michał Czuba, Piotr Bródka
