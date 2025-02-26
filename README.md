# Explaining and avoiding failures modes in goal-directed generation

This code reproduces the results found in the paper "Explaining and avoiding failures modes in goal-directed generation".
The paper builds on the work of Renz *and al* <sup>1</sup>, that is available at: https://www.sciencedirect.com/science/article/pii/S1740674920300159

This code is a fork from the repository supporting <sup>1</sup>. This code is available at: https://github.com/ml-jku/mgenerators-failure-modes. The main difference with the original codebase is new notebooks supporting our experiments. 
Another modification has been performed: the call to the GPL-ed library implementing the Levenshtein distance in addcarbon.py has been replaced by a function included in addcarbon.py.

We thank the authors of <sup>1</sup> both for their very insightful work, and their well-written and reproducible codebase. 

<sup>1</sup> https://doi.org/10.1016/j.ddtec.2020.09.003 

## Summary of the results

Renz *and al* highlighted (among many other interesting results on generative models for molecules) that in goal-directed generation, molecules generated can have high optimization scores and in the meantime low scores according to control models:

![plot](readme_figures/median_scores.png)

To explain those results, we looked at the agreement between optimization model and control models on the initial data distribution:

![plot](readme_figures/dataset_analysis.PNG)

We then assess whether this initial difference could explain the previous results:

![plot](readme_figures/tolerance_intervals_data_control.PNG)

The main conclusion of our work is that the underlying issue lies in the initial disagreement between optimization and control models, and not with the goal-directed generation algorithm. 

## Code
The instructions for installation are the same as described in https://github.com/ml-jku/mgenerators-failure-modes.

### Install dependencies
```
pip install -r requirements.txt
conda install rdkit -c rdkit
wget https://raw.githubusercontent.com/jrwnter/cddd/master/download_default_model.sh -O- -q | bash
```

### Download Guacamol data splits
The compounds are used for distribution learning and for starting populations for the graph-based genetic algorithm.
```
mkdir data
wget -O data/guacamol_v1_all.smiles https://ndownloader.figshare.com/files/13612745
wget -O data/guacamol_v1_test.smiles https://ndownloader.figshare.com/files/13612757
wget -O data/guacamol_v1_valid.smiles https://ndownloader.figshare.com/files/13612766
wget -O data/guacamol_v1_train.smiles https://ndownloader.figshare.com/files/13612760
```
### Bioactivity data
The csv-files downloaded from ChEMBL are located in `assays/raw`.
Running the `preprocess.py` script will transform the data into binary classification tasks and store them in `assays/processed`.

## Experiments

An alternative to running the experiments (which can take time) is to unzip the "results.zip" archive. Results from the paper can then be reproduced from there by running the different notebooks.
1. To reproduce the original results presented in "On failure modes in molecule generation and optimization":
```
python run_goal_directed.py --log_base results/original_start_chembl --nruns 10 --random_start 
```

2. To run the same analysis while using the dataset as a starting point:
```
python run_goal_directed.py --log_base results/original_start_dataset --nruns 10
```

3. To run the experiments on the ALDH1 dataset and the JAK2 dataset with modified parameters for the predictive model:
```
python run_goal_directed.py --log_base results/new_datasets_start_from_chembl --nruns 10 --random_start --chids_set alternative --n_estimators 200 --min_samples_leaf 3
```
4. `dataset_analysis.ipynb`: Analysis of the relationships between optimization and control scores on the distribution of the dataset.
5. `run_analysis.ipynb`: Analysis of the experiment on the new datasets (ALDH1 and JAK2 modified).
5. `tolerance_intervals.ipynb`: Computes tolerance intervals for expected control scores, and plot them alongside actual control scores obtained during the experiments.
6. `nn_analysis.ipynb`: Analyze whether there is already a bias towards higher similarities with molecules from Split 1 for high scoring molecules in the dataset. 
7. `display_molecules.ipynb`: shows outliers from the DRD2 dataset, and molecules generated during the different experiments. 
