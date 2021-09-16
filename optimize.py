import json
import os
import pickle
import sys
from time import time
import rdkit
from rdkit import Chem

from rdkit.Chem import rdMolDescriptors, AllChem

mso_dir = os.path.join(os.path.dirname(__file__), 'mso')
sys.path.append(mso_dir)

import numpy as np
import pandas as pd
import torch
from cddd.inference import InferenceModel
from mso.objectives.scoring import ScoringFunction
from mso.optimizer import BasePSOptimizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from guacamol_baselines.graph_ga.goal_directed_generation import \
    GB_GA_Generator
from guacamol_baselines.smiles_lstm_hc.smiles_rnn_directed_generator import \
    SmilesRnnDirectedGenerator
from utils import TPScoringFunction, calc_auc, can_list, ecfp, score, timestamp


def fit_clfs(chid, n_estimators, n_jobs, random_seed_0=0, random_seed_1=0, min_samples_leaf=1, max_depth=None, return_training_set=False):
    """
    Args:
        chid: which assay to use:
        external_file:
    Returns:
        clfs: Dictionary of fitted classifiers
        aucs: Dictionary of AUCs
        balance: Two numbers showing the number of actives in split 1 / split 2
        df1: data in split 1
        df2: data in split 2
    """
    # read data and calculate ecfp fingerprints
    
    if chid=="ALDH1":
        assay_file = f'./assays/processed/210415_LIT_PCBA_ALDH1_Best_AB_sets.csv'
        print(f'Reading data from: {assay_file}')
        df = pd.read_csv("210415_LIT_PCBA_ALDH1_Best_AB_sets.csv")
        in_set_A = np.where(df['in_set_A']==1)[0]
        in_set_B = np.where(df['in_set_B']==1)[0]
        smiles = df["SMILES"]
        activity = df["Act"]
        index_test = []
        smiles_test = []
        smiles_train = []
        for i in range(len(smiles)):
            if i not in in_set_A and i not in in_set_B:
                index_test.append(i)
                smiles_test.append(smiles[i])
            if i in in_set_A:
                smiles_train.append(smiles[i])

        in_test = np.array(index_test)

        X1 = np.array(ecfp(smiles[in_set_A]))
        X2 = np.array(ecfp(smiles[in_set_B]))

        y1 = np.array(activity[in_set_A])
        y2 = np.array(activity[in_set_B])
        balance = (np.mean(y1), np.mean(y2))
    
    # train classifiers and store them in dictionary
        clfs = {}
        clfs['Split1'] = RandomForestClassifier(
            n_estimators=n_estimators, random_state=0)
        clfs['Split1'].fit(X1, y1)

        clfs['Split1_alt'] = RandomForestClassifier(
            n_estimators=n_estimators, random_state=1)
        clfs['Split1_alt'].fit(X1, y1)

        clfs['Split2'] = RandomForestClassifier(
            n_estimators=n_estimators, random_state=0)
        clfs['Split2'].fit(X2, y2)
        
    else:
        assay_file = f'./assays/processed/{chid}.csv'
        print(f'Reading data from: {assay_file}')
        df = pd.read_csv(assay_file)

        df['ecfp'] = ecfp(df.smiles)
        df_train, df_test = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=random_seed_0)


        df1, df2 = train_test_split(df_train, test_size=0.5, stratify=df_train['label'], random_state=random_seed_1)
        X1 = np.array(list(df1['ecfp']))
        X2 = np.array(list(df2['ecfp']))

        y1 = np.array(list(df1['label']))
        y2 = np.array(list(df2['label']))

        del df1['ecfp']
        del df2['ecfp']
        smiles_test = list(df_test.smiles)
        smiles_train = list(df1.smiles)
        balance = (np.mean(y1), np.mean(y2))
    
    # train classifiers and store them in dictionary
        clfs = {}
        clfs['Split1'] = RandomForestClassifier(
            n_estimators=n_estimators, n_jobs=n_jobs, random_state=0, min_samples_leaf=min_samples_leaf, max_depth=max_depth)
        clfs['Split1'].fit(X1, y1)

        clfs['Split1_alt'] = RandomForestClassifier(
            n_estimators=n_estimators, n_jobs=n_jobs, random_state=1, min_samples_leaf=min_samples_leaf, max_depth=max_depth)
        clfs['Split1_alt'].fit(X1, y1)

        clfs['Split2'] = RandomForestClassifier(
            n_estimators=n_estimators, n_jobs=n_jobs, random_state=0, min_samples_leaf=min_samples_leaf, max_depth=max_depth)
        clfs['Split2'].fit(X2, y2)
    

    # calculate AUCs for the clfs
    aucs = {}
    aucs['Split1'] = calc_auc(clfs['Split1'], X2, y2)
    aucs['Split1_alt'] = calc_auc(clfs['Split1_alt'], X2, y2)
    aucs['Split2'] = calc_auc(clfs['Split2'], X1, y1)
    print("AUCs:")
    for k, v in aucs.items():
        print(f'{k}: {v}')
    if return_training_set:
        return clfs, aucs, balance, df1, df2, smiles_train, max(clfs['Split1'].predict_proba(ecfp(smiles_test))[:, 1])
    else:
        return clfs, aucs, balance, df1, df2, smiles_test, max(clfs['Split1'].predict_proba(ecfp(smiles_test))[:, 1])


def optimize(chid,
             n_estimators,
             n_jobs,
             external_file,
             n_external,
             seed,
             opt_name,
             optimizer_args,
             log_base,
             random_seed_0,
             random_seed_1,
             min_samples_leaf,
             max_depth,
             random_start, 
             use_max_score,
             return_training_set):
    """
    Args:
        - chid: which assay to use
        - n_estimators: how many trees to use in Random Forest
        - n_jobs: how many parallel processes to use
        - external_file: Smiles that are not used for optimization
        - n_external: on how many such independent random points to calculate scores
        - seed: which random seed to use
        - opt_name: which optimizer to use (graph_ga or lstm_hc)
        - optimizer_args: dictionary with arguments for the optimizer
        - log_base: Where to store results. Will be appended by timestamp
    """
    config = locals()

    # Results might not be fully reproducible when using pytorch
    # https://pytorch.org/docs/stable/notes/randomness.html
    np.random.seed(seed)
    torch.manual_seed(seed)

    # set up logging
    results_dir = os.path.join(log_base, opt_name, chid, timestamp())
    os.makedirs(results_dir)

    config_file = os.path.join(results_dir, 'config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f)
    
    clfs, aucs, balance, df1, df2, initial_smiles, max_score = fit_clfs(chid, n_estimators, n_jobs, random_seed_0, random_seed_1, min_samples_leaf, max_depth, return_training_set)
    results = {}
    results['AUC'] = aucs
    results['balance'] = balance

    clf_file = os.path.join(results_dir, 'classifiers.p')
    with open(clf_file, 'wb') as f:
        pickle.dump(clfs, f)

    df1.to_csv(os.path.join(results_dir, 'split1.csv'), index=False)
    df2.to_csv(os.path.join(results_dir, 'split2.csv'), index=False)

    # Create guacamol scoring function with clf trained on split 1
    if use_max_score:
        scoring_function = TPScoringFunction(clfs['Split1'], max_score)
    else:
        scoring_function = TPScoringFunction(clfs['Split1'], None)
    infer_model = InferenceModel(model_dir='default_model') 
    # The CDDD inference model used to encode/decode molecular SMILES strings to/from the CDDD space. You might need to specify the path to the pretrained model (e.g. default_model)
   
    mso_score = [ScoringFunction(func=scoring_function.raw_score_list, name='score', is_mol_func=False, is_smiles_func=True)]



    class MsoWrapper(object):
        def __init__(self, init_smiles, smi_file, num_part, num_iter):
            self.smi_file = smi_file
            self.num_part = num_part
            self.num_iter = num_iter
            self.init_smiles = init_smiles
            with open(self.smi_file) as f:
                self.start_pool = f.read().split()

        def run(self):
            if random_start:
                init_smiles = list(np.random.choice(self.start_pool, self.num_part))
            else:
                init_smiles = self.init_smiles

            opt = BasePSOptimizer.from_query(
                init_smiles=init_smiles,
                num_part=200,
                num_swarms=1,
                inference_model=infer_model,
                scoring_functions=mso_score)

            _, smiles_history = opt.run(self.num_iter)
            return smiles_history

    # run optimization
    t0 = time()
    if opt_name == 'graph_ga':
        optimizer = GB_GA_Generator(**optimizer_args)
    elif opt_name == 'lstm_hc':
        optimizer = SmilesRnnDirectedGenerator(**optimizer_args)
    elif opt_name == 'mso':
        optimizer = MsoWrapper(initial_smiles, **optimizer_args)
    else:
        raise ValueError(f'Invalid optimizer: {opt_name}')

    if opt_name == 'mso':
        smiles_history = optimizer.run()
    else:
        if random_start:
            smiles_history = optimizer.generate_optimized_molecules(
                scoring_function, 100, get_history=True)
        else:
            smiles_history = optimizer.generate_optimized_molecules(
                scoring_function, 100, get_history=True, starting_population=initial_smiles)

    smiles_history = [can_list(e) for e in smiles_history]

    t1 = time()
    opt_time = t1 - t0

    # make a list of dictionaries for every time step
    # this is far from an optimal data structure
    statistics = []
    for optimized_smiles in smiles_history:
        row = {}
        row['smiles'] = optimized_smiles
        row['preds'] = {}
        for k, clf in clfs.items():
            preds = score(optimized_smiles, clf)
            if None in preds:
                print('Invalid score. Debug message')
            row['preds'][k] = preds
        statistics.append(row)

    results['statistics'] = statistics

    stat_time = time() - t1
    # add predictions on external set
    # load external smiles for evaluation
    with open(external_file) as f:
        external_smiles = f.read().split()
    external_smiles = np.random.choice(external_smiles, n_external)
    results['predictions_external'] = {k: score(external_smiles, clf) for k, clf in clfs.items()}

    results_file = os.path.join(results_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f)

    print(f'Storing results in {results_dir}')
    print(f'Optimization time {opt_time:.2f}')
    print(f'Statistics time {stat_time:.2f}')


if __name__ == '__main__':
    # some default settings for both optimizers

    opt_args = {}
    opt_args['graph_ga'] = dict(
        smi_file='./data/guacamol_v1_valid.smiles',
        population_size=100,
        offspring_size=200,
        generations=5,
        mutation_rate=0.01,
        n_jobs=-1,
        random_start=True,
        patience=150,
        canonicalize=False)

    opt_args['lstm_hc'] = dict(
        pretrained_model_path='./guacamol_baselines/smiles_lstm_hc/pretrained_model/model_final_0.473.pt',
        n_epochs=1,
        mols_to_sample=1028,
        keep_top=512,
        optimize_n_epochs=1,
        max_len=100,
        optimize_batch_size=64,
        number_final_samples=1028,
        sample_final_model_only=False,
        random_start=True,
        smi_file='./data/guacamol_v1_train.smiles',
        n_jobs=-1,
        canonicalize=False)

    opt_args['mso'] = dict(
        smi_file='./data/guacamol_v1_valid.smiles',
        num_part=200,
        num_iter=150)

    # which optimizer to use
    # opt_name = 'graph_ga'
    # opt_name = 'lstm_hc'
    opt_name = 'mso'
    optimizer_args = opt_args[opt_name]

    config = dict(
        chid='CHEMBL3888429',
        n_estimators=100,
        n_jobs=8,
        external_file='./data/guacamol_v1_test.smiles',
        n_external=3000,
        seed=101,
        opt_name=opt_name,
        optimizer_args=optimizer_args,
        log_base='test')

    optimize(**config)
