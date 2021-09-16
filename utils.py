import uuid
from functools import partial
from multiprocessing import Pool
from time import gmtime, strftime

import numpy as np
from guacamol.scoring_function import BatchScoringFunction
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import roc_auc_score
from rdkit import DataStructs


def timestamp(adduuid=False):
    s = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    if adduuid:
        s = s + '_' + uuid.uuid4().hex
    return s


def can_list(smiles):
    ms = [Chem.MolFromSmiles(s) for s in smiles]
    return [Chem.MolToSmiles(m) for m in ms if m is not None]


def one_ecfp(smile, radius=2):
    "Calculate ECFP fingerprint. If smiles is invalid return none"
    try:
        m = Chem.MolFromSmiles(smile)
        fp = np.array(AllChem.GetMorganFingerprintAsBitVect(
            m, radius, nBits=1024))
        return fp
    except:
        return None


def ecfp(smiles, radius=2, n_jobs=12):
    with Pool(n_jobs) as pool:
        X = pool.map(partial(one_ecfp, radius=radius), smiles)
    return X


def calc_auc(clf, X_test, y_test):
    preds = clf.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, preds)


def score(smiles_list, clf, max_score=None, fps=None):
    """Makes predictions for a list of smiles. Returns none if smiles is invalid"""
    X = ecfp(smiles_list)
    X_valid = [x for x in X if x is not None]
    if len(X_valid) == 0:
        return X

    preds_valid = clf.predict_proba(np.array(X_valid))[:, 1]
    preds = []
    i = 0
    for li, x in enumerate(X):
        if x is None:
            # print(smiles_list[li], Chem.MolFromSmiles(smiles_list[li]), x)
            preds.append(None)
        else:
            if fps:

                mol = Chem.MolFromSmiles(smiles_list[li])
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 4)
                sim = max([DataStructs.TanimotoSimilarity(query_fp, fp) for query_fp in fps])
                if sim<0.0:
                    score = preds_valid[i] * sim/0.5
                else:
                    score = preds_valid[i]
            else:
                score = preds_valid[i]
            if max_score:
                preds.append(min(max_score, score))
            else:
                preds.append(score)
            assert preds_valid[i] is not None
            i += 1
    return preds


class TPScoringFunction(BatchScoringFunction):
    def __init__(self, clf, max_score, fps):
        super().__init__()
        self.clf = clf
        self.max_score = max_score
        self.fps = fps

    def raw_score_list(self, smiles_list):
        return score(smiles_list, self.clf, self.max_score, self.fps)
