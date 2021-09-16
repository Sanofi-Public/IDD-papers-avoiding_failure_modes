from copy import deepcopy
import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
from optimize import optimize

opt_args = {}
opt_args['graph_ga'] = dict(
    smi_file='./data/guacamol_v1_train.smiles',
    population_size=100,
    offspring_size=200,
    generations=150,
    mutation_rate=0.01,
    n_jobs=-1,
    patience=150,
    canonicalize=False)

opt_args['lstm_hc'] = dict(
    pretrained_model_path='./guacamol_baselines/smiles_lstm_hc/pretrained_model/model_final_0.473.pt',
    n_epochs=151,
    mols_to_sample=1028,
    keep_top=512,
    optimize_n_epochs=1,
    max_len=100,
    optimize_batch_size=64,
    number_final_samples=1028,
    sample_final_model_only=False,
    smi_file='./data/guacamol_v1_train.smiles',
    n_jobs=-1,
    canonicalize=False)

opt_args['mso'] = dict(
    smi_file='./data/guacamol_v1_valid.smiles',
    num_part=200,
    num_iter=150)

# Set everything that varies in the loop to None
base_config = dict(
    chid=None,
    n_estimators=100,
    n_jobs=8,
    external_file='./data/guacamol_v1_test.smiles',
    n_external=3000,
    seed=None,
    opt_name=None,
    optimizer_args=None)


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    from parametersearch import ParameterSearch
    import os

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--host", type=str, help='host address', default="localhost")
    parser.add_argument("--port", type=int, help='host port', default="7532")
    parser.add_argument("--server", help="run as client process", action="store_true")
    parser.add_argument("--work", help="run as client process", action="store_true")
    parser.add_argument("--nruns", type=int, help='How many runs to perform per task', default=3)
    parser.add_argument("--random_seed_0", type=int, help='Random seed to use when splitting in train/test', default=0)
    parser.add_argument("--random_seed_1", type=int, help='Random seed to use when splitting in split1/split2', default=0)
    parser.add_argument("--min_samples_leaf", type=int, help='min_samples_leaf parameter for the RF', default=1)
    parser.add_argument("--max_depth", type=int, help='max_depth parameter for the RF', default=None)
    parser.add_argument("--random_start", action="store_true")
    parser.add_argument("--return_training_set", action="store_true")

    parser.add_argument("--use_max_score", action="store_true")
    parser.add_argument("--log_base", help='', default='results/test')
    parser.add_argument("--chids_set", help='', default='classic')
    parser.add_argument("--n_estimators", type=int, help='number of trees to use in the RF', default=100)

    args = parser.parse_args()
    for opt_name in ['mso', 'graph_ga', 'lstm_hc']:
        optimizer_args = opt_args[opt_name]
        if args.random_start and opt_name in ['graph_ga', 'lstm_hc']:
            optimizer_args['random_start'] = True
        if args.chids_set == 'classic':
            chids = ['CHEMBL1909203', 'CHEMBL1909140', 'CHEMBL3888429']
        else:
            chids = ['CHEMBL3888429', 'ALDH1']
        for chid in chids:
            for i in range(0, args.nruns):
                config = deepcopy(base_config)
                config['chid'] = chid
                config['seed'] = i
                config['opt_name'] = opt_name
                config['optimizer_args'] = optimizer_args
                config['random_seed_0'] = args.random_seed_0
                config['random_seed_1'] = args.random_seed_1
                config['min_samples_leaf'] = args.min_samples_leaf
                config['max_depth'] = args.max_depth
                config['random_start'] = args.random_start
                config['return_training_set'] = args.return_training_set
                config['use_max_score'] = args.use_max_score 
                config['log_base'] = args.log_base 
                config['n_estimators'] = args.n_estimators

                print(f'Run {i+1}/{args.nruns}, {opt_name}, {chid}')
                optimize(**config)


       
