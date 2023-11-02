import argparse
import importlib
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
import numpy as np
import tensorflow as tf

seed = 1234
os.environ['PYTHONHASHSEED'] = str(seed)
# os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

parser = argparse.ArgumentParser(description='GNN-based Entity Alignment')
parser.add_argument('--input', type=str, default='data/DBP15K/zh_en/')
# parser.add_argument('--input', type=str, default='data/SRPRS/en_fr_15k_V1/')
parser.add_argument('--train_ratio', type=float, default=0.3)

parser.add_argument('--embedding_model', type=str, default='GCN', choices=['GCN', 'GAT'])
parser.add_argument('--ent_dim', type=int, default=300)
parser.add_argument('--layer_num', type=int, default=2)
parser.add_argument('--head_num', type=int, default=1)
parser.add_argument('--skip_conn', type=str, default='none', choices=['none', 'highway', 'concat'])
parser.add_argument('--ent_name_init', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--pretrained_model', type=str, default='fasttext', choices=['fasttext', 'bert'])
parser.add_argument('--activation', type=str, default='tanh')
parser.add_argument('--dropout', type=float, default=0.2, help='best dropout values for GCN, GCN+highway, GCN+concat, '
                                                               'GAT, GAT+highway, GAT+concat are '
                                                               '0.2, 0.3, 0.1, 0.0, 0.1 and 0.2 respectively')

parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--max_iter', type=int, default=3)
parser.add_argument('--iter_type', type=str, default='bi-direction', choices=['bi-direction', 'bootstrapping'])
parser.add_argument('--update_adj', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--max_epoch', type=int, default=2000)
parser.add_argument('--batch_num', type=int, default=1)
parser.add_argument('--neg_strategy', type=str, default='nearest', choices=['nearest', 'uniform'])
parser.add_argument('--neg_multi', type=int, default=30)
parser.add_argument('--neg_margin', type=float, default=3)

parser.add_argument('--inference_strategy', type=str, default='NN', choices=['NN', 'SM'])
parser.add_argument('--eval_metric', type=str, default='cosine', choices=['cosine', 'manhattan', 'euclidean'])
parser.add_argument('--eval_csls', type=int, default=0)

parser.add_argument('--early_stopping', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--start_valid', type=int, default=100)
parser.add_argument('--eval_freq', type=int, default=10)
parser.add_argument('--neg_update_freq', type=int, default=10)
parser.add_argument('--hits_k', type=list, default=[1, 10])
parser.add_argument('--eval_threads_num', type=int, default=10)


def get_model(model_name):
    module = importlib.import_module("models.gnn_models")
    return getattr(module, model_name)


if __name__ == '__main__':
    args = parser.parse_args()
    # print(args)

    model = get_model(args.embedding_model)()
    model.set_args(args)
    model.set_kgs()
    model.init()

    for iter in range(args.max_iter):
        print("Iteration {}:".format(iter))
        if iter > 0:
            if args.iter_type == 'bi-direction':
                model.generate_new_alignments()
            else:
                model.bootstrapping()
            # model.eval_new_alignments()
            if args.update_adj:
                model.update_adj()
        model.train()
        model.test()


