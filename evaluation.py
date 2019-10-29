# Original work Copyright (c) 2017-present, Facebook, Inc.
# Modified work Copyright (c) 2019, Hagai Taitelbaum
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import argparse
import torch

from src.utils import bool_flag, initialize_exp
from src.models import build_model
from src.trainer import Trainer
from src.evaluation import Evaluator


VALIDATION_METRIC = 'mean_cosine-csls_knn_10-S2T-10000_{}'
#   default criterion in the MUSE code: 'precision_at_1-csls_knn_10'

# main
parser = argparse.ArgumentParser(description="Multilingual Word Translation")
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--exp_name", type=str, default="results", help="Experiment name")
parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
# data
parser.add_argument("--langs", nargs='+', default = "en de fr es it pt", help="list of languages")
parser.add_argument("--dicts_path", help="path to evaluation dictionaries", default='./data/dictionaries')
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
#mapping
parser.add_argument("--map_id_init", type=bool_flag, default=True, help="Initialize the mapping as an identity matrix")
parser.add_argument("--mappings_path", default="./data/mappings/six_european", help="path to mappings")
# reload pre-trained embeddings
parser.add_argument("--embs", nargs='+', type=str, default="", help="Reload embeddings")
parser.add_argument("--normalize_embeddings", type=str, default="renorm,center", help="Normalize embeddings before training")
# inference parameters
parser.add_argument("--multilingual_inference_method", nargs='+',help="which inference methods to use", default=['BI','NT','CNT','CAT'])


# parse parameters
params = parser.parse_args()
# check parameters
assert not params.cuda or torch.cuda.is_available()
assert all(os.path.isfile(emb) for emb in params.embs)
assert len(params.langs) == len(params.embs)
assert all([inf_met in ['BI','NT','CNT','CAT'] or inf_met.startswith('CAT')
            for inf_met in params.multilingual_inference_method])

# build logger / model / trainer / evaluator
logger = initialize_exp(params)
embs, mapping = build_model(params)
trainer = Trainer(embs, mapping, params)
evaluator = Evaluator(trainer)

"""
Inference with MWT (Multilingual Word Translation)
"""
logger.info('Starting inference...')

# embeddings evaluation
evaluator.word_translation()
logger.info('End of inference.\n\n')