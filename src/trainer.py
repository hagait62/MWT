# Original work Copyright (c) 2017-present, Facebook, Inc.
# Modified work Copyright (c) 2019, Hagai Taitelbaum
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from logging import getLogger
import torch


logger = getLogger()


class Trainer(object):

    def __init__(self, embs, mapping, params):
        """
        Initialize trainer script.
        """
        self.embs = embs
        self.lang_dico = getattr(params, 'lang_dico', None)
        self.mapping = mapping
        self.params = params
        self.load_mpsr_matrices()

    def load_mpsr_matrices(self):
        logger.info('init matrices: load matrices...')
        T = {lang: self.mapping[lang].weight.data for lang in self.params.langs}
        for lang in self.params.langs:
            if lang == 'en':
                continue
            path = os.path.join(self.params.mappings_path ,f'best_mapping_{lang}2en.t7')
            logger.info(f'* Reloading the best {lang} to en model from {path} ...')
            # reload the model
            assert os.path.isfile(path)
            to_reload = torch.from_numpy(torch.load(path))
            assert to_reload.size() == T[lang].size()
            T[lang].copy_(to_reload.type_as(T[lang]))