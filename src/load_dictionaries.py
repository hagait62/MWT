# Original work Copyright (c) 2017-present, Facebook, Inc.
# Modified work Copyright (c) 2019, Hagai Taitelbaum
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from logging import getLogger
import numpy as np
import torch

logger = getLogger()

def load_dictionary(path, word2id1, word2id2,return_numpy=False):
    """
    Return a torch tensor of size (n, 2) where n is the size of the
    loader dictionary, and sort it by source word frequency.
    """
    print(path)
    assert os.path.isfile(path)
    included=[]
    pairs = []
    not_found = 0
    not_found1 = 0
    not_found2 = 0

    with open(path, 'r') as f:
        for i, line in enumerate(f):
            assert line == line.lower()
            try:
                word1, word2 = line.rstrip().split()
            except:
                continue
            if word1 in word2id1 and word2 in word2id2:
                pairs.append((word1, word2))
                included.append(i)
            else:
                not_found += 1
                not_found1 += int(word1 not in word2id1)
                not_found2 += int(word2 not in word2id2)

    logger.info("Found %i pairs of words in the dictionary (%i unique). "
                "%i other pairs contained at least one unknown word "
                "(%i in lang1, %i in lang2)"
                % (len(pairs), len(set([x for x, _ in pairs])),
                   not_found, not_found1, not_found2))

    # sort the dictionary by source word frequencies
    pairs = sorted(pairs, key=lambda x: word2id1[x[0]])
    if return_numpy: dico = np.empty([len(pairs),2],dtype=np.int64)
    else: dico = torch.LongTensor(len(pairs), 2)
    for i, (word1, word2) in enumerate(pairs):
        dico[i, 0] = word2id1[word1]
        dico[i, 1] = word2id2[word2]

    return dico