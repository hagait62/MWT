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
from collections import defaultdict, Counter

from .word_translation import BI_translation, update_translation_for_all_langs
from src.load_dictionaries import load_dictionary
from src.utils import apply_mapping, save_results

logger = getLogger()

class Evaluator(object):

    def __init__(self, trainer):
        """
        Initialize evaluator.
        """
        self.embs = trainer.embs
        self.lang_dico = trainer.lang_dico
        self.mapping = trainer.mapping
        self.params = trainer.params

    def word_translation(self):
        """
        Evaluation on word translation.
        """
        # mapped word embeddings
        all_emb = {l2: apply_mapping(self.mapping[l2], self.embs[l2].weight).data.cpu() for l2 in self.params.langs}
        results = defaultdict(dict)
        # for computational efficiency, iterate over source languages and calculate all methods for each one
        for src_lang in self.params.langs:
            logger.info('\n\n\n\nSource Language: {}\n\n\n\n'.format(src_lang))
            torch.cuda.empty_cache()

            # get source queries
            paths = self.get_dico_paths(src_lang)
            query_ids = self.aggregate_query_ids(paths, src_lang)
            if query_ids is None:
                logger.info('Warning: No test dictionary was found for source language {}. Skipping!'.format(src_lang))
                continue

            method = 'csls_knn_10'
            # init translation
            init_trans, top_scores = BI_translation(src_lang,query_ids,method,all_emb)

            for inf_met in self.params.multilingual_inference_method:
                logger.info('\n\nMultilingual inference method: {}\n\n'.format(inf_met))

                # improve source word representation, and re-translate
                if inf_met != 'BI':
                    updated_trans, used_langs = update_translation_for_all_langs(
                        self.params.langs, src_lang, query_ids, all_emb, init_trans, method, inf_met, top_scores)
                else:
                    used_langs = None
                    updated_trans = init_trans

                # re-arrange translations for convenience
                translation_by_src_id, used_langs_by_src_id = self.translation_by_src_id(updated_trans, used_langs, src_lang)

                # calcualte accuracy, and matching per source word
                for tgt_lang, path in paths.items():
                    pair_result = self.get_pair_accuracy(path, src_lang, self.lang_dico[src_lang].word2id,
                                            tgt_lang, self.lang_dico[tgt_lang].word2id,translation_by_src_id, method)
                    if inf_met != 'BI':
                        self.print_aux_statistics(src_lang, tgt_lang, path, used_langs_by_src_id)
                    results[(src_lang,tgt_lang)][inf_met] = pair_result

        save_results(self.params, results, self.params.multilingual_inference_method)

    def get_dico_paths(self,l1):
        paths = {}
        for l2 in [x for x in self.params.langs if x != l1]:
            path = os.path.join(self.params.dicts_path, '%s-%s.5000-6500.txt' % (l1, l2))
            print(path)
            if not os.path.exists(path):
                logger.info('Warning: Test dictionary for {}-{} not exists. Skipping this pair'.format(l1, l2))
                continue
            paths[l2] = path
        return paths

    def aggregate_query_ids(self, paths, src_lang):
        # find all query words from all available dictionaries
        query_ids = defaultdict(dict)
        for l2, path in paths.items():
            pair_test_dico = load_dictionary(path, self.lang_dico[src_lang].word2id,
                                             self.lang_dico[l2].word2id)
            for q,a in pair_test_dico:
                query_ids[q.item()][l2] = a.item()
        if not bool(query_ids): # empty set
            return None

        logger.info('Number of queries: {}'.format(len(query_ids)))
        return query_ids

    def translation_by_src_id(self, trans_ids, used_langs, src_lang):
        # init.
        test_translation = defaultdict(dict)
        used_langs_by_src_id = defaultdict(dict)
        # iterate over all source ids
        for ii, src_id in enumerate(trans_ids[src_lang]):
            test_translation[src_id.item()] = {k: v[ii].item() for k, v in trans_ids.items()
                                               if k != src_lang}
            if used_langs is not None:
                used_langs_by_src_id[src_id.item()] = {k: v[:, ii] for k, v in used_langs.items()}
            else: used_langs_by_src_id = None
        return test_translation, used_langs_by_src_id

    def get_pair_accuracy(self, path, src_lang, src_wor2id, tgt_lang, tgt_word2id, test_translation, method_name):
        pair_test_dico = load_dictionary(path, src_wor2id, tgt_word2id, True)

        matching = defaultdict(dict)
        prediction = np.array([test_translation[x][tgt_lang] for x in pair_test_dico[:, 0]])
        _matching = prediction == pair_test_dico[:, 1]

        # allow for multiple possible translations
        for i, src_id in enumerate(pair_test_dico[:, 0]):
            matching[src_id] = min(matching.get(src_id, 0) + _matching[i], 1)

        # evaluate precision@k
        precision_at_1 = 100 * np.mean(list(matching.values()))

        logger.info("Language pair {}-{}:".format(src_lang, tgt_lang))
        logger.info("%i source words - %s - Precision at k = 1: %f" %
                    (len(matching), method_name, precision_at_1))
        results = precision_at_1
        return results

    def aux_statistics(self, path, src_word2id, tgt_word2id, used_langs_by_src_id, tgt_lang):
        # extract task specific source words
        pair_test_dico = load_dictionary(path, src_word2id, tgt_word2id, True)
        src_ids = pair_test_dico[:, 0]

        # find used languages
        mask = torch.stack([used_langs_by_src_id[src_id][tgt_lang]
                            for src_id in np.unique(src_ids)], dim=0).gt(0).float()

        # how many times each language was used
        counts = mask.sum(0)
        # how many languages were used for each source word, sum -1 to igonre source language which always included
        dist_of_number_of_aux_langs = Counter((mask.sum(1) - 1).long().tolist())
        return counts, dist_of_number_of_aux_langs

    def print_aux_statistics(self, src_lang, tgt_lang, path, used_langs_by_src_id):

        counts, dist_of_number_of_aux_langs = self.aux_statistics(path, self.lang_dico[src_lang].word2id,
                                               self.lang_dico[tgt_lang].word2id, used_langs_by_src_id, tgt_lang)

        # frequencies of used auxiliary languages
        langs_with_none = self.params.langs + ['None']
        none_aux_count = torch.Tensor([dist_of_number_of_aux_langs.get(0,0)])
        counts = torch.cat([counts,none_aux_count])
        freqs = counts/counts[self.params.langs.index(src_lang)]
        freqs_str = '; '.join(['{}-{:.4f}'.format(lang,f) for lang,f in zip(langs_with_none,freqs)])
        logger.info('pair {}-{}: frequencies: {}'.format(src_lang, tgt_lang, freqs_str))

        # most used auxiliary language
        max_freq_ix = freqs.topk(2)[1][1].item() # most frequent is the source language itself
        logger.info('Most frequent auxiliary languauge: {}. frequency: {:.4f}'.format(langs_with_none[max_freq_ix],freqs[max_freq_ix]))

        # distribution of number of used auxiliary languages
        aux_langs_dist_str = '\n'.join(['{} words used {} aux languages'.format(v,k)
                                        for k, v in sorted(dist_of_number_of_aux_langs.items())])
        logger.info('\n'.join(['number of aux langs used:',aux_langs_dist_str]))
        avg_aux_langs = sum([k*v for k,v in dist_of_number_of_aux_langs.items()])/\
                        sum(list(dist_of_number_of_aux_langs.values()))
        logger.info('Avergae number of aux langs: {:.4f}'.format(avg_aux_langs))