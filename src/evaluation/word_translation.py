# Original work Copyright (c) 2017-present, Facebook, Inc.
# Modified work Copyright (c) 2019, Hagai Taitelbaum
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import torch
from ..utils import get_nn_avg_dist

logger = getLogger()


def translation(emb1,emb2,method,query_ids,improved_query=None):

    # normalize word embeddings
    emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
    emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)
    sorted_query_keys = sorted(query_ids.keys())

    if improved_query is None:
        query = emb1[sorted_query_keys].cuda()
    else:
        improved_query = improved_query.cuda()
        query = improved_query / improved_query.norm(2, 1, keepdim=True).expand_as(
            improved_query)

    # nearest neighbors
    if method == 'nn':
        emb2 = emb2.cuda()
        scores = query.mm(emb2.transpose(0, 1)).cpu()

    # contextual dissimilarity measure
    elif method.startswith('csls_knn_'):
        # average distances to k nearest neighbors
        knn = method[len('csls_knn_'):]
        assert knn.isdigit()
        knn = int(knn)
        average_dist1 = get_nn_avg_dist(emb2, emb1, knn)
        average_dist2 = get_nn_avg_dist(emb1, emb2, knn)
        average_dist1 = torch.from_numpy(average_dist1).type_as(emb1)
        average_dist2 = torch.from_numpy(average_dist2).type_as(emb2)
        # queries / scores
        emb2 = emb2.cuda()
        scores = query.mm(emb2.transpose(0, 1)).cpu()
        scores.mul_(2)
        scores.sub_(average_dist1[sorted_query_keys][:, None] + average_dist2[None, :])

    else:
        raise Exception('Unknown method: "%s"' % method)

    top_scores, top_matches = scores.max(1)
    return top_matches.cpu(), top_scores.cpu()

def BI_translation(src_lang,query_ids,method,all_emb):
    '''
    return top ids, and scores of BI translation to all languages. 
    '''
    logger.info('translate BI translation...')
    init_trans = {tgt: translation(all_emb[src_lang], emb, method, query_ids)
                  for tgt, emb in all_emb.items() if tgt != src_lang}
    top_scores = {l2: x[1] for l2, x in init_trans.items()}
    init_trans = {l2: x[0] for l2, x in init_trans.items()}
    # add source ids
    init_trans.update({src_lang: torch.LongTensor(sorted(query_ids.keys()))})

    logger.info('finish BI for source lang: {}, method: {}'.format(src_lang, method))
    return init_trans, top_scores

def update_translation_for_all_langs(lang_list,src_lang,query_ids,all_emb,init_trans,method,inf_met,scores_dict):
    # re-arrange languages with src_lang first
    langs_wo_src = [x for x in lang_list if x != src_lang]
    new_lang_list = [src_lang] + langs_wo_src
    updated_trans = {src_lang:init_trans[src_lang]}

    # organize embs & scores by new order
    init_trans_aux_emb = torch.stack([all_emb[l][init_trans[l]] for l in new_lang_list], dim=0)
    scores = torch.stack([scores_dict[l] for l in langs_wo_src], dim=0)

    # init.
    mask = torch.zeros(len(lang_list),len(query_ids))
    mask[0] = 1  # source language always included
    used_langs_ids = {}

    # improve translation for each target language
    for ix,tgt_lang in enumerate(langs_wo_src,1):
        scores_wo_tgt = scores.clone()
        scores_wo_tgt[ix-1] = -1e12 # target not included
        tgt_scores = scores[langs_wo_src.index(tgt_lang)]

        # compute mask of relevant aux embs
        compute_mask(inf_met, mask, scores_wo_tgt, tgt_scores)
        assert all(mask[ix] == 0) # validate that target not included

        used_langs_ids[tgt_lang] = get_shifted_ids_of_used_langs(mask,src_lang,init_trans,lang_list)

        # calculate weights for improved representation
        langs_weights = mask.clone()
        calc_weights(inf_met, langs_weights, scores_wo_tgt, tgt_scores)
        
        # calculate improved representation
        improved_emb_query = init_trans_aux_emb.permute(1,2,0).bmm(langs_weights.permute(1,0)[:,:,None]).squeeze(2)
        improved_emb_query = improved_emb_query/(langs_weights.sum(dim=0,keepdim=True).t())

        # re-translate using multilingual information
        updated_trans[tgt_lang], _ = translation(all_emb[src_lang], all_emb[tgt_lang], method,query_ids, improved_query=improved_emb_query)
    return updated_trans, used_langs_ids

def get_shifted_ids_of_used_langs(mask,src_lang,trans_aux_ids,lang_list):
    # ids of used languages for each source word (shifted by 1 to differentiate with masked entries)

    # re-order mask to be compatible with lang_list
    src_old_ix = lang_list.index(src_lang)
    ordered_mask = torch.cat([mask[1:1 + src_old_ix], mask[0:1], mask[1 + src_old_ix:]]).long()
    # Shift to differentiate between (1) id 0, and (2) masked value
    return torch.stack([trans_aux_ids[l] + 1 for l in lang_list], dim=0) * ordered_mask

def compute_mask(inf_met,mask,scores_wo_tgt,tgt_scores):
    if inf_met.startswith('CAT'):
        mask_CAT(mask, scores_wo_tgt, tgt_scores)
    elif inf_met == 'CNT':
        mask_CNT(scores_wo_tgt, mask, tgt_scores)
    elif inf_met == 'NT':
        mask_NT(scores_wo_tgt, mask)
    else:
        raise ValueError('Unrecognized inference method. got inf_met={}'.format(inf_met))

def mask_NT(scores_wo_tgt,mask):
    # use language with maximum score
    max_score = scores_wo_tgt.max(dim=0)[0]
    mask[1:] = (scores_wo_tgt == max_score).float()
    # number of langs=2: src, aux
    assert all(mask.sum(0) == 2)

def mask_CNT(scores_wo_tgt,mask,tgt_scores):
    # use language with maximum score, only if its score is greater then target score
    max_score = scores_wo_tgt.max(dim=0)[0]
    mask[1:] = scores_wo_tgt.ge(torch.max(tgt_scores, max_score)).float()
    # number of langs<=2: src, and maybe aux
    assert all(mask.sum(0) <= 2)

def mask_CAT(mask,scores_wo_tgt,tgt_scores):
    # use all languages higher score than target BI translation score
    mask[1:] = (scores_wo_tgt > tgt_scores).float()

def calc_weights(inf_met,langs_weights,scores_wo_tgt,tgt_scores):
    if inf_met == 'CAT':
        weight_CAT(langs_weights)
    elif inf_met == 'CAT_by_score':
        weight_CAT_by_score(langs_weights, scores_wo_tgt)
    elif inf_met == 'CAT_by_relative_score':
        weight_CAT_by_relative_score(langs_weights, scores_wo_tgt, tgt_scores)
    elif inf_met == 'CAT_by_exp_relative_score':
        weight_CAT_by_exp_relative_score(langs_weights, scores_wo_tgt, tgt_scores)

def weight_CAT(langs_weights):
    # weight source word as the some of weights of all auxiliary languages
    langs_weights[0] = torch.max(torch.ones(langs_weights.size(1)), langs_weights[1:].sum(0))

def weight_CAT_by_score(langs_weights, scores_wo_tgt):
    langs_weights[1:] = langs_weights[1:] * scores_wo_tgt
    langs_weights[0] = langs_weights[1:].sum(0)
    langs_weights[0][langs_weights[0] == 0] = 1

def weight_CAT_by_relative_score(langs_weights,scores_wo_tgt,tgt_scores):
    langs_weights[1:] = langs_weights[1:] * (scores_wo_tgt - tgt_scores)
    langs_weights[0] = langs_weights[1:].sum(0)
    langs_weights[0][langs_weights[0] == 0] = 1

def weight_CAT_by_exp_relative_score(langs_weights,scores_wo_tgt,tgt_scores):
    langs_weights[1:] = torch.where(langs_weights[1:] != 0,
                                    langs_weights[1:] * torch.exp(scores_wo_tgt - tgt_scores),
                                    torch.zeros_like(langs_weights[1:]))
    langs_weights[0] = langs_weights[1:].sum(0)
    langs_weights[0][langs_weights[0] == 0] = 1