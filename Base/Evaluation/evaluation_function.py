#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 21/10/2018

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import scipy.sparse as sps
from tqdm import tqdm



def precision(is_relevant, relevant_items):

    #is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)

    return precision_score



def recall(is_relevant, relevant_items):

    #is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]

    return recall_score



def MAP(is_relevant, relevant_items):

    #is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

    return map_score



def evaluate_algorithm(URM_test, urm_train, recommender_object, at=10):

    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_MAP = 0.0

    num_eval = 0

    URM_test = sps.csr_matrix(URM_test)

    n_users = URM_test.shape[0]

    user_mask = np.ediff1d(URM_test.indptr) > 0

    user_list = np.arange(n_users)

    user_list = user_list[user_mask]

    for user_id in tqdm(user_list):

        start_pos = URM_test.indptr[user_id]
        end_pos = URM_test.indptr[user_id + 1]
        relevant_items = URM_test.indices[start_pos:end_pos]

        recommended_items = recommend_lightfm(recommender_object,
                                                  user_id,
                                                  urm_train,
                                                  at=at)
        num_eval+=1

        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)


        cumulative_MAP += MAP(is_relevant, relevant_items)



    cumulative_MAP /= num_eval

    print("Recommender performance is:  MAP = {:.4f}".format(cumulative_MAP))

    result_dict = {
        "MAP": cumulative_MAP,
    }

    return result_dict

def recommend_lightfm(lightfm_model, user_id, urm_train, at=10):
    n_users, n_items = urm_train.shape
    score = lightfm_model.predict(user_ids=user_id, item_ids=np.arange(2), num_threads=1)
    #score = _remove_seen_on_scores(urm_train, user_id, score)
    #sort_score = np.argsort(-score)
    #recommended_items = sort_score[:at]
    return score


def _remove_seen_on_scores(urm_train, user_id, scores):

    assert urm_train.getformat() == "csr", "Recommender_Base_Class: URM_train is not CSR, this will cause errors in filtering seen items"

    seen = urm_train.indices[urm_train.indptr[user_id]:urm_train.indptr[user_id + 1]]

    scores[seen] = -np.inf
    return scores
