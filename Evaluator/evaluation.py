#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 21/10/2018

@author: Maurizio Ferrari Dacrema
"""

import pandas as pd
import numpy as np
import scipy.sparse as sps
from tqdm import tqdm
from DataManager.split_train_validation_leave_k_out import split_train_leave_k_out_user_wise



def MAP(is_relevant, relevant_items):

    #is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

    return map_score


def evaluate(URM_test, recommender_object, at=10):

    print('Evaluating Test Set...')

    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_MAP = 0.0

    num_eval = 0

    URM_test = sps.csr_matrix(URM_test)

    test_coo = URM_test.tocoo(copy=False)
    urm_test_df = pd.DataFrame({'row': test_coo.row, 'col': test_coo.col, 'data': test_coo.data}
                 )[['row', 'col', 'data']].sort_values(['row', 'col']).reset_index(drop=True)

    users = np.sort(urm_test_df['row'].unique())

    grouped_urm_df = urm_test_df.groupby('row', as_index=True).apply(lambda x: list(x['col']))

    for user_id in tqdm(users):

        relevant_items = np.array(grouped_urm_df[user_id])

        recommended_items = recommender_object.recommend(user_id, at)
        num_eval += 1

        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

        # In case no recommendation are given is relevant is False always

        if len(is_relevant) == 0:
            is_relevant = [False for i in range(10)]
            is_relevant = np.array(is_relevant)

        cumulative_MAP += MAP(is_relevant, relevant_items)

    cumulative_MAP /= num_eval

    print("Recommender performance is: MAP = {:.4f}".format(cumulative_MAP))

    result_dict = {
        "MAP": cumulative_MAP,
    }

    return result_dict
