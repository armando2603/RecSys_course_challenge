#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

from Base.Recommender_utils import check_matrix
from Base.BaseSimilarityMatrixRecommender import BaseUserSimilarityMatrixRecommender
from Base.IR_feature_weighting import okapi_BM_25, TF_IDF
import numpy as np
from tqdm import tqdm
import scipy as sc
from scipy import sparse

from Base.Similarity.Compute_Similarity import Compute_Similarity


class UserKNNCBFRecommender(BaseUserSimilarityMatrixRecommender):
    """ UserKNN recommender"""

    RECOMMENDER_NAME = "UserKNNCBFRecommender"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]

    def __init__(self, URM_train, UCM_train, verbose=True):
        super(UserKNNCBFRecommender, self).__init__(URM_train, verbose=verbose)
        self.URM_train
        self.UCM_train = UCM_train
        self.S = None

    def fit(self, topK=50, shrink=100, normalize=True, feature_weighting="none"):

        self.topK = topK
        self.shrink = shrink

        if feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError(
                "Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'".format(
                    self.FEATURE_WEIGHTING_VALUES, feature_weighting))

        if feature_weighting == "BM25":
            self.UCM_train = self.UCM_train.astype(np.float32)
            self.UCM_train = okapi_BM_25(self.UCM_train)

        elif feature_weighting == "TF-IDF":
            self.UCM_train = self.UCM_train.astype(np.float32)
            self.UCM_train = TF_IDF(self.UCM_train)

        S_matrix_list = []

        UCM = self.UCM_train.tocsr()
        UCM_T = UCM.T.tocsr()

        for i in tqdm(range(0, UCM.shape[1])):
            S_row = UCM_T[i] * UCM
            r = S_row.data.argsort()[:-topK]
            S_row.data[r] = 0

            sparse.csr_matrix.eliminate_zeros(S_row)
            S_matrix_list.append(S_row)

        S = sc.sparse.vstack(S_matrix_list, format='csr')
        S.setdiag(0)
        self.S = S

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        item_scores = self.URM_train[user_id_array, :] * self.S
        return item_scores

