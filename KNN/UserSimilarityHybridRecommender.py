#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/04/18

@author: Maurizio Ferrari Dacrema
"""

from Base.Recommender_utils import check_matrix, similarityMatrixTopK
from Base.BaseSimilarityMatrixRecommender import BaseUserSimilarityMatrixRecommender
from KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
import scipy.sparse


class UserSimilarityHybridRecommender(BaseUserSimilarityMatrixRecommender):
    """ UserSimilarityHybridRecommender
    Hybrid of two similarities S = S1*alpha + S2*(1-alpha)

    """

    RECOMMENDER_NAME = "UserSimilarityHybridRecommender"


    def __init__(self, urm_train, Similarity_1, Similarity_2, verbose = True):
        super(UserSimilarityHybridRecommender, self).__init__(urm_train, verbose = verbose)


        if Similarity_1.shape != Similarity_2.shape:
            raise ValueError("UserSimilarityHybridRecommender: similarities have different size, S1 is {}, S2 is {}".format(
                Similarity_1.shape, Similarity_2.shape
            ))

        # CSR is faster during evaluation
        self.Similarity_1 = check_matrix(Similarity_1.copy(), 'csr')
        self.Similarity_2 = check_matrix(Similarity_2.copy(), 'csr')


    def fit(self, topK=None, alpha = 0.5):

        self.topK = topK
        self.alpha = alpha

        W_sparse = self.Similarity_1 * self.alpha
        W_sparse += self.Similarity_2 * (1-self.alpha)
        self.W_sparse = W_sparse
        if topK is not None:
            self.W_sparse = similarityMatrixTopK(W_sparse, k=self.topK)
        self.W_sparse = check_matrix(self.W_sparse, format='csr')
