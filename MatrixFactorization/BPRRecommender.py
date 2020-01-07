

from Base.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from Base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Base.Recommender_utils import check_matrix
import numpy as np
import implicit






class BPRRecommender(BaseMatrixFactorizationRecommender, Incremental_Training_Early_Stopping):
    RECOMMENDER_NAME = "BPRRecommender"

    def fit(self, num_factors=150, learning_rate=0.002, reg=0.0015, iterations=100, alpha=5, valid=True,
            **earlystopping_kwargs):

        self.num_epochs = iterations
        self.num_factors = num_factors
        self.alpha = alpha
        self.reg = reg

        if valid:
            self.valid = 5
        else:
            self.valid = -1

        model = implicit.bpr.BayesianPersonalizedRanking(validate_proportion=0.2,
                                                     validate_N=2,
                                                     validate_step=self.valid,
                                                     factors=self.num_factors,
                                                     regularization=self.reg,
                                                     iterations=self.num_epochs)

        # sparse_item_user = self.URM_train.T
        # data_conf = (sparse_item_user * self.alpha).astype('double')
        model.fit(self.URM_train.T)

        self.USER_factors = model.user_factors
        self.ITEM_factors = model.item_factors




