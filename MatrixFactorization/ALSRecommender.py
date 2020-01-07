"""
Created on 23/03/2019

@author: Maurizio Ferrari Dacrema
"""



from Base.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from Base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Base.Recommender_utils import check_matrix
import numpy as np
import implicit


class ALSRecommender(BaseMatrixFactorizationRecommender, Incremental_Training_Early_Stopping):
    """

    Binary/Implicit Alternating Least Squares (IALS)
    See:
    Y. Hu, Y. Koren and C. Volinsky, Collaborative filtering for implicit feedback datasets, ICDM 2008.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.167.5120&rep=rep1&type=pdf

    R. Pan et al., One-class collaborative filtering, ICDM 2008.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.306.4684&rep=rep1&type=pdf

    Factorization model for binary feedback.
    First, splits the feedback matrix R as the element-wise a Preference matrix P and a Confidence matrix C.
    Then computes the decomposition of them into the dot product of two matrices X and Y of latent factors.
    X represent the user latent factors, Y the item latent factors.

    The model is learned by solving the following regularized Least-squares objective function with Stochastic Gradient Descent
    \operatornamewithlimits{argmin}\limits_{x*,y*}\frac{1}{2}\sum_{i,j}{c_{ij}(p_{ij}-x_i^T y_j) + \lambda(\sum_{i}{||x_i||^2} + \sum_{j}{||y_j||^2})}
    """

    RECOMMENDER_NAME = "ALSRecommender"



    def fit(self, num_factors=250, reg=0.3, iterations=200, alpha=5, valid=True,
            **earlystopping_kwargs):

        self.num_epochs = iterations
        self.num_factors = num_factors
        self.alpha = alpha
        self.reg = reg

        if valid:
            self.valid=5
        else:
            self.valid=-1

        model = implicit.als.AlternatingLeastSquares(validate_proportion=0.2,
                                                     validate_N=2,
                                                     validate_step=self.valid,
                                                     factors=self.num_factors,
                                                     regularization=self.reg,
                                                     iterations=self.num_epochs)

        sparse_item_user = self.URM_train.T
        data_conf = (sparse_item_user * self.alpha).astype('double')
        model.fit(data_conf)

        self.USER_factors = model.user_factors
        self.ITEM_factors = model.item_factors











