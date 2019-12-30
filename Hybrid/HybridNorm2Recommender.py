from Base.Recommender_utils import check_matrix, similarityMatrixTopK
from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Base.NonPersonalizedRecommender import TopPop
from KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from KNN.UserSimilarityHybridRecommender import UserSimilarityHybridRecommender
from Hybrid.HybridGenRecommender import HybridGenRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from MatrixFactorization.IALSRecommender import IALSRecommender
from Hybrid.HybridNorm1Recommender import HybridNorm1Recommender
import numpy as np
from sklearn.preprocessing import normalize

class HybridNorm2Recommender(BaseItemSimilarityMatrixRecommender):
    """ HybridNormRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)

    """

    RECOMMENDER_NAME = "HybridNorm2Recommender"

    def __init__(self, urm_train):
        super(HybridNorm2Recommender, self).__init__(urm_train)

        urm_train = check_matrix(urm_train.copy(), 'csr')
        self.num_users = urm_train.shape[0]
        recommender_1 = HybridNorm1Recommender(urm_train)
        recommender_1.fit()

        # recommender_2 = HybridGenRecommender(urm_train, ucm_all, icm_all)
        # recommender_2.fit()

        recommender_2 = UserKNNCFRecommender(urm_train)
        recommender_2.fit(topK=697, shrink=1000, feature_weighting='TF-IDF', similarity='tversky', normalize=False, tversky_alpha=1.0, tversky_beta=1.0)

        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2

    def fit(self, alpha=0.6):
        # alpha=0.2, beta=0.8, gamma=0.012, phi=1.2
        self.alpha = alpha

        # self.score_matrix_1 = self.recommender_1._compute_item_matrix_score(np.arange(self.num_users))
        # self.score_matrix_2 = self.recommender_2._compute_item_matrix_score(np.arange(self.num_users))
        #
        # # normalize row-wise
        #
        # item_score_matrix_1 = normalize(self.score_matrix_1, norm='max', axis=1)
        # item_score_matrix_2 = normalize(self.score_matrix_2, norm='max', axis=1)
        #
        # # normalize column-wise
        #
        # user_score_matrix_1 = normalize(self.score_matrix_1.tocsc(), norm='max', axis=0)
        # user_score_matrix_2 = normalize(self.score_matrix_2.tocsc(), norm='max', axis=0)
        #
        # # perform a weighted sum with alpha = 0.6 as the paper do
        #
        # self.score_matrix_1 = item_score_matrix_1 * 0.6 + user_score_matrix_1.tocsr() * 0.4
        # self.score_matrix_2 = item_score_matrix_2 * 0.6 + user_score_matrix_2.tocsr() * 0.4

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        # item_weights_1 = self.score_matrix_1[user_id_array].toarray()
        # item_weights_2 = self.score_matrix_2[user_id_array].toarray()
        # item_weights_3 = self.recommender_3._compute_item_score(user_id_array)

        item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array)

        item_weights = item_weights_1 * self.alpha
        item_weights += item_weights_2 * (1 - self.alpha)
        # item_weights += item_weights_3 * self.gamma

        return item_weights


    # def _compute_item_matrix_score(self, user_id_array, items_to_compute=None):
    #     score_matrix_norm2 = self.score_matrix_1 * self.alpha + self.score_matrix_2 * (1 - self.alpha)
    #     return score_matrix_norm2
