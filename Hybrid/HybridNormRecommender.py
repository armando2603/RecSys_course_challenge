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
from Hybrid.HybridNorm2Recommender import HybridNorm2Recommender
import numpy as np
from sklearn.preprocessing import normalize

class HybridNormRecommender(BaseItemSimilarityMatrixRecommender):
    """ HybridNormRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)

    """

    RECOMMENDER_NAME = "HybridNormRecommender"

    def __init__(self, urm_train):
        super(HybridNormRecommender, self).__init__(urm_train)

        self.num_users = urm_train.shape[0]

        urm_train = check_matrix(urm_train.copy(), 'csr')

        recommender_1 = HybridGenRecommender(urm_train)
        recommender_1.fit()

        recommender_2 = ItemKNNCFRecommender(urm_train)
        recommender_2.fit(shrink=30, topK=20)

        # recommender_2 = ItemKNNCFRecommender(urm_train)
        # recommender_2.fit(topK=5, shrink=500, feature_weighting='BM25', similarity='tversky', normalize=False, tversky_alpha=0.0, tversky_beta=1.0)

        recommender_3 = UserKNNCFRecommender(urm_train)
        recommender_3.fit(shrink=2, topK=600, normalize=True)
        # recommender_3 = UserKNNCFRecommender(urm_train)
        # recommender_3.fit(topK=697, shrink=1000, feature_weighting='TF-IDF', similarity='tversky', normalize=False,
        #                   tversky_alpha=1.0, tversky_beta=1.0)

        recommender_4 = RP3betaRecommender(urm_train)
        recommender_4.fit(topK=16, alpha=0.03374950051351756, beta=0.24087176329409027, normalize_similarity=True)

        # recommmender_5 = SLIM_BPR_Cython(urm_train)
        # recommmender_5.fit(lambda_i=0.0926694015, lambda_j=0.001697250, learning_rate=0.002391, epochs=65, topK=200)

        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2
        self.recommender_3 = recommender_3
        self.recommender_4 = recommender_4
        # self.recommender_5 = recommmender_5



    def fit(self, alpha=0.2, beta=0.8, gamma=0.012, phi=1.2, psi=0):
        # alpha=0.2, beta=0.8, gamma=0.012, phi=1.2
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.phi = phi
        self.psi = psi

        self.score_matrix_1 = self.recommender_1._compute_item_score(np.arange(self.num_users))
        self.score_matrix_2 = self.recommender_2._compute_item_score(np.arange(self.num_users))
        self.score_matrix_3 = self.recommender_3._compute_item_score(np.arange(self.num_users))
        self.score_matrix_4 = self.recommender_4._compute_item_score(np.arange(self.num_users))
        # self.score_matrix_5 = self.recommender_5._compute_item_score(np.arange(self.num_users))

        # normalize row-wise

        item_score_matrix_1 = normalize(self.score_matrix_1, norm='max', axis=1)
        item_score_matrix_2 = normalize(self.score_matrix_2, norm='max', axis=1)
        item_score_matrix_3 = normalize(self.score_matrix_3, norm='max', axis=1)
        item_score_matrix_4 = normalize(self.score_matrix_4, norm='max', axis=1)

        # normalize column-wise

        user_score_matrix_1 = normalize(self.score_matrix_1, norm='max', axis=0)
        user_score_matrix_2 = normalize(self.score_matrix_2, norm='max', axis=0)
        user_score_matrix_3 = normalize(self.score_matrix_3, norm='max', axis=0)
        user_score_matrix_4 = normalize(self.score_matrix_4, norm='max', axis=0)

        #perform a weighted sum with alpha = 0.6 as the paper do

        self.score_matrix_1 = item_score_matrix_1 * 0.6 + user_score_matrix_1 * 0.4
        self.score_matrix_2 = item_score_matrix_2 * 0.6 + user_score_matrix_2 * 0.4
        self.score_matrix_3 = item_score_matrix_3 * 0.6 + user_score_matrix_3 * 0.4
        self.score_matrix_4 = item_score_matrix_4 * 0.6 + user_score_matrix_4 * 0.4




    def _compute_item_score(self, user_id_array, items_to_compute=None):

        item_weights_1 = self.score_matrix_1[user_id_array]
        item_weights_2 = self.score_matrix_2[user_id_array]
        item_weights_3 = self.score_matrix_3[user_id_array]
        item_weights_4 = self.score_matrix_4[user_id_array]
        # item_weights_5 = self.recommender_5._compute_item_score(user_id_array)


        # item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
        # item_weights_2 = self.recommender_2._compute_item_score(user_id_array)
        # item_weights_3 = self.recommender_3._compute_item_score(user_id_array)
        # item_weights_4 = self.recommender_4._compute_item_score(user_id_array)
        # # item_weights_5 = self.recommender_5._compute_item_score(user_id_array)

        item_weights = item_weights_1 * self.alpha
        item_weights += item_weights_2 * self.beta
        item_weights += item_weights_3 * self.gamma
        item_weights += item_weights_4 * self.phi
        # item_weights += item_weights_5 * self.psi

        return item_weights
