from Base.Recommender_utils import check_matrix, similarityMatrixTopK
from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Base.NonPersonalizedRecommender import TopPop
from KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from KNN.UserSimilarityHybridRecommender import UserSimilarityHybridRecommender
import scipy.sparse as sp
from DataManager.DataManager import DataManager
import numpy as np
from sklearn.preprocessing import normalize

class HybridGen2Recommender(BaseItemSimilarityMatrixRecommender):
    """ HybridGen2Recommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)

    """

    RECOMMENDER_NAME = "HybridGen2Recommender"

    def __init__(self, urm_train):
        super(HybridGen2Recommender, self).__init__(urm_train)

        self.num_users = urm_train.shape[0]

        urm_train = check_matrix(urm_train.copy(), 'csr')

        icm_weighted = sp.load_npz('Data/icm_weighted.npz')
        data = DataManager()
        ucm_age, ucm_region, ucm_all = data.get_ucm()
        _, _, _, icm_all = data.get_icm()

        recommender_1 = ItemKNNCBFRecommender(urm_train, icm_weighted)
        recommender_1.fit(topK= 100, shrink=5, feature_weighting='none', normalize=False)

        recommender_2 = UserKNNCBFRecommender(urm_train, ucm_all)
        recommender_2.fit(shrink=1777, topK=1998, similarity='tversky',
                    feature_weighting='BM25',
                    tversky_alpha=0.1604953616,
                    tversky_beta=0.9862348646)

        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2
        # self.recommender_3 = recommender_3

    def fit(self, alpha=0.11, beta=0.03, gamma=0):
        self.alpha = alpha
        # self.beta = beta
        # self.gamma = gamma

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
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array)

        # item_weights_1 = self.score_matrix_1[user_id_array].toarray()
        # item_weights_2 = self.score_matrix_2[user_id_array].toarray()
        # item_weights_3 = self.recommender_3._compute_item_score(user_id_array)

        item_weights = item_weights_1 * self.alpha
        item_weights += item_weights_2 * (1 - self.alpha)
        # item_weights += item_weights_3 * self.gamma

        return item_weights


    # def _compute_item_matrix_score(self, user_id_array, items_to_compute=None):
    #     return self.score_matrix_1 * self.alpha + self.score_matrix_2 * (1 - self.alpha)