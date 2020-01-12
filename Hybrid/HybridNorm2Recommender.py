from Base.Recommender_utils import check_matrix, similarityMatrixTopK
from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Base.NonPersonalizedRecommender import TopPop
from KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from KNN.UserSimilarityHybridRecommender import UserSimilarityHybridRecommender
from Hybrid.HybridGenRecommender import HybridGenRecommender
from Hybrid.HybridGen2Recommender import HybridGen2Recommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from MatrixFactorization.ALSRecommender import ALSRecommender
from Hybrid.HybridNorm1Recommender import HybridNorm1Recommender
import numpy as np
from sklearn.preprocessing import normalize
import scipy.sparse as sps
from pathlib import Path

class HybridNorm2Recommender(BaseItemSimilarityMatrixRecommender):
    """ HybridNormRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)

    """

    RECOMMENDER_NAME = "HybridNorm2Recommender"

    def __init__(self, urm_train, eurm=False):
        super(HybridNorm2Recommender, self).__init__(urm_train)
        self.data_folder = Path(__file__).parent.parent.absolute()
        self.eurm = eurm
        self.num_users = urm_train.shape[0]

        urm_train = check_matrix(urm_train.copy(), 'csr')

        recommender_2 = ItemKNNCFRecommender(urm_train)
        recommender_2.fit(topK=5, shrink=500, feature_weighting='BM25', similarity='tversky', normalize=False, tversky_alpha=0.0, tversky_beta=1.0)

        recommender_3 = UserKNNCFRecommender(urm_train)
        recommender_3.fit(shrink=2, topK=600, normalize=True)
        # recommender_3 = UserKNNCFRecommender(urm_train)
        # recommender_3.fit(topK=697, shrink=1000, feature_weighting='TF-IDF', similarity='tversky', normalize=False,
        #                   tversky_alpha=1.0, tversky_beta=1.0)

        recommender_4 = RP3betaRecommender(urm_train)
        recommender_4.fit(topK=16, alpha=0.03374950051351756, beta=0.24087176329409027, normalize_similarity=True)

        recommender_5 = SLIM_BPR_Cython(urm_train)
        recommender_5.fit(lambda_i=0.0926694015, lambda_j=0.001697250, learning_rate=0.002391, epochs=65, topK=200)

        recommender_6 = ALSRecommender(urm_train)
        recommender_6.fit(alpha=5, iterations=40, reg=0.3)


        self.recommender_2 = recommender_2
        self.recommender_3 = recommender_3
        self.recommender_4 = recommender_4
        self.recommender_5 = recommender_5
        self.recommender_6 = recommender_6

        if self.eurm:

            self.score_matrix_1 = sps.load_npz(self.data_folder / 'Data/icm_sparse.npz')
            self.score_matrix_2 = self.recommender_2._compute_item_matrix_score(np.arange(self.num_users))
            self.score_matrix_3 = self.recommender_3._compute_item_matrix_score(np.arange(self.num_users))
            self.score_matrix_4 = self.recommender_4._compute_item_matrix_score(np.arange(self.num_users))
            self.score_matrix_5 = self.recommender_5._compute_item_matrix_score(np.arange(self.num_users))
            self.score_matrix_6 = self.recommender_6._compute_item_score(np.arange(self.num_users))
            self.score_matrix_7 = sps.load_npz(self.data_folder / 'Data/ucm_sparse.npz')

            self.score_matrix_1 = normalize(self.score_matrix_1, norm='l2', axis=1)
            self.score_matrix_2 = normalize(self.score_matrix_2, norm='l2', axis=1)
            self.score_matrix_3 = normalize(self.score_matrix_3, norm='l2', axis=1)
            self.score_matrix_4 = normalize(self.score_matrix_4, norm='l2', axis=1)
            self.score_matrix_5 = normalize(self.score_matrix_5, norm='l2', axis=1)
            self.score_matrix_6 = normalize(self.score_matrix_6, norm='l2', axis=1)
            self.score_matrix_7 = normalize(self.score_matrix_7, norm='l2', axis=1)


            # user_score_matrix_2 = normalize(self.score_matrix_2.tocsc(), norm='max', axis=0)
            # user_score_matrix_3 = normalize(self.score_matrix_3.tocsc(), norm='max', axis=0)
            # user_score_matrix_4 = normalize(self.score_matrix_4.tocsc(), norm='max', axis=0)
            # user_score_matrix_5 = normalize(self.score_matrix_5.tocsc(), norm='max', axis=0)
            # user_score_matrix_6 = normalize(self.score_matrix_6.tocsc(), norm='max', axis=0)
            #
            # # perform a weighted sum with alpha = 0.6 as the paper do
            #
            # self.score_matrix_2 = item_score_matrix_2 * 0.6 + user_score_matrix_2.tocsr() * 0.4
            # self.score_matrix_3 = item_score_matrix_3 * 0.6 + user_score_matrix_3.tocsr() * 0.4
            # self.score_matrix_4 = item_score_matrix_4 * 0.6 + user_score_matrix_4.tocsr() * 0.4
            # self.score_matrix_5 = item_score_matrix_5 * 0.6 + user_score_matrix_5.tocsr() * 0.4
            # self.score_matrix_6 = item_score_matrix_6 * 0.6 + user_score_matrix_6.tocsr() * 0.4



    def fit(self, alpha=0.2, beta=0.8, gamma=0.012, phi=1.2, psi=0, li=0, mi=0):
        # alpha=0.2, beta=0.8, gamma=0.012, phi=1.2
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.phi = phi
        self.psi = psi
        self.li = li
        self.mi = mi


    def _compute_item_score(self, user_id_array, items_to_compute=None):

        if self.eurm:
            item_weights_1 = self.score_matrix_1[user_id_array].toarray()
            item_weights_2 = self.score_matrix_2[user_id_array].toarray()
            item_weights_3 = self.score_matrix_3[user_id_array].toarray()
            item_weights_4 = self.score_matrix_4[user_id_array].toarray()
            item_weights_5 = self.score_matrix_5[user_id_array].toarray()
            item_weights_6 = self.score_matrix_6[user_id_array]
            item_weights_7 = self.score_matrix_7[user_id_array].toarray()

        else:
            item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
            item_weights_2 = self.recommender_2._compute_item_score(user_id_array)
            item_weights_3 = self.recommender_3._compute_item_score(user_id_array)
            item_weights_4 = self.recommender_4._compute_item_score(user_id_array)
            item_weights_5 = self.recommender_5._compute_item_score(user_id_array)
            item_weights_6 = self.recommender_6._compute_item_score(user_id_array)

        item_weights = item_weights_1 * self.alpha
        item_weights += item_weights_2 * self.beta
        item_weights += item_weights_3 * self.gamma
        item_weights += item_weights_4 * self.phi
        item_weights += item_weights_5 * self.psi
        item_weights += item_weights_6 * self.li


        return item_weights
