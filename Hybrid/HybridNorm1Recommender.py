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
from Hybrid.HybridGen2Recommender import HybridGen2Recommender

class HybridNorm1Recommender(BaseItemSimilarityMatrixRecommender):
    """ HybridNormRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)

    """

    RECOMMENDER_NAME = "HybridNorm1Recommender"

    def __init__(self, urm_train):
        super(HybridNorm1Recommender, self).__init__(urm_train)

        urm_train = check_matrix(urm_train.copy(), 'csr')

        recommender_1 = HybridGen2Recommender(urm_train)
        recommender_1.fit()

        recommender_4 = ItemKNNCFRecommender(urm_train)
        recommender_4.fit(topK=5, shrink=500, feature_weighting='BM25', similarity='tversky', normalize=False,
                        tversky_alpha=0.0, tversky_beta=1.0)

        recommender_3 = UserKNNCFRecommender(urm_train)
        recommender_3.fit(topK=697, shrink=1000, feature_weighting='TF-IDF', similarity='tversky', normalize=False, tversky_alpha=1.0, tversky_beta=1.0)

        recommender_2 = RP3betaRecommender(urm_train)
        recommender_2.fit(topK=16, alpha=0.03374950051351756, beta=0.24087176329409027, normalize_similarity=True)

        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2
        self.recommender_3 = recommender_3
        self.recommender_4 = recommender_4

    def fit(self, alpha=0.35, beta=0.03, gamma=0.9, phi=0, psi=0):
        # alpha=0.2, beta=0.8, gamma=0.012, phi=1.2
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array)
        item_weights_3 = self.recommender_3._compute_item_score(user_id_array)
        item_weights_4 = self.recommender_4._compute_item_score(user_id_array)

        item_weights = item_weights_1 * self.alpha
        item_weights += item_weights_2 * (1 - self.alpha)
        item_weights += item_weights_3 * self.beta
        item_weights += item_weights_4 * self.gamma


        return item_weights