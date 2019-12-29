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

class HybridGen2Recommender(BaseItemSimilarityMatrixRecommender):
    """ HybridGen2Recommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)

    """

    RECOMMENDER_NAME = "HybridGen2Recommender"

    def __init__(self, urm_train):
        super(HybridGen2Recommender, self).__init__(urm_train)

        urm_train = check_matrix(urm_train.copy(), 'csr')

        icm_weighted = sp.load_npz('Data/icm_weighted.npz')
        data = DataManager()
        ucm_age, ucm_region, ucm_all = data.get_ucm()
        _, _, _, icm_all = data.get_icm()

        recommender_1 = ItemKNNCBFRecommender(urm_train, icm_all)
        recommender_1.fit(shrink=40, topK=20, feature_weighting='BM25')

        recommender_2 = UserKNNCBFRecommender(urm_train, ucm_all)
        recommender_2.fit(shrink=1777, topK=1998, similarity='tversky',
                    feature_weighting='BM25',
                    tversky_alpha=0.1604953616,
                    tversky_beta=0.9862348646)




        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2
        # self.recommender_3 = recommender_3


    def fit(self, alpha=0.06, beta=0.03, gamma=0):
        self.alpha = alpha
        # self.beta = beta
        # self.gamma = gamma

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array)
        # item_weights_3 = self.recommender_3._compute_item_score(user_id_array)

        item_weights = item_weights_1 * self.alpha
        item_weights += item_weights_2 * (1 - self.alpha)
        # item_weights += item_weights_3 * self.gamma

        return item_weights