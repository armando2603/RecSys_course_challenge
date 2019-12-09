from Base.Recommender_utils import check_matrix, similarityMatrixTopK
from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Base.NonPersonalizedRecommender import TopPop

class Hybrid2PredRecommender(BaseItemSimilarityMatrixRecommender):
    """ HybridPredRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)

    """

    RECOMMENDER_NAME = "Hybrid2PredRecommender"

    def __init__(self, urm_train):
        super(Hybrid2PredRecommender, self).__init__(urm_train)

        urm_train = check_matrix(urm_train.copy(), 'csr')

        recommender_1 = ItemKNNCFRecommender(urm_train)
        recommender_1.fit(topK=20, shrink=30)

        recommender_2 = TopPop(urm_train)
        recommender_2.fit()

        # recommender_3 = UserKNNCFRecommender(URM_train)
        # recommender_3.fit(shrink=4, topK=400)

        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2
        # self.recommender_3 = recommender_3

    def fit(self, alpha=.3, beta=.03, gamma=0.3):
        self.alpha = alpha
        self.beta = beta
        # self.gamma = gamma

    def _compute_item_score(self, user_id_array, items_to_compute):
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array)
        # item_weights_3 = self.recommender_3._compute_item_score(user_id_array)

        item_weights = item_weights_1 * self.alpha
        item_weights += item_weights_2 * self.beta

        return item_weights