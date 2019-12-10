from Base.Recommender_utils import check_matrix, similarityMatrixTopK
from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Base.NonPersonalizedRecommender import TopPop

class HybridWarmRecommender(BaseItemSimilarityMatrixRecommender):
    """ HybridWarmRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)

    """

    RECOMMENDER_NAME = "HybridWarmRecommender"

    def __init__(self, URM_train):
        super(HybridWarmRecommender, self).__init__(URM_train)

        recommender_1 = ItemKNNCFRecommender(URM_train)
        recommender_1.fit(topK=20, shrink=30)

        # recommender_2 = SLIM_BPR_Cython(URM_train)
        # recommender_2.fit(epochs=60, lambda_i=0.0297, lambda_j=0.0188, learning_rate=0.008083, topK=202)
        #
        # recommender_3 = TopPop(URM_train)
        # recommender_3.fit()


        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.recommender_1 = recommender_1
        # self.recommender_2 = recommender_2
        # self.recommender_3 = recommender_3

    def fit(self, alpha=.3, beta=.03, gamma=0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _compute_item_score(self, user_id_array, items_to_compute):
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
        # item_weights_2 = self.recommender_2._compute_item_score(user_id_array)
        # item_weights_3 = self.recommender_3._compute_item_score(user_id_array)

        item_weights = item_weights_1 * self.alpha
        # item_weights += item_weights_2 * self.beta
        # item_weights += item_weights_3 * self.gamma

        return item_weights