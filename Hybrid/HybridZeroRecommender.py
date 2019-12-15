from Base.Recommender_utils import check_matrix, similarityMatrixTopK
from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Base.NonPersonalizedRecommender import TopPop
from KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from KNN.UserSimilarityHybridRecommender import UserSimilarityHybridRecommender
import scipy.sparse


class HybridZeroRecommender(BaseItemSimilarityMatrixRecommender):
    """ HybridZeroRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)

    """

    RECOMMENDER_NAME = "HybridZeroRecommender"

    def __init__(self, urm_train, ucm_all):
        super(HybridZeroRecommender, self).__init__(urm_train)

        urm_train = check_matrix(urm_train.copy(), 'csr')

        recommender_1 = TopPop(urm_train)
        recommender_1.fit()

        recommender1 = UserKNNCFRecommender(urm_train)
        recommender1.fit(topK=330, shrink=5)

        # recommender2 = UserKNNCBFRecommender(urm_train, ucm_all)
        # recommender2.fit(shrink=5, topK=400)

        sparse_matrix = scipy.sparse.load_npz('Data/csr_matrix_age.npz')

        recommender_2 = UserSimilarityHybridRecommender(urm_train, recommender1.W_sparse, sparse_matrix)
        recommender_2.fit(alpha=0.93)

        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2
        # self.recommender_3 = recommender_3

    def fit(self, alpha=0.3, beta=0.3, gamma=0):
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