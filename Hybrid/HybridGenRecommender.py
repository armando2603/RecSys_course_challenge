from Base.Recommender_utils import check_matrix, similarityMatrixTopK
from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Base.NonPersonalizedRecommender import TopPop
from KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from KNN.UserSimilarityHybridRecommender import UserSimilarityHybridRecommender
from DataManager.DataManager import DataManager
import numpy as np
from sklearn.preprocessing import normalize
from pathlib import Path
import scipy.sparse as sps



class HybridGenRecommender(BaseItemSimilarityMatrixRecommender):
    """ HybridGenRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)

    """

    RECOMMENDER_NAME = "HybridGenRecommender"

    def __init__(self, urm_train, eurm=False):
        super(HybridGenRecommender, self).__init__(urm_train)

        self.data_folder = Path(__file__).parent.parent.absolute()

        self. eurm = eurm

        self.num_users = urm_train.shape[0]
        data = DataManager()

        urm_train = check_matrix(urm_train.copy(), 'csr')
        icm_price, icm_asset, icm_sub, icm_all = data.get_icm()
        ucm_age, ucm_region, ucm_all = data.get_ucm()

        recommender_1 = ItemKNNCBFRecommender(urm_train, icm_all)
        recommender_1.fit(shrink=40, topK=20, feature_weighting='BM25')

        recommender_2 = UserKNNCBFRecommender(urm_train, ucm_all)
        recommender_2.fit(shrink=500, topK=1600, normalize=True)

        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2
        # self.recommender_3 = recommender_3

    def fit(self, alpha=0.2, beta=0.03, gamma=0):
        # original 0.2
        self.alpha = alpha
        # self.beta = beta
        # self.gamma = gamma

        self.score_matrix_1 = None
        self.score_matrix_2 = None

        if self.eurm:

            if Path(self.data_folder / 'Data/icm_sparse.npz').is_file():
                print("Previous icm_sparse found")
                self.score_matrix_1 = sps.load_npz(self.data_folder / 'Data/icm_sparse.npz')
            else:
                print("icm_sparse not found, create new one...")
                self.score_matrix_1 = self.recommender_1._compute_item_matrix_score(np.arange(self.num_users))
                item_score_matrix_1 = normalize(self.score_matrix_1, norm='max', axis=1)
                user_score_matrix_1 = normalize(self.score_matrix_1.tocsc(), norm='max', axis=0)
                self.score_matrix_1 = item_score_matrix_1 * 0.6 + user_score_matrix_1.tocsr() * 0.4
                sps.save_npz(self.data_folder / 'Data/icm_sparse.npz', self.score_matrix_1)

            if Path(self.data_folder / 'Data/ucm_sparse.npz').is_file():
                print("Previous ucm_sparse found")
                self.score_matrix_2 = sps.load_npz(self.data_folder / 'Data/ucm_sparse.npz')
            else:
                print("ucm_sparse not found, create new one...")
                self.score_matrix_2 = self.recommender_2._compute_item_matrix_score(np.arange(self.num_users))
                item_score_matrix_2 = normalize(self.score_matrix_2, norm='max', axis=1)
                user_score_matrix_2 = normalize(self.score_matrix_2.tocsc(), norm='max', axis=0)
                self.score_matrix_2 = item_score_matrix_2 * 0.6 + user_score_matrix_2.tocsr() * 0.4
                sps.save_npz(self.data_folder / 'Data/ucm_sparse.npz', self.score_matrix_2)


    def _compute_item_score(self, user_id_array, items_to_compute=None):
        if self.eurm:

            item_weights_1 = self.score_matrix_1[user_id_array].toarray()
            item_weights_2 = self.score_matrix_2[user_id_array].toarray()
        else:

            item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
            item_weights_2 = self.recommender_2._compute_item_score(user_id_array)

        item_weights = item_weights_1 * self.alpha
        item_weights += item_weights_2 * (1 - self.alpha)

        return item_weights

    def _compute_item_matrix_score(self, user_id_array, items_to_compute=None):
        return self.score_matrix_1 * self.alpha + self.score_matrix_2 * (1 - self.alpha)
