from Base.Recommender_utils import check_matrix, similarityMatrixTopK
from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Base.NonPersonalizedRecommender import TopPop
from KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from KNN.UserSimilarityHybridRecommender import UserSimilarityHybridRecommender
import scipy.sparse as sps
from DataManager.DataManager import DataManager
import numpy as np
from sklearn.preprocessing import normalize
from pathlib import Path

class HybridGen2Recommender(BaseItemSimilarityMatrixRecommender):
    """ HybridGen2Recommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)

    """

    RECOMMENDER_NAME = "HybridGen2Recommender"

    def __init__(self, urm_train, eurm=False):
        super(HybridGen2Recommender, self).__init__(urm_train)
        self.eurm = eurm
        self.data_folder = Path(__file__).parent.parent.absolute()

        self.num_users = urm_train.shape[0]

        urm_train = check_matrix(urm_train.copy(), 'csr')



        data = DataManager()


        if Path(self.data_folder / 'Data/icm_sparse.npz').is_file() and self.eurm:
            print("Previous icm_sparse found")
        else:
            _, _, _, icm_all = data.get_icm()

            args = {'topK': 6, 'shrink': 5, 'feature_weighting': 'TF-IDF', 'similarity': 'cosine', 'normalize': False}

            recommender_1 = ItemKNNCBFRecommender(urm_train, icm_all)
            recommender_1.fit(**args)
            self.recommender_1 = recommender_1


        if Path(self.data_folder / 'Data/ucm_sparse.npz').is_file() and self.eurm:
            print("Previous ucm_sparse found")
        else:
            ucm_age, ucm_region, ucm_all = data.get_ucm()
            recommender_2 = UserKNNCBFRecommender(urm_train, ucm_all)
            recommender_2.fit(shrink=1777, topK=1998, similarity='tversky',
                        feature_weighting='BM25',
                        tversky_alpha=0.1604953616,
                        tversky_beta=0.9862348646)
            self.recommender_2 = recommender_2

        if self.eurm:
            beta = 0.6
            if Path(self.data_folder / 'Data/icm_sparse.npz').is_file():
                self.score_matrix_1 = sps.load_npz(self.data_folder / 'Data/icm_sparse.npz')
            else:
                print("icm_sparse not found, create new one...")
                self.score_matrix_1 = self.recommender_1._compute_item_matrix_score(np.arange(self.num_users))
                user_score_matrix_1 = normalize(self.score_matrix_1, norm='max', axis=1)
                item_score_matrix_1 = normalize(self.score_matrix_1.tocsc(), norm='max', axis=0)
                self.score_matrix_1 = item_score_matrix_1 * beta + user_score_matrix_1.tocsr() * (1-beta)
                sps.save_npz(self.data_folder / 'Data/icm_sparse.npz', self.score_matrix_1)

            if Path(self.data_folder / 'Data/ucm_sparse.npz').is_file():
                self.score_matrix_2 = sps.load_npz(self.data_folder / 'Data/ucm_sparse.npz')
            else:
                print("ucm_sparse not found, create new one...")
                self.score_matrix_2 = self.recommender_2._compute_item_matrix_score(np.arange(self.num_users))
                user_score_matrix_2 = normalize(self.score_matrix_2, norm='max', axis=1)
                item_score_matrix_2 = normalize(self.score_matrix_2.tocsc(), norm='max', axis=0)
                self.score_matrix_2 = item_score_matrix_2 * beta + user_score_matrix_2.tocsr() * (1 - beta)
                sps.save_npz(self.data_folder / 'Data/ucm_sparse.npz', self.score_matrix_2)

    def fit(self, alpha=0.11):
        self.alpha = alpha

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        if self.eurm:

            item_weights_1 = self.score_matrix_1[user_id_array]
            item_weights_2 = self.score_matrix_2[user_id_array]
        else:

            item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
            item_weights_2 = self.recommender_2._compute_item_score(user_id_array)

        item_weights = item_weights_1 * self.alpha
        item_weights += item_weights_2 * (1 - self.alpha)
        # item_weights = normalize(item_weights, norm='max', axis=1)
        # from sklearn.preprocessing import normalize
        # item_scores = normalize(item_scores, norm='max', axis=1)
        return item_weights

    def _compute_item_matrix_score(self, user_id_array, items_to_compute=None):
        return self.score_matrix_1 * self.alpha + self.score_matrix_2 * (1 - self.alpha)
