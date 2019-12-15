from Base.Recommender_utils import check_matrix, similarityMatrixTopK
from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from lightfm import LightFM
import numpy as np

class LighFMRecommender(BaseItemSimilarityMatrixRecommender):
    """ LightFMRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)

    """

    RECOMMENDER_NAME = "LightFMRecommender"

    def __init__(self, urm_train, no_components=10, k=5, n=10,
                 learning_schedule='adagrad',
                 loss='logistic',
                 learning_rate=0.05, rho=0.95, epsilon=1e-6,
                 item_alpha=0.0, user_alpha=0.0, max_sampled=10,
                 random_state=None):
        super(LighFMRecommender, self).__init__(urm_train)

        self.urm_train = urm_train
        self.ucm = None
        self.icm = None
        self.model = LightFM(no_components=no_components, k=k, n=n,
                             learning_schedule=learning_schedule,
                             loss=loss,
                             learning_rate=learning_rate, rho=rho, epsilon=epsilon,
                             item_alpha=item_alpha, user_alpha=user_alpha, max_sampled=max_sampled,
                             random_state=random_state,
                             )

    def fit(self,
            user_features=None, item_features=None,
            sample_weight=None,
            epochs=1, num_threads=1, verbose=True):

        self.model.fit(interactions=self.urm_train,
            user_features=user_features, item_features=item_features,
            sample_weight=sample_weight,
            epochs=epochs, num_threads=num_threads, verbose=verbose)

        self.ucm = user_features
        self.icm = item_features

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        n_items = self.urm_train.shape[1]
        item_weights = np.zeros(shape=[len(user_id_array), n_items])
        index = 0
        for user_id in user_id_array:
            user_score = self.model.predict(user_id,
                                            np.arange(n_items),
                                            item_features=self.icm,
                                            user_features=self.ucm,
                                            num_threads=3)
            item_weights[index] = user_score
            index += 1
        return item_weights

    def fit_partial(self,
                    user_features=None, item_features=None,
                    sample_weight=None,
                    epochs=1, num_threads=1, verbose=True):

        self.model.fit_partial(interactions=self.urm_train,
                               user_features=user_features, item_features=item_features,
                               sample_weight=sample_weight,
                               epochs=epochs, num_threads=num_threads, verbose=verbose)
        self.ucm = user_features
        self.icm = item_features

