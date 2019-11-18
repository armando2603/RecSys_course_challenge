import numpy as np


class RandomRecommender(object):

    def __init__(self):
        self.num_items = []

    def fit(self, urm_train):
        self.num_items = urm_train.shape[0]

    def recommend(self, user_id, at=5):
        recommended_items = np.random.choice(self.num_items, at)

        return recommended_items
