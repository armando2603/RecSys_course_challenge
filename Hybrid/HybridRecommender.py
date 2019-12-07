"""
Created on 01/10/19

@author: Giuseppe Serna
"""
from Base.Recommender_utils import check_matrix, similarityMatrixTopK
from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython


class HybridRecommender(BaseItemSimilarityMatrixRecommender):
    """ HybridRecommender
    Hybrid of two similarities S = S1*alpha + S2*(1-alpha)

    """

    RECOMMENDER_NAME = "HybridRecommender"


    def __init__(self, URM_train, sparse_weights=True):
        super(HybridRecommender, self).__init__(URM_train)


        recommender1 = ItemKNNCFRecommender(URM_train)
        recommender1.fit(topK=40, shrink=30)
        Similarity_1 = recommender1.W_sparse

        recommender2 = SLIM_BPR_Cython(URM_train)
        recommender2.fit(epochs=100, lambda_i=0.3499518, lambda_j=0.3897793, learning_rate=0.002391)
        Similarity_2 = recommender2.W_sparse

        if Similarity_1.shape != Similarity_2.shape:
            raise ValueError("HybridRecommender: similarities have different size, S1 is {}, S2 is {}".format(
                Similarity_1.shape, Similarity_2.shape
            ))

        # CSR is faster during evaluation
        self.Similarity_1 = check_matrix(Similarity_1.copy(), 'csr')
        self.Similarity_2 = check_matrix(Similarity_2.copy(), 'csr')


    def fit(self, topK=100, alpha = 0.5):

        self.topK = topK
        self.alpha = alpha

        W = self.Similarity_1*self.alpha + self.Similarity_2*(1-self.alpha)
        self.W_sparse = similarityMatrixTopK(W, k=self.topK).tocsr()
