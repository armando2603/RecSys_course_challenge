"""
Created on 01/10/19

@author: Giuseppe Serna
"""
from tqdm import tqdm
from DataManager.DataManager import DataManager
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython
from MatrixFactorization.PureSVDRecommender import PureSVDRecommender


from Base.NonPersonalizedRecommender import TopPop, Random, GlobalEffects
import pandas as pd

Data = DataManager()

MyRecommender = ItemKNNCFRecommender(Data.get_urm())
MyRecommender.fit(topK=15, shrink=30)
users = Data.get_target_users()
SecondRecommender = UserKNNCFRecommender(Data.get_urm())
SecondRecommender.fit(topK=150, shrink=10)


recommended_list = []
for user in tqdm(users):
    recommended_items = MyRecommender.recommend(user, 10)
    if len(recommended_items) == 0:
        recommended_items = SecondRecommender.recommend(user, 10)
    items_strings = ' '.join([str(i) for i in recommended_items])
    recommended_list.append(items_strings)

submission = pd.DataFrame(list(zip(users, recommended_list)), columns=['user_id', 'item_list'])
submission.to_csv('Data/my_submission.csv', index=False)

