"""
Created on 01/10/19

@author: Giuseppe Serna
"""
from tqdm import tqdm
from DataManager.DataManager import DataManager
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython
from MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from Base.NonPersonalizedRecommender import TopPop, Random, GlobalEffects
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
import pandas as pd
from Evaluator.data_splitter import train_test_holdout
from Evaluator.evaluation import evaluate
from DataManager.split_train_validation_leave_k_out import split_train_leave_k_out_user_wise

Data = DataManager()
#urm_train, urm_test = train_test_holdout(Data.get_urm())
urm_train, urm_test = split_train_leave_k_out_user_wise(Data.get_urm(), use_validation_set=False, leave_random_out=True)
MyRecommender = SLIM_BPR_Cython(urm_train)
MyRecommender.fit(topK=21, epochs=399, batch_size=1, lambda_i=0.0444, lambda_j=0.02658285, learning_rate=1e-7)
#MyRecommender.fit(topK=50, shrink=10, normalize=True)
evaluate(urm_test, MyRecommender)


