"""
Created on 01/10/19

@author: Giuseppe Serna
"""
from tqdm import tqdm
import matplotlib.pyplot as pyplot
import numpy as np
from DataManager.DataManager import DataManager
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from KNN.ItemKNNSimilarityHybridRecommender import ItemKNNSimilarityHybridRecommender
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython
from MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from Base.NonPersonalizedRecommender import TopPop, Random, GlobalEffects
from MatrixFactorization.IALSRecommender import IALSRecommender
import pandas as pd
import scipy.sparse as sp
from DataManager.split_train_validation_leave_k_out import split_train_leave_k_out_user_wise
from pathlib import Path
from Base.Evaluation.Evaluator import EvaluatorHoldout
from Notebooks_utils.data_splitter import train_test_holdout
from Hybrid.HybridColdRecommender import HybridColdRecommender
from KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from Hybrid.HybridZeroRecommender import HybridZeroRecommender
from Hybrid.HybridWarmRecommender import HybridWarmRecommender
from KNN.UserSimilarityHybridRecommender import UserSimilarityHybridRecommender
from Hybrid.HybridGenRecommender import HybridGenRecommender
from Hybrid.HybridNormRecommender import HybridNormRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from FeatureWeighting.CFW_D_Similarity_Linalg import CFW_D_Similarity_Linalg
from Hybrid.HybridNorm1Recommender import HybridNorm1Recommender
from Hybrid.HybridNorm2Recommender import HybridNorm2Recommender
from Hybrid.HybridGen2Recommender import HybridGen2Recommender
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython
data_folder = Path(__file__).parent.absolute()
from FeatureWeighting.User_CFW_D_Similarity_Linalg import  User_CFW_D_Similarity_Linalg
from Hybrid.HybridNorm3Recommender import HybridNorm3Recommender
from MatrixFactorization.ALSRecommender import ALSRecommender
from MatrixFactorization.BPRRecommender import BPRRecommender

data = DataManager()

urm_train, urm_test = split_train_leave_k_out_user_wise(data.get_urm(), temperature='normal')
urm_train, urm_valid = split_train_leave_k_out_user_wise(urm_train, temperature='valid')
evaluator_valid = EvaluatorHoldout(urm_valid, cutoff_list=[10])
evaluator_test = EvaluatorHoldout(urm_test, cutoff_list=[10])

# recommender_1 =
# recommender_2 =
# recommender_3 =
#
#
# result, str_result = evaluator_test.evaluateRecommender(recommender_1)
# print('The Map valid of 1 is : {}'.format(result[10]['MAP']))
#
# result, str_result = evaluator_test.evaluateRecommender(recommender_2)
# print('The Map valid of 2 is : {}'.format(result[10]['MAP']))
#
# result, str_result = evaluator_test.evaluateRecommender(recommender_3)
# print('The Map valid of 3 is : {}'.format(result[10]['MAP']))