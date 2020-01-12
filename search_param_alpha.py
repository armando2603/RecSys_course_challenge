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

num_test = 4
x_tick = np.arange(start=0, stop=0.010001, step=0.001)

def single_test(urm_train, urm_test, urm_valid, x_tick):
    evaluator_valid = EvaluatorHoldout(urm_valid, cutoff_list=[10], verbose=False)

    MAP_per_k_valid = []

    recommender = HybridNorm3Recommender(urm_train)

    for alpha in tqdm(x_tick):
        recommender.fit(beta=alpha)

        result_dict, res_str = evaluator_valid.evaluateRecommender(recommender)
        MAP_per_k_valid.append(result_dict[10]["MAP"])

    return MAP_per_k_valid


data = DataManager()

my_input = []

for i in np.arange(num_test):

    urm_train, urm_test = split_train_leave_k_out_user_wise(data.get_urm(), temperature='normal')
    urm_train, urm_valid = split_train_leave_k_out_user_wise(urm_train, temperature='valid')
    # urm_test = data.create_test_warm_users(urm_test, threshold=5)
    urm_valid = data.create_test_warm_users(urm_valid, threshold=3)
    my_input.append([urm_train, urm_test, urm_valid, x_tick])

from multiprocessing import Pool

pool = Pool(num_test)
res = pool.starmap(single_test, my_input)
pool.close()
pool.join()

count = 0
for value in x_tick:
    count += 1

valid_final = np.zeros(count)
for valid in res:
    valid_final += valid

valid_final = valid_final/count

best_index = np.argmax(valid_final)
best_alpha = x_tick[best_index]

pyplot.plot(x_tick, valid_final)
pyplot.ylabel('MAP_valid')
pyplot.xlabel('alpha')
pyplot.savefig('newfig')

print('The best alpha is : {}'.format(best_alpha))

def single_test(urm_train, urm_test, urm_valid, x_tick):
    evaluator_test = EvaluatorHoldout(urm_test, cutoff_list=[10])

    recommender = HybridNorm3Recommender(urm_train)
    recommender.fit(beta=best_alpha)

    result, str_result = evaluator_test.evaluateRecommender(recommender)
    return result[10]['MAP']

pool = Pool(num_test)
res = pool.starmap(single_test, my_input)
pool.close()
pool.join()

res = np.array(res)
print('Il MAP del test è : {}'.format(res.mean()))



