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
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
import pandas as pd
from DataManager.split_train_validation_leave_k_out import split_train_leave_k_out_user_wise
from FeatureWeighting.CFW_D_Similarity_Linalg import CFW_D_Similarity_Linalg
from FeatureWeighting.User_CFW_D_Similarity_Linalg import User_CFW_D_Similarity_Linalg
from Utils.s_plus import dot_product
import scipy.sparse as sps
from Base.Evaluation.Evaluator import EvaluatorHoldout
from GraphBased.RP3betaRecommender import RP3betaRecommender
from ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from skopt.space import Real, Integer, Categorical
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs

Data = DataManager()

ucm_age, ucm_region, ucm_all = Data.get_ucm()

icm_price, icm_asset, icm_sub, icm_all = Data.get_icm()

urm_train, urm_test = split_train_leave_k_out_user_wise(Data.get_urm(), temperature='normal')
urm_train, urm_valid = split_train_leave_k_out_user_wise(urm_train, temperature='valid')
evaluator_test = EvaluatorHoldout(urm_test, cutoff_list=[10])
evaluator_valid = EvaluatorHoldout(urm_valid, cutoff_list=[10])

recommender_4 = RP3betaRecommender(urm_train)
recommender_4.fit(topK=16, alpha=0.03374950051351756, beta=0.24087176329409027, normalize_similarity=True)


# recommender_3 = UserKNNCFRecommender(urm_train)
# recommender_3.fit(shrink=2, topK=600, normalize=True)

W_sparse_CF = recommender_4.W_sparse

recommender_class = CFW_D_Similarity_Linalg

parameterSearch = SearchBayesianSkopt(recommender_class,
                                 evaluator_validation=evaluator_valid,
                                 evaluator_test=evaluator_test)


hyperparameters_range_dictionary = {}
hyperparameters_range_dictionary["topK"] = Integer(5, 1000)
hyperparameters_range_dictionary["add_zeros_quota"] = Real(low = 0, high = 1, prior = 'uniform')
hyperparameters_range_dictionary["normalize_similarity"] = Categorical([True, False])


recommender_input_args = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS = [urm_train, icm_asset, W_sparse_CF],
    CONSTRUCTOR_KEYWORD_ARGS = {},
    FIT_POSITIONAL_ARGS = [],
    FIT_KEYWORD_ARGS = {}
)


output_folder_path = "result_experiments/"

import os

# If directory does not exist, create
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

n_cases = 50
metric_to_optimize = "MAP"


# Clone data structure to perform the fitting with the best hyperparameters on train + validation data
recommender_input_args_last_test = recommender_input_args.copy()
recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = urm_train + urm_valid


parameterSearch.search(recommender_input_args,
                       recommender_input_args_last_test = recommender_input_args_last_test,
                       parameter_search_space = hyperparameters_range_dictionary,
                       n_cases = n_cases,
                       n_random_starts = int(n_cases/3),
                       save_model = "no",
                       output_folder_path = output_folder_path,
                       output_file_name_root = recommender_class.RECOMMENDER_NAME + ' asset',
                       metric_to_optimize = metric_to_optimize
                      )