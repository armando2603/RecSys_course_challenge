"""
Created on 04/12/19

@author: Giuseppe Serna
"""
from DataManager.DataManager import DataManager
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from DataManager.split_train_validation_leave_k_out import split_train_leave_k_out_user_wise
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython
from Notebooks_utils.data_splitter import train_test_holdout
from Base.Evaluation.Evaluator import EvaluatorHoldout
from MatrixFactorization.IALSRecommender import IALSRecommender
from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import  RP3betaRecommender
from ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from skopt.space import Real, Integer, Categorical
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
import numpy as np


Data = DataManager()


urm_train, urm_test = split_train_leave_k_out_user_wise(Data.get_urm(), threshold=10, temperature='normal')
urm_train, urm_valid = split_train_leave_k_out_user_wise(urm_train, threshold=10, temperature='valid')
evaluator_valid = EvaluatorHoldout(urm_valid, cutoff_list=[10])
evaluator_test = EvaluatorHoldout(urm_test, cutoff_list=[10])

recommender = RP3betaRecommender

parameterSearch = SearchBayesianSkopt(recommender,
                                 evaluator_validation=evaluator_valid,
                                 evaluator_test=evaluator_test)

# earlystopping_keywargs = {"validation_every_n": 5,
#                               "stop_on_validation": True,
#                               "evaluator_object": evaluator_valid,
#                               "lower_validations_allowed": 2,
#                               "validation_metric": "MAP"
#                           }

hyperparameters_range_dictionary = {}
hyperparameters_range_dictionary["topK"] = Integer(5, 200)
hyperparameters_range_dictionary["alpha"] = Real(0, 2)
hyperparameters_range_dictionary["beta"] = Real(0, 2)
# hyperparameters_range_dictionary["similarity"] = Categorical(["cosine", "jaccard", "adjusted"])
hyperparameters_range_dictionary["normalize_similarity"] = Categorical([True, False])


recommender_input_args = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS=[urm_train],
    CONSTRUCTOR_KEYWORD_ARGS={},
    FIT_POSITIONAL_ARGS=[],
    FIT_KEYWORD_ARGS={}
)

output_folder_path = "result_experiments/"

import os

# If directory does not exist, create
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

n_cases = 200
metric_to_optimize = "MAP"
parameterSearch.search(recommender_input_args,
                       parameter_search_space = hyperparameters_range_dictionary,
                       n_cases = n_cases,
                       n_random_starts = 10,
                       save_model = "best",
                       output_folder_path = output_folder_path,
                       output_file_name_root = recommender.RECOMMENDER_NAME,
                       metric_to_optimize = metric_to_optimize
                      )

