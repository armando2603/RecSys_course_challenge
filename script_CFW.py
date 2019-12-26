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

data = DataManager()

ucm_age, ucm_region, ucm_all = data.get_ucm()

icm_price, icm_asset, icm_sub, icm_all = data.get_icm()

recommender_4 = RP3betaRecommender(data.get_urm())
recommender_4.fit(topK=16, alpha=0.03374950051351756, beta=0.24087176329409027, normalize_similarity=True)

W_sparse_CF = recommender_4.W_sparse

cfw = CFW_D_Similarity_Linalg(URM_train=data.get_urm(),
                              ICM=icm_all.copy(),
                              S_matrix_target=W_sparse_CF
                              )

cfw.fit(topK=26, add_zeros_quota=0.453567, normalize_similarity=True)

weights = sps.diags(cfw.D_best)

icm_weighted = icm_all.dot(weights)

sps.save_npz("Data/icm_weighted.npz", icm_weighted)



