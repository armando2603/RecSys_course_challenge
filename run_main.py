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

test = True
threshold = 2
temperature = 'normal'
Data = DataManager()
urm_train = Data.get_urm()

valid = False

multiple_test = False
num_test = 1

if multiple_test:
    # res = np.zeros(num_test)
    # for i in np.arange(num_test):
    #     Data = DataManager()
    #     urm_train = Data.get_urm()
    #     urm_train, urm_test = split_train_leave_k_out_user_wise(urm_train, threshold=threshold, temperature=temperature)
    #     urm_train, urm_valid = split_train_leave_k_out_user_wise(urm_train, threshold=threshold, temperature='valid')
    #     evaluator_valid = EvaluatorHoldout(urm_valid, cutoff_list=[10])
    #     evaluator_test = EvaluatorHoldout(urm_test, cutoff_list=[10])
    #
    #     recommender = HybridNormRecommender(urm_train)
    #     recommender.fit()
    #
    #     result, str_result = evaluator_test.evaluateRecommender(recommender)
    #     res[i] = result[10]['MAP']
    # print(res)
    # print('Il MAP del test è : {}'.format(res.mean()))


    def single_test(urm_train, urm_test, urm_valid):
        evaluator_valid = EvaluatorHoldout(urm_valid, cutoff_list=[10])
        evaluator_test = EvaluatorHoldout(urm_test, cutoff_list=[10])

        recommender = HybridNormRecommender(urm_train)
        recommender.fit()

        result, str_result = evaluator_test.evaluateRecommender(recommender)
        # result, str_result = evaluator_valid.evaluateRecommender(recommender)
        # res[num_test] = result[10]['MAP']
        return result[10]['MAP']

    my_input = []

    for i in np.arange(num_test):

        urm_train, urm_test = split_train_leave_k_out_user_wise(Data.get_urm(), threshold=threshold, temperature=temperature)
        urm_train, urm_valid = split_train_leave_k_out_user_wise(urm_train, threshold=threshold, temperature='valid')
        my_input.append([urm_train, urm_test, urm_valid])

    from multiprocessing import Pool

    pool = Pool(5)
    res = pool.starmap(single_test, my_input)
    pool.close()
    pool.join()

    print(res)
    res = np.array(res)
    print('Il MAP del test è : {}'.format(res.mean()))
else:


    sparse_matrix = sp.load_npz(data_folder / 'Data/csr_matrix_age.npz')

    if test:
        urm_train, urm_test = split_train_leave_k_out_user_wise(urm_train, threshold=threshold, temperature=temperature)

        if valid:
            urm_train, urm_valid = split_train_leave_k_out_user_wise(urm_train, threshold=threshold, temperature='valid')
            evaluator_valid = EvaluatorHoldout(urm_valid, cutoff_list=[10])
        evaluator_test = EvaluatorHoldout(urm_test, cutoff_list=[10])
    else:
        urm_train = Data.get_urm()

    zero_recommender = None
    cold_recommender = None
    warm_recommender = None
    normal_recommender = None

    if temperature == 'cold' and test is True:
        # icm_weighted = sp.load_npz(data_folder / 'Data/icm_weighted.npz')
        icm_price, icm_asset, icm_sub, icm_all = Data.get_icm()
        ucm_age, ucm_region, ucm_all = Data.get_ucm()


        # recommender = ItemKNNCFRecommender(urm_train)
        # recommender.fit(shrink=29, topK=6, similarity='jaccard')

        # recommender = HybridNormRecommender(urm_train, ucm_all, icm_all)
        # recommender.fit()

        # recommender = RP3betaRecommender(urm_train)
        # recommender.fit(topK=16, alpha=0.03374950051351756, beta=0.24087176329409027, normalize_similarity=True)

        # x_tick = np.arange(start=0.18, stop=0.25001, step=0.01)
        # print(len(x_tick))
        # MAP_per_k_test = []
        #
        # MAP_per_k_valid = []
        #
        # recommender = HybridColdRecommender(urm_train, ucm_all, icm_weighted)
        #
        #
        # best_alpha = 0
        # best_res = 0
        # for alpha in tqdm(x_tick):
        #
        #     recommender.fit(alpha=alpha)
        #
        #     result_dict, res_str = evaluator_valid.evaluateRecommender(recommender)
        #     MAP_per_k_valid.append(result_dict[10]["MAP"])
        #
        #     if result_dict[10]["MAP"] > best_res:
        #         best_alpha = alpha
        #         best_res = result_dict[10]["MAP"]
        #
        # print('The alpha is {}'.format(best_alpha))
        #
        #
        #
        # pyplot.plot(x_tick, MAP_per_k_valid)
        # pyplot.ylabel('MAP_valid')

        #
        # recommender.fit(alpha=best_alpha)

        recommender = HybridNormRecommender(urm_train)
        recommender.fit()

        # recommender = UserKNNCBFRecommender(urm_train, ucm_all)
        # recommender.fit(shrink=500, topK=1600, normalize=True)

        cold_recommender = recommender

    if temperature == 'zero' or test is False:
        # icm_price, icm_asset, icm_sub, icm_all = Data.get_icm()
        ucm_age, ucm_region, ucm_all = Data.get_ucm()

        # x_tick = np.arange(start=0, stop=1.0001, step=0.1)
        # MAP_per_k_test = []
        #
        # MAP_per_k_valid = []
        #
        # recommender = HybridZeroRecommender(urm_train)
        # best_alpha = 0
        # best_res = 0
        # for alpha in x_tick:
        #
        #     recommender.fit(alpha=alpha)
        #
        #     result_dict, res_str = evaluator_valid.evaluateRecommender(recommender)
        #     MAP_per_k_valid.append(result_dict[10]["MAP"])
        #
        #     if result_dict[10]["MAP"] > best_res:
        #         best_alpha = alpha
        #         best_res = result_dict[10]["MAP"]
        #
        # print('The alpha is {}'.format(best_alpha))
        #
        # pyplot.plot(x_tick, MAP_per_k_valid)
        # pyplot.ylabel('MAP_valid')
        # pyplot.xlabel('alpha')
        # pyplot.show()
        #
        # recommender = UserKNNCBFRecommender(urm_train, ucm_all)
        # recommender.fit(shrink=500, topK=1600, similarity='tversky')

        recommender = UserKNNCBFRecommender(urm_train, ucm_all)
        recommender.fit(shrink=1777, topK=1998, similarity='tversky',
                        feature_weighting='BM25',
                        tversky_alpha=0.1604953616,
                        tversky_beta=0.9862348646)

        # recommender = UserKNNCBFRecommender(urm_train, ucm_all)
        # recommender.fit(shrink=979, topK=1677, similarity='jaccard')
        #
        # recommender = UserKNNCBFRecommender(urm_train, ucm_all)
        # recommender.fit(shrink=1000, topK=1700, similarity='tanimoto', normalize=False)

        # recommender_3 = UserKNNCFRecommender(urm_train)
        # recommender_3.fit(shrink=2, topK=600, normalize=True)
        #
        # recommender = User_CFW_D_Similarity_Linalg(urm_train, ucm_all, recommender_3.W_sparse)
        # recommender.fit(topK=1740, add_zeros_quota=0.3528735601555612, normalize_similarity=True)

        zero_recommender = recommender

    if temperature == 'warm' and test is True:
        # icm_price, icm_asset, icm_sub, icm_all = Data.get_icm()
        # ucm_age, ucm_region, ucm_all = Data.get_ucm()


        # recommender = ItemKNNCFRecommender(urm_train)
        # recommender.fit(shrink=30, topK=20)

        # recommender = HybridNormRecommender(urm_train, ucm_all, icm_all)
        # recommender.fit()

        warm_recommender= cold_recommender

    if temperature == 'normal' or test is False:
        icm_weighted = sp.load_npz(data_folder / 'Data/icm_weighted.npz')
        ucm_age, ucm_region, ucm_all = Data.get_ucm()
        icm_price, icm_asset, icm_sub, icm_all = Data.get_icm()

        # x_tick = np.arange(start=0, stop=1.001, step=0.02)
        # print(len(x_tick))
        # MAP_per_k_test = []
        #
        # MAP_per_k_valid = []
        #
        # recommender = HybridNorm2Recommender(urm_train)
        #
        #
        # best_alpha = 0
        # best_res = 0
        # for alpha in tqdm(x_tick):
        #
        #     recommender.fit(alpha=alpha)
        #
        #     result_dict, res_str = evaluator_valid.evaluateRecommender(recommender)
        #     MAP_per_k_valid.append(result_dict[10]["MAP"])
        #
        #     if result_dict[10]["MAP"] > best_res:
        #         best_alpha = alpha
        #         best_res = result_dict[10]["MAP"]
        #
        # print('The alpha is {}'.format(best_alpha))
        #
        #
        #
        # pyplot.plot(x_tick, MAP_per_k_valid)
        # pyplot.ylabel('MAP_valid')
        # pyplot.xlabel('alpha')
        # pyplot.show()
        #
        # recommender.fit(alpha=alpha)

        # earlystopping_keywargs = {"epochs_max": 2
        #                           }
        #
        # recommender = SLIM_BPR_Cython(urm_train)
        # recommender.fit(epochs=60, lambda_i=0.0297, lambda_j=0.0188, learning_rate=0.008083, topK=202, **earlystopping_keywargs)
        #
        # res = recommender.best_validation_metric
        # n_epochs = recommender.epochs_best

        recommender = ALSRecommender(urm_train)
        recommender.fit()

        # recommender = MF_MSE_PyTorch(urm_train)
        # recommender.fit()

        # recommender = ItemKNNCFRecommender(urm_train)
        # recommender.fit(topK=5, shrink=500, feature_weighting='BM25', similarity='tversky', normalize=False, tversky_alpha=0.0, tversky_beta=1.0)

        # recommender = UserKNNCFRecommender(urm_train)
        # recommender.fit(topK=697, shrink=1000, feature_weighting='TF-IDF', similarity='tversky', normalize=False, tversky_alpha=1.0, tversky_beta=1.0)

        # recommender = RP3betaRecommender(urm_train)
        # recommender.fit(topK=31, alpha=0.40346505, beta=0.1560091, normalize_similarity=True)

        normal_recommender = recommender

    if test:

        if temperature == 'cold':
            result, str_result = evaluator_test.evaluateRecommender(cold_recommender)
            print('The Map is : {}'.format(result[10]['MAP']))

        if temperature == 'zero':
            result, str_result = evaluator_test.evaluateRecommender(zero_recommender)
            print('The Map of test is : {}'.format(result[10]['MAP']))
            # if valid:
            #     result, str_result = evaluator_valid.evaluateRecommender(zero_recommender)
            #     print('The Map of valid is : {}'.format(result[10]['MAP']))
        if temperature == 'warm':
            result, str_result = evaluator_test.evaluateRecommender(warm_recommender)
            print('The Map is : {}'.format(result[10]['MAP']))
            # if valid:
            #     result, str_result = evaluator_valid.evaluateRecommender(warm_recommender)
            #     print('The Map of valid is : {}'.format(result[10]['MAP']))

        if temperature == 'normal':
            result, str_result = evaluator_test.evaluateRecommender(normal_recommender)
            print('The Map of test is : {}'.format(result[10]['MAP']))


    else:

        users = Data.get_target_users()

        recommended_list = []
        for user in tqdm(users):
            if Data.get_cold_users(0)[user]:
                recommended_items = zero_recommender.recommend(user, 10)
            else:
                recommended_items = normal_recommender.recommend(user, 10)

            items_strings = ' '.join([str(i) for i in recommended_items])
            recommended_list.append(items_strings)

        submission = pd.DataFrame(list(zip(users, recommended_list)), columns=['user_id', 'item_list'])
        submission.to_csv(data_folder / 'Data/Submissions/29-12_2_submission.csv', index=False)
