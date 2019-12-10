"""
Created on 04/12/19

@author: Giuseppe Serna
"""
from DataManager.DataManager import DataManager
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from DataManager.split_train_validation_leave_k_out import split_train_leave_k_out_user_wise
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Evaluator.evaluation import evaluate
from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from MatrixFactorization.PyTorch.MF_MSE_PyTorch import MF_MSE_PyTorch
from Notebooks_utils.data_splitter import train_test_holdout
from Base.Evaluation.Evaluator import EvaluatorHoldout

Data = DataManager()


urm_train, urm_test = split_train_leave_k_out_user_wise(Data.get_urm(), use_validation_set=False, leave_random_out=True)

# urm_train, urm_valid = split_train_leave_k_out_user_wise(urm_train, use_validation_set=False, leave_random_out=True)

# urm_train, urm_test = train_test_holdout(Data.get_urm(), train_perc=0.8)
# urm_train, urm_valid = train_test_holdout(urm_train, train_perc=0.8)
# evaluator_valid = EvaluatorHoldout(urm_valid, cutoff_list=[10])
evaluator_test = EvaluatorHoldout(urm_test, cutoff_list=[10])

recommender = SLIM_BPR_Cython(urm_train)
earlystopping_keywargs = {"validation_every_n": 5,
                              "stop_on_validation": True,
                              "evaluator_object": evaluator_test,
                              "lower_validations_allowed": 5,
                              "validation_metric": "MAP"
                          }
tuning_params = dict()
tuning_params = {
    "L1": (0.0001, 0.1),
    "L2": (0.0001, 0.1),
    "LE": (0.00001, 0.1),
    "NN": (150, 220),
    # "SH": (1, 10)
 }


def search_param(**tuning_params):
    recommender.fit(epochs=100, topK=int(tuning_params['NN']), lambda_i=tuning_params['L1'],
                    lambda_j=tuning_params['L2'], learning_rate=tuning_params['LE'], **earlystopping_keywargs)
    # recommender.fit(topK=int(tuning_params['NN']), shrink=tuning_params['SH'])
    res_test, str = evaluator_test.evaluateRecommender(recommender)
    # res_test = evaluate(urm_test, recommender)
    return res_test[10]["MAP"]


optimizer = BayesianOptimization(
    f=search_param,
    pbounds=tuning_params,
    verbose=3,
    random_state=5,
)

#load_logs(optimizer, logs=["./logs.json"])


logger = JSONLogger(path="/content/RecSys_course_challenge/Logs/tmp/" + recommender.RECOMMENDER_NAME + ".json")
optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

optimizer.probe(
    params={
    "NN": 198,
    "LE": 1.0e-05,
    "L1": 0.0001,
    "L2": 0.5},
    lazy=True,
)


optimizer.maximize(
    init_points=15,
    n_iter=40,
)

print(optimizer.max)

