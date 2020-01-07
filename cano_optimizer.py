


from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
import numpy as np
from Base.Evaluation.Evaluator import EvaluatorHoldout
from Hybrid.HybridNormRecommender import HybridNormRecommender
from DataManager.split_train_validation_leave_k_out import split_train_leave_k_out_user_wise
from DataManager.DataManager import DataManager
from Hybrid.HybridNormOrigRecommender import HybridNormOrigRecommender

data = DataManager()


tuning_params = dict()
tuning_params = {
  "alpha": (0, 1),
  "beta": (0, 1),
  "gamma": (0, 1),
  "phi": (0, 1),
  "psi": (0, 1),
  "li": (0, 1),
 }



# def search_param(alpha, beta, gamma, phi, psi, li):
#     recommender.fit(alpha=alpha, beta=beta, gamma=gamma, phi=phi, psi=psi, li=li)
#     result_test, str_result = evaluator_test.evaluateRecommender(recommender)
#     result_valid, str_result = evaluator_valid.evaluateRecommender(recommender)
#     print('Il Map del test è : {}'.format(result_test[10]['MAP']))
#     return result_valid[10]['MAP']

num_test = 5

my_input = []

for i in np.arange(num_test):

    urm_train, urm_test = split_train_leave_k_out_user_wise(data.get_urm(), temperature='normal')
    urm_train, urm_valid = split_train_leave_k_out_user_wise(urm_train, temperature='valid2')
    recommender = HybridNormOrigRecommender(urm_train)
    my_input.append([urm_valid, recommender])



def search_param(alpha, beta, gamma, phi, psi, li):
    res = []
    for current in my_input:
        recommender = current[1]
        urm_valid = current[0]
        evaluator_valid = EvaluatorHoldout(urm_valid, cutoff_list=[10])

        recommender.fit(alpha=alpha, beta=beta, gamma=gamma, phi=phi, psi=psi, li=li)

        result_valid, str_result = evaluator_valid.evaluateRecommender(recommender)

        res.append(result_valid[10]['MAP'])
    print('Il max è : {}'.format(optimizer.max))
    res = np.array(res)
    print('Il Map è : {}'.format(res.mean()))
    return res.mean()

optimizer = BayesianOptimization(
    f=search_param,
    pbounds=tuning_params,
    verbose=3,
    random_state=50,
)

logger = JSONLogger(path="./Logs/tmp/" + 'HybridNormRecommender' + ".json")
optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

# optimizer.probe(
#     params={
#     'A': 5.0,
#     'NF': 99.9773715259928,
#     'REG': 0.0189810660740337},
#     lazy=True,
# )

optimizer.maximize(
    init_points=15,
    n_iter=500,
)

print(optimizer.max)
