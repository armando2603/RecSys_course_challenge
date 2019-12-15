from lightfm import LightFM
from DataManager.DataManager import DataManager
from lightfm.evaluation import auc_score
from DataManager.split_train_validation_leave_k_out import split_train_leave_k_out_user_wise
from Base.Evaluation.Evaluator import EvaluatorHoldout
from Hybrid.LighFm import LighFMRecommender
import matplotlib.pyplot as pyplot
from sklearn import  metrics, ensemble

data = DataManager()
urm = data.get_urm()
threshold = 10
temperature = 'normal'
ucm_age, ucm_region, ucm_all = data.get_ucm()
icm_price, icm_asset, icm_sub, icm_all = data.get_icm()
urm_train, urm_test = split_train_leave_k_out_user_wise(urm, threshold=threshold, temperature=temperature)
urm_train, urm_valid = split_train_leave_k_out_user_wise(urm_train, threshold=threshold, temperature=temperature)

evaluator_test = EvaluatorHoldout(urm_test, cutoff_list=[10])
evaluator_valid = EvaluatorHoldout(urm_valid, cutoff_list=[10])



# recommender = LighFMRecommender(urm_train,
#                                 no_components=150,
#                                 loss='warp',
#                                 learning_rate=0.09,
#                                 random_state=2019)
#
#
# recommender.fit(epochs=10, user_features=ucm_all, item_features=icm_sub, num_threads=4)
#
# result, str_result = evaluator_test.evaluateRecommender(recommender)
# print('The Map is : {}'.format(result[10]['MAP']))

x_tick = [100, 150, 200]

MAP_per_k_test = []

MAP_per_k_valid = []

for param in x_tick:
    recommender = LighFMRecommender(urm_train, no_components=param,
                                    loss='warp',
                                    learning_rate=0.03,
                                    random_state=2019,
                                    user_alpha=1e-6,
                                    item_alpha=0,
                                    )

    recommender.fit(epochs=40, user_features=None, item_features=None, num_threads=4)

    result_dict, res_str = evaluator_test.evaluateRecommender(recommender)
    MAP_per_k_test.append(result_dict[10]["MAP"])

    result_dict, res_str = evaluator_valid.evaluateRecommender(recommender)
    MAP_per_k_valid.append(result_dict[10]["MAP"])

pyplot.plot(x_tick, MAP_per_k_test)
pyplot.ylabel('MAP_test')
pyplot.xlabel('n_compo')
pyplot.show()

pyplot.plot(x_tick, MAP_per_k_valid)
pyplot.ylabel('MAP_valid')
pyplot.xlabel('n_comp')
pyplot.show()



# for param in x_tick:
#     MAP_per_k_test = []
#
#     MAP_per_k_valid = []
#     recommender = LighFMRecommender(urm_train, no_components=90,
#                                     loss='warp',
#                                     learning_rate=0.03,
#                                     random_state=2019,
#                                     user_alpha=0.0,
#                                     item_alpha=0.0,
#                                     )
#     tmp = 0
#     epochs = [30, 40, 60]
#     for epoch in epochs:
#
#         epoch_param = epoch - tmp
#         tmp = epoch
#         recommender.fit_partial(epochs=epoch_param, user_features=ucm_all, item_features=icm_sub, num_threads=4)
#
#         result_dict, res_str = evaluator_test.evaluateRecommender(recommender)
#         MAP_per_k_test.append(result_dict[10]["MAP"])
#
#         result_dict, res_str = evaluator_valid.evaluateRecommender(recommender)
#         MAP_per_k_valid.append(result_dict[10]["MAP"])
#
#     pyplot.plot(epochs, MAP_per_k_test)
#     pyplot.ylabel('MAP_test')
#     pyplot.xlabel('Epochs')
#     pyplot.show()
#
#     pyplot.plot(epochs, MAP_per_k_valid)
#     pyplot.ylabel('MAP_valid')
#     pyplot.xlabel('Epochs with')
#     pyplot.show()
