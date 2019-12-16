#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/04/2019

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import scipy.sparse as sps
from DataManager.IncrementalSparseMatrix import IncrementalSparseMatrix


def split_train_leave_k_out_user_wise(URM, k_out = 1, use_validation_set = False, leave_random_out = True, threshold=10, temperature="normal"):
    """
    The function splits an URM in two matrices selecting the k_out interactions one user at a time
    :param temperature:
    :param threshold:
    :param URM:
    :param k_out:
    :param use_validation_set:
    :param leave_random_out:
    :return:
    """

    temperature_values = ["cold", "warm", "zero", "normal", 'valid']

    assert temperature in temperature_values, 'temperature must be "cold", "warm", "valid", "zero" or "normal"'
    assert k_out > 0, "k_out must be a value greater than 0, provided was '{}'".format(k_out)

    URM = sps.csr_matrix(URM)
    n_users, n_items = URM.shape


    URM_train_builder = IncrementalSparseMatrix(auto_create_row_mapper=False, n_rows = n_users,
                                        auto_create_col_mapper=False, n_cols = n_items)

    URM_test_builder = IncrementalSparseMatrix(auto_create_row_mapper=False, n_rows = n_users,
                                        auto_create_col_mapper=False, n_cols = n_items)

    if use_validation_set:
         URM_validation_builder = IncrementalSparseMatrix(auto_create_row_mapper=False, n_rows = n_users,
                                                          auto_create_col_mapper=False, n_cols = n_items)



    for user_id in range(n_users):

        start_user_position = URM.indptr[user_id]
        end_user_position = URM.indptr[user_id+1]

        user_profile = URM.indices[start_user_position:end_user_position]
        if temperature == "cold":
            if len(user_profile) <= threshold:
                if leave_random_out:
                    indices_to_suffle = np.arange(len(user_profile), dtype=np.int)

                    np.random.shuffle(indices_to_suffle)

                    user_interaction_items = user_profile[indices_to_suffle]
                    user_interaction_data = URM.data[start_user_position:end_user_position][indices_to_suffle]

                else:

                    # The first will be sampled so the last interaction must be the first one
                    interaction_position = URM.data[start_user_position:end_user_position]

                    sort_interaction_index = np.argsort(-interaction_position)

                    user_interaction_items = user_profile[sort_interaction_index]
                    user_interaction_data = URM.data[start_user_position:end_user_position][sort_interaction_index]

                # Test interactions
                user_interaction_items_test = user_interaction_items[0:k_out]
                user_interaction_data_test = user_interaction_data[0:k_out]

                URM_test_builder.add_data_lists([user_id] * len(user_interaction_items_test),
                                                user_interaction_items_test,
                                                user_interaction_data_test)

                # Train interactions
                user_interaction_items_train = user_interaction_items[k_out:]
                user_interaction_data_train = user_interaction_data[k_out:]

                URM_train_builder.add_data_lists([user_id] * len(user_interaction_items_train),
                                                 user_interaction_items_train, user_interaction_data_train)

                URM_train_builder.add_data_lists([user_id] * len(user_interaction_items_train),
                                                 user_interaction_items_train, user_interaction_data_train)
            else:
                user_interaction_items_train = user_profile
                user_interaction_data_train = URM.data[start_user_position:end_user_position]

                URM_train_builder.add_data_lists([user_id] * len(user_interaction_items_train),
                                                 user_interaction_items_train, user_interaction_data_train)

        if temperature == 'warm':
            if len(user_profile) > threshold:
                if leave_random_out:
                    indices_to_suffle = np.arange(len(user_profile), dtype=np.int)

                    np.random.shuffle(indices_to_suffle)

                    user_interaction_items = user_profile[indices_to_suffle]
                    user_interaction_data = URM.data[start_user_position:end_user_position][indices_to_suffle]

                else:

                    # The first will be sampled so the last interaction must be the first one
                    interaction_position = URM.data[start_user_position:end_user_position]

                    sort_interaction_index = np.argsort(-interaction_position)

                    user_interaction_items = user_profile[sort_interaction_index]
                    user_interaction_data = URM.data[start_user_position:end_user_position][sort_interaction_index]

                # Test interactions
                user_interaction_items_test = user_interaction_items[0:k_out]
                user_interaction_data_test = user_interaction_data[0:k_out]

                URM_test_builder.add_data_lists([user_id] * len(user_interaction_items_test),
                                                user_interaction_items_test,
                                                user_interaction_data_test)

                # Train interactions
                user_interaction_items_train = user_interaction_items[k_out:]
                user_interaction_data_train = user_interaction_data[k_out:]

                URM_train_builder.add_data_lists([user_id] * len(user_interaction_items_train),
                                                 user_interaction_items_train, user_interaction_data_train)
            else:
                user_interaction_items_train = user_profile
                user_interaction_data_train = URM.data[start_user_position:end_user_position]

                URM_train_builder.add_data_lists([user_id] * len(user_interaction_items_train),
                                                 user_interaction_items_train, user_interaction_data_train)

        if temperature == 'zero':

            if len(user_profile) == 1:
                if leave_random_out:
                    indices_to_suffle = np.arange(len(user_profile), dtype=np.int)

                    np.random.shuffle(indices_to_suffle)

                    user_interaction_items = user_profile[indices_to_suffle]
                    user_interaction_data = URM.data[start_user_position:end_user_position][indices_to_suffle]

                else:

                    # The first will be sampled so the last interaction must be the first one
                    interaction_position = URM.data[start_user_position:end_user_position]

                    sort_interaction_index = np.argsort(-interaction_position)

                    user_interaction_items = user_profile[sort_interaction_index]
                    user_interaction_data = URM.data[start_user_position:end_user_position][sort_interaction_index]

                # Test interactions
                user_interaction_items_test = user_interaction_items[0:k_out]
                user_interaction_data_test = user_interaction_data[0:k_out]

                URM_test_builder.add_data_lists([user_id] * len(user_interaction_items_test),
                                                user_interaction_items_test,
                                                user_interaction_data_test)



                # Train interactions
                user_interaction_items_train = user_interaction_items[k_out:]
                user_interaction_data_train = user_interaction_data[k_out:]

                URM_train_builder.add_data_lists([user_id] * len(user_interaction_items_train),
                                                 user_interaction_items_train, user_interaction_data_train)
            else:
                user_interaction_items_train = user_profile
                user_interaction_data_train = URM.data[start_user_position:end_user_position]

                URM_train_builder.add_data_lists([user_id] * len(user_interaction_items_train),
                                                 user_interaction_items_train, user_interaction_data_train)

        if temperature == 'normal':

            if leave_random_out:
                indices_to_suffle = np.arange(len(user_profile), dtype=np.int)

                np.random.shuffle(indices_to_suffle)

                user_interaction_items = user_profile[indices_to_suffle]
                user_interaction_data = URM.data[start_user_position:end_user_position][indices_to_suffle]

            else:

                # The first will be sampled so the last interaction must be the first one
                interaction_position = URM.data[start_user_position:end_user_position]

                sort_interaction_index = np.argsort(-interaction_position)

                user_interaction_items = user_profile[sort_interaction_index]
                user_interaction_data = URM.data[start_user_position:end_user_position][sort_interaction_index]

                # Test interactions

            # if len(user_profile) < 3:
            #     k_out = 1
            # else:
            #     k_out = int(0.2*len(user_profile))

            user_interaction_items_test = user_interaction_items[0:k_out]
            user_interaction_data_test = user_interaction_data[0:k_out]

            URM_test_builder.add_data_lists([user_id] * len(user_interaction_items_test), user_interaction_items_test,
                                            user_interaction_data_test)

            # Train interactions
            user_interaction_items_train = user_interaction_items[k_out:]
            user_interaction_data_train = user_interaction_data[k_out:]

            URM_train_builder.add_data_lists([user_id] * len(user_interaction_items_train),
                                             user_interaction_items_train, user_interaction_data_train)

        if temperature == 'valid':

            if leave_random_out:
                indices_to_suffle = np.arange(len(user_profile), dtype=np.int)

                np.random.shuffle(indices_to_suffle)

                user_interaction_items = user_profile[indices_to_suffle]
                user_interaction_data = URM.data[start_user_position:end_user_position][indices_to_suffle]

            else:

                # The first will be sampled so the last interaction must be the first one
                interaction_position = URM.data[start_user_position:end_user_position]

                sort_interaction_index = np.argsort(-interaction_position)

                user_interaction_items = user_profile[sort_interaction_index]
                user_interaction_data = URM.data[start_user_position:end_user_position][sort_interaction_index]

                # Test interactions

            if len(user_profile) < 3:
                k_out = 1
            else:
                k_out = int(0.2*len(user_profile))

            user_interaction_items_test = user_interaction_items[0:k_out]
            user_interaction_data_test = user_interaction_data[0:k_out]

            URM_test_builder.add_data_lists([user_id] * len(user_interaction_items_test), user_interaction_items_test,
                                            user_interaction_data_test)

            # Train interactions
            user_interaction_items_train = user_interaction_items[k_out:]
            user_interaction_data_train = user_interaction_data[k_out:]

            URM_train_builder.add_data_lists([user_id] * len(user_interaction_items_train),
                                             user_interaction_items_train, user_interaction_data_train)

    URM_train = URM_train_builder.get_SparseMatrix()
    URM_test = URM_test_builder.get_SparseMatrix()

    URM_train = sps.csr_matrix(URM_train)
    user_no_item_train = np.sum(np.ediff1d(URM_train.indptr) == 0)

    if user_no_item_train != 0:
        print("Warning: {} ({:.2f} %) of {} users have no Train items".format(user_no_item_train, user_no_item_train/n_users*100, n_users))



    if use_validation_set:
        URM_validation = URM_validation_builder.get_SparseMatrix()

        URM_validation = sps.csr_matrix(URM_validation)
        user_no_item_validation = np.sum(np.ediff1d(URM_validation.indptr) == 0)

        if user_no_item_validation != 0:
            print("Warning: {} ({:.2f} %) of {} users have no Validation items".format(user_no_item_validation, user_no_item_validation/n_users*100, n_users))


        return URM_train, URM_validation, URM_test


    return URM_train, URM_test




