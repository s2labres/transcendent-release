import logging
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC
from termcolor import cprint
from tqdm import tqdm

import matplotlib.pyplot as plt

import transcend.calibration as calibration
import transcend.data as data
import transcend.scores as scores
import transcend.thresholding as thresholding
import transcend.utils as utils

import pickle

import json

import numpy as np
from datetime import datetime
from tesseract import temporal


def main():
    # ---------------------------------------- #
    # 0. Prelude                               #
    # ---------------------------------------- #

    utils.configure_logger()
    args = utils.parse_args()

    if args.pval_consider != 'cal-only':
        raise ValueError('This is ICE - only compute test pvals with cal ncms')

    if len(args.dataset):
        logging.info('Loading {} features...'.format(args.dataset))
        with open(f"./features/{args.dataset}-X.json", 'rb') as f:
            X = json.load(f)

        logging.info('Loading {} labels...'.format(args.dataset))
        with open(f"./features/{args.dataset}-y.json", 'r') as f:
            y = json.load(f)

        logging.info('Loading {} metadata...'.format(args.dataset))
        with open(f"./features/{args.dataset}-meta.json", 'r') as f:
            T = json.load(f)
            T = [o['dex_date'] for o in T]
            t = np.array(
                [datetime.strptime(o, '%Y-%m-%dT%H:%M:%S') if "T" in o else datetime.strptime(o, '%Y-%m-%d %H:%M:%S')
                 for o in T])

        logging.info('Vectorize {}...'.format(args.dataset))
        vec = DictVectorizer()
        X = vec.fit_transform(X)
        y = np.asarray(y)

        logging.info('Partition {} training, testing, and timestamps...'.format(args.dataset))
        # Partition dataset via TESSERACT
        splits = temporal.time_aware_train_test_split(X, y, t, train_size=12, test_size=1, granularity='month')
        X_train, X_test, y_train, y_test, t_train, t_test = splits

        argstrain = args.dataset
    else:

        logging.info('Loading {} training features...'.format(args.train))

        X_train = pickle.load(open(f"./features/{args.train}_X.p", "rb"))
        y_train = pickle.load(open(f"./features/{args.train}_y.p", "rb"))

        X_test = pickle.load(open(f"./features/{args.test}_X.p", "rb"))
        y_test = pickle.load(open(f"./features/{args.test}_y.p", "rb"))
        # X_train, y_train = data.load_features(args.train)
        argstrain = args.train

    logging.info('Loaded: {}'.format(X_train.shape, y_train.shape))

    test_size = 0.34

    # saved_data_folder = os.path.join('models', '{}-fold'.format(args.folds))
    saved_data_folder = os.path.join('models', 'ice-{}-{}'.format(args.folds, argstrain))

    # ---------------------------------------- #
    # 1. Calibration                           #
    # ---------------------------------------- #

    logging.info('Training calibration set...')

    X_proper_train, X_cal, y_proper_train, y_cal = train_test_split(
        X_train, y_train, test_size=test_size, random_state=3)

    cal_results_dict = calibration.train_calibration_ice(
        X_proper_train=X_proper_train,
        X_cal=X_cal,
        y_proper_train=y_proper_train,
        y_cal=y_cal,
        fold_index='ice_{}'.format(test_size),
        saved_data_folder=saved_data_folder
    )

    # fold_results_list = calibration.train_calibration_set(
    #     X_train, y_train, args.folds, args.ncpu, saved_data_folder)
    # logging.info('Concatenating calibration fold results...')
    # cal_results_dict = calibration.concatenate_calibration_set_results(
    #     fold_results_list)

    # ---------------------------------------- #
    # 2. Find Calibration Thresholds           #
    # ---------------------------------------- #

    cred_p_val_cal = cal_results_dict['cred_p_val_cal']
    conf_p_val_cal = cal_results_dict['conf_p_val_cal']

    pred_cal = cal_results_dict['pred_cal']
    groundtruth_cal = cal_results_dict['groundtruth_cal']

    probas_cal = cal_results_dict['probas_cal']
    pred_proba_cal = cal_results_dict['pred_proba_cal']

    if args.thresholds == 'quartiles':
        if 'cred' in args.criteria:
            logging.info('Finding cred p-value thresholds (quartiles)...')
            cred_p_val_thresholds = thresholding.find_quartile_thresholds(
                scores=cred_p_val_cal,
                predicted_labels=pred_cal,
                groundtruth_labels=groundtruth_cal,
                consider=args.q_consider)

        if 'conf' in args.criteria:
            logging.info('Finding conf p-value thresholds (quartiles)...')
            conf_p_val_thresholds = thresholding.find_quartile_thresholds(
                scores=conf_p_val_cal,
                predicted_labels=pred_cal,
                groundtruth_labels=groundtruth_cal,
                consider=args.q_consider)

        logging.info('Finding probability thresholds (quartiles)...')
        probas_thresholds = thresholding.find_quartile_thresholds(
            scores=probas_cal,
            predicted_labels=pred_proba_cal,
            groundtruth_labels=groundtruth_cal,
            consider=args.q_consider)

    elif args.thresholds == 'random-search':
        scores_p_val_cal = package_cred_conf(
            cred_p_val_cal, conf_p_val_cal, args.criteria)

        p_val_found_thresholds = thresholding.find_random_search_thresholds(
            scores=scores_p_val_cal,
            predicted_labels=pred_cal,
            groundtruth_labels=groundtruth_cal,
            max_metrics=args.rs_max,
            min_metrics=args.rs_min,
            ceiling=args.rs_ceiling,
            max_samples=args.rs_samples)

        scores_probas_cal = {'cred': probas_cal}

        prob_found_thresholds = thresholding.find_random_search_thresholds(
            scores=scores_probas_cal,
            predicted_labels=pred_proba_cal,
            groundtruth_labels=groundtruth_cal,
            max_metrics=args.rs_max,
            min_metrics=args.rs_min,
            ceiling=args.rs_ceiling,
            max_samples=args.rs_samples)

    elif args.thresholds == 'constrained-search':

        scores_p_val_cal = package_cred_conf(
            cred_p_val_cal, conf_p_val_cal, args.criteria)

        # TODO: add the same logic to random-search and quartiles; this
        # is a computationally-intensive task, so caching them helps if
        # re-running experiments. Make sure stale caches are properly
        # deleted, tho!

        # Cache pval scores in thresholding.find_random_search_thresholds_with_constraints
        statistic_name = 'svm_scores_p_val_cal_ice_{}.p'.format(test_size)
        statistic_name = os.path.join(saved_data_folder, statistic_name)

        p_val_found_thresholds = thresholding.find_random_search_thresholds_with_constraints(
            scores=scores_p_val_cal,
            predicted_labels=pred_cal,
            groundtruth_labels=groundtruth_cal,
            maximise_vals=args.cs_max,
            constraint_vals=args.cs_con,
            # statistic_name=statistic_name,
            max_samples=args.rs_samples)

        scores_probas_cal = {'cred': probas_cal}

        # Cache proba scores in thresholding.find_random_search_thresholds_with_constraints
        statistic_name = 'svm_scores_p_probas_cal_ice_{}.p'.format(test_size)
        statistic_name = os.path.join(saved_data_folder, statistic_name)

        prob_found_thresholds = thresholding.find_random_search_thresholds_with_constraints(
            scores=scores_probas_cal,
            predicted_labels=pred_proba_cal,
            groundtruth_labels=groundtruth_cal,
            maximise_vals=args.cs_max,
            constraint_vals=args.cs_con,
            max_samples=args.rs_samples,
            # statistic_name=statistic_name,
            quiet=True)

    else:
        msg = 'Unknown option: args.thresholds = {}'.format(args.threshold)
        logging.critical(msg)
        raise ValueError(msg)

    # ---------------------------------------- #
    # 3. Generate 'Full' Model for Deployment  #
    # ---------------------------------------- #

    logging.info('Beginning TEST phase.')

    logging.info('Training model on full training set...')

    # model_name = 'svm_full_test_phase.p'

    model_name = 'svm_cal_fold_ice_{}.p'.format(test_size)
    model_name = os.path.join(saved_data_folder, model_name)

    if os.path.exists(model_name):
        logging.warning('FOUND SAVED ICE PROPER TRAIN MODEL.')
        svm = data.load_cached_data(model_name)
    else:
        logging.warning('NOT FOUND SAVED ICE PROPER TRAIN MODEL. Retraining...')
        svm = SVC(probability=True, kernel='linear', verbose=True)
        svm.fit(X_train, y_train)
        data.cache_data(svm, model_name)

    # ---------------------------------------- #
    # 4. Score and Predict Test Observations   #
    # ---------------------------------------- #

    if len(args.dataset):
        argstest = args.dataset
    else:
        # X_test_temp, y_test_temp = data.load_features(args.test)
        argstest = args.test

    X_test_temp = X_test
    y_test_temp = y_test

    logging.info('Loading {} test features...'.format(argstest))

    test_X_y = zip(X_test_temp, y_test_temp)
    time_series_p_val_results, time_series_p_vals, p_val_keep_masks, \
        time_series_proba_results, time_series_probas, proba_keep_masks = [], [], [], [], [], []

    for X_test, y_test in tqdm(test_X_y):
        # Probability scores

        logging.info('Getting probabilities for test ({})...'.format(argstest))
        probas_test, pred_proba_test = scores.get_svm_probs(svm, X_test)

        # P-value scores

        logging.info('Computing p-values for test ({})...'.format(argstest))
        pred_test = svm.predict(X_test)

        saved_data_name = 'p_vals_ncms_{}_svm_full_test_phase.p'.format(
            args.pval_consider.replace('-', '_'))
        saved_data_name = os.path.join(saved_data_folder, saved_data_name)

        # removing this for debug!!

        # if os.path.exists(saved_data_name):
        #    p_val_test_dict = data.load_cached_data(saved_data_name)

        # else:

        if True:
            if args.pval_consider == 'full-train':
                logging.info('Getting NCMs for train ({})...'.format(argstrain))
                ncms = scores.get_svm_ncms(svm, X_train, y_train)
                groundtruth = y_train
            elif args.pval_consider == 'cal-only':
                logging.info('Using calibration ncms...')
                ncms = cal_results_dict['ncms_cal']
                groundtruth = groundtruth_cal
            else:
                raise ValueError('Unknown value: args.pval_consider={}'.format(
                    args.pval_consider))

            logging.info('Getting NCMs for test ({})...'.format(argstest))
            ncms_full_test = scores.get_svm_ncms(svm, X_test, pred_test)

            p_val_test_dict = scores.compute_p_values_cred_and_conf(
                train_ncms=ncms,
                groundtruth_train=groundtruth,
                test_ncms=ncms_full_test,
                y_test=pred_test)
            data.cache_data(p_val_test_dict, saved_data_name)

        # ---------------------------------------- #
        # 5. Apply Thresholds, Compare Results     #
        # ---------------------------------------- #

        report_str = ''

        def print_and_extend(report_line):
            nonlocal report_str
            cprint(report_line, 'red')
            report_str += report_line + '\n'

        if args.thresholds == 'quartiles':
            for q in ('q1', 'q2', 'q3', 'mean'):
                p_val_binary_thresholds = {}

                if 'cred' in args.criteria:
                    p_val_binary_thresholds['cred'] = cred_p_val_thresholds[q]
                if 'conf' in args.criteria:
                    print('CONF HERE!')
                    print(args.criteria)
                    p_val_binary_thresholds['conf'] = conf_p_val_thresholds[q]

                print_and_extend('=' * 40)
                print_and_extend('[P-VALS] Threshold criteria: {}'.format(q))
                print_thresholds(p_val_binary_thresholds)

                results, keep_mask = thresholding.test_with_rejection(
                    binary_thresholds=p_val_binary_thresholds,
                    test_scores=p_val_test_dict,
                    groundtruth_labels=y_test,
                    predicted_labels=pred_test)

                time_series_p_val_results.append(results)
                time_series_p_vals.append(p_val_test_dict)
                p_val_keep_masks.append(keep_mask)

                report_str += thresholding.report_results(results)

                prob_binary_thresholds = {'cred': probas_thresholds[q]}
                prob_test_scores = {'cred': probas_test}

                print_and_extend('=' * 40)
                print_and_extend('[PROBS] Threshold criteria: {}'.format(q))
                print_thresholds(prob_binary_thresholds)

                results, keep_mask = thresholding.test_with_rejection(
                    binary_thresholds=prob_binary_thresholds,
                    test_scores=prob_test_scores,
                    groundtruth_labels=y_test,
                    predicted_labels=pred_proba_test)

                time_series_proba_results.append(results)
                time_series_probas.append(prob_test_scores)
                proba_keep_masks.append(keep_mask)

                report_str += thresholding.report_results(results)

        elif args.thresholds in ('random-search', 'constrained-search'):

            print_and_extend('=' * 40)
            print_and_extend('[P-VALS] Threshold with random grid search')
            print_thresholds(p_val_found_thresholds)

            results, keep_mask = thresholding.test_with_rejection(
                binary_thresholds=p_val_found_thresholds,
                test_scores=p_val_test_dict,
                groundtruth_labels=y_test,
                predicted_labels=pred_test)

            time_series_p_val_results.append(results)
            time_series_p_vals.append(p_val_test_dict)
            p_val_keep_masks.append(keep_mask)

            report_str += thresholding.report_results(results)

            print_and_extend('=' * 40)
            print_and_extend('[PROBS] Threshold with random grid search')
            print_thresholds(prob_found_thresholds)

            prob_test_scores = {'cred': probas_test}

            results, keep_mask = thresholding.test_with_rejection(
                binary_thresholds=prob_found_thresholds,
                test_scores=prob_test_scores,
                groundtruth_labels=y_test,
                predicted_labels=pred_proba_test)

            time_series_proba_results.append(results)
            time_series_probas.append(prob_test_scores)
            proba_keep_masks.append(keep_mask)

            report_str += thresholding.report_results(results)

        else:
            raise ValueError(
                'Unknown option: args.thresholds = {}'.format(args.threshold))

        data.save_results(report_str, args)

    data.cache_data(time_series_p_val_results, 'timeseries_cred_conf/ice_p_val_results.p')
    data.cache_data(time_series_p_vals, 'timeseries_cred_conf/ice_p_vals.p')
    data.cache_data(p_val_keep_masks, 'timeseries_cred_conf/ice_p_val_keep_masks.p')
    data.cache_data(time_series_proba_results, 'timeseries_cred_conf/ice_proba_results.p')
    data.cache_data(time_series_probas, 'timeseries_cred_conf/ice_probas.p')
    data.cache_data(proba_keep_masks, 'timeseries_cred_conf/ice_proba_keep_masks.p')


def package_cred_conf(cred_values, conf_values, criteria):
    package = {}

    if 'cred' in criteria:
        package['cred'] = cred_values
    if 'conf' in criteria:
        package['conf'] = conf_values

    return package


def print_thresholds(binary_thresholds):
    # Display per-class thresholds
    if 'cred' in binary_thresholds:
        s = ('Cred thresholds: mw {:.6f}, gw {:.6f}'.format(
            binary_thresholds['cred']['mw'],
            binary_thresholds['cred']['gw']))
    if 'conf' in binary_thresholds:
        s = ('Conf thresholds: mw {:.6f}, gw {:.6f}'.format(
            binary_thresholds['conf']['mw'],
            binary_thresholds['conf']['gw']))
    logging.info(s)
    return s


if __name__ == '__main__':
    main()
