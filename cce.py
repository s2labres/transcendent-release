import json
import logging
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction import DictVectorizer
from termcolor import cprint
from tqdm import tqdm, trange

import transcend.calibration as calibration
import transcend.data as data
import transcend.scores as scores
import transcend.thresholding as thresholding
import transcend.utils as utils

import pickle
import multiprocessing as mp
import numpy as np
from datetime import datetime
from tesseract import temporal


def main():
    # ---------------------------------------------- #
    # 0. Prelude                                     #
    # ---------------------------------------------- # 

    utils.configure_logger()
    global args, X_test, y_test
    args = utils.parse_args()

    if args.pval_consider != 'cal-only':
        raise ValueError('This is CCE - only compute test pvals with cal ncms')

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
    else:

        logging.info('Loading {} training features...'.format(args.train))

        X_train = pickle.load(open(f"./features/{args.train}_X.p", "rb"))
        y_train = pickle.load(open(f"./features/{args.train}_y.p", "rb"))

        logging.info('Loading {} test features...'.format(args.test))
        X_test = pickle.load(open(f'features/{args.test}_X.p', 'rb'))
        y_test = pickle.load(open(f'features/{args.test}_y.p', 'rb'))

    logging.info('Loaded {}'.format(X_train.shape, y_train.shape))

    NCPU = mp.cpu_count() - 1 if mp.cpu_count() > 2 else 1

    global saved_data_folder
    saved_data_folder = os.path.join('models',
                                     'cce-{}-{}'.format(args.folds, args.dataset if len(args.dataset) else args.train))

    global folds
    folds = args.folds

    logging.info('Working directory: {}'.format(saved_data_folder))

    # logging.info('Loaded: {}'.format(X_test.shape, y_test.shape))

    # ------------------------------------------------ #
    # 1. Calibration                                   #
    # ------------------------------------------------ # 

    logging.info('Training models on calibration set...')

    '''
        This process is similar to ICE. We will use the same method, but use a stratified 10-fold
        to train 10 different models. We will then perform ICE on the individual folds and use a 
        voting system to accept or reject a point depending on k which defines how many accepted 
        decisions each the point should have.
    '''

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=21)

    folds_generator = ({
        'X_train': X_train[train_index],
        'y_train': y_train[train_index],
        'X_cal': X_train[cal_index],
        'y_cal': y_train[cal_index],
        'idx': idx,
        'X_test': X_test,
        'y_test': y_test,
        'folder': saved_data_folder,
        'args': args
    } for idx, (train_index, cal_index) in enumerate(skf.split(X_train, y_train)))

    p_val_masks, proba_masks = [], []
    p_val_preds, proba_preds = [], []

    with mp.Pool(processes=NCPU) as p:
        for res in p.imap(parallel_ice, folds_generator):
            if res is not None:
                p_val_masks.append(res['mask_p_val'])
                proba_masks.append(res['mask_proba'])

                p_val_preds.append(res['predict_p_val'])
                proba_preds.append(res['predict_proba'])
            else:
                raise RuntimeError('CCE response is empty')

    time_series_p_val, time_series_proba = [], []
    time_series_p_val_keep_mask, time_series_proba_keep_mask = [], []

    for i in range(len(p_val_preds[0])):
        p_val_preds_temp, p_val_masks_temp = [p_vals[i] for p_vals in p_val_preds], [masks[i] for masks in p_val_masks]
        proba_preds_temp, proba_masks_temp = [proba[i] for proba in proba_preds], [masks[i] for masks in proba_masks]

        p_val_y_pred, p_val_keep_mask = get_results(p_val_preds_temp, p_val_masks_temp)
        proba_y_pred, proba_keep_mask = get_results(proba_preds_temp, proba_masks_temp)

        logging.info('Final results:')

        report_str = ''

        def print_and_extend(report_line):
            nonlocal report_str
            cprint(report_line, 'red')
            report_str += report_line + '\n'

        print_and_extend('=' * 40)
        print_and_extend('[P-VALS] Threshold with random grid search')

        p_val_keep_mask = p_val_keep_mask.astype(bool)

        results = thresholding.get_performance_with_rejection(
            y_true=y_test[i],
            y_pred=p_val_y_pred,
            keep_mask=p_val_keep_mask)

        time_series_p_val.append(results)
        time_series_p_val_keep_mask.append(p_val_keep_mask)
        report_str += thresholding.report_results(results)

        print_and_extend('=' * 40)
        print_and_extend('[PROBS] Threshold with random grid search')

        proba_keep_mask = proba_keep_mask.astype(bool)

        results = thresholding.get_performance_with_rejection(
            y_true=y_test[i],
            y_pred=proba_y_pred,
            keep_mask=proba_keep_mask)

        time_series_proba.append(results)
        time_series_proba_keep_mask.append(proba_keep_mask)
        report_str += thresholding.report_results(results)

        data.save_results(report_str, args)

    data.cache_data(time_series_p_val, 'timeseries_cred_conf/cce_p_val_results.p')
    data.cache_data(time_series_proba, 'timeseries_cred_conf/cce_proba_results.p')
    data.cache_data(time_series_p_val_keep_mask, 'timeseries_cred_conf/cce_p_val_keep_mask.p')
    data.cache_data(time_series_proba_keep_mask, 'timeseries_cred_conf/cce_proba_keep_mask.p')


def parallel_ice(folds_gen):
    X_train, y_train = folds_gen['X_train'], folds_gen['y_train']
    X_cal, y_cal = folds_gen['X_cal'], folds_gen['y_cal']
    X_test, y_test = folds_gen['X_test'], folds_gen['y_test']
    idx = folds_gen['idx']
    saved_data_folder = folds_gen['folder']
    args = folds_gen['args']

    logging.info('[{}] Starting calibration...'.format(idx))

    cal_results_dict = calibration.train_calibration_ice(
        X_proper_train=X_train,
        X_cal=X_cal,
        y_proper_train=y_train,
        y_cal=y_cal,
        fold_index='cce_{}'.format(idx),
        saved_data_folder=saved_data_folder
    )

    svm = cal_results_dict['model']

    # --------------------------------------- #
    # 2. Find Calibration Thresholds          #
    # --------------------------------------- # 

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

        p_val_found_thresholds = thresholding.find_random_search_thresholds_with_constraints(
            scores=scores_p_val_cal,
            predicted_labels=pred_cal,
            groundtruth_labels=groundtruth_cal,
            maximise_vals=args.cs_max,
            constraint_vals=args.cs_con,
            max_samples=args.rs_samples,
            ncpu=1)

        scores_probas_cal = {'cred': probas_cal}

        prob_found_thresholds = thresholding.find_random_search_thresholds_with_constraints(
            scores=scores_probas_cal,
            predicted_labels=pred_proba_cal,
            groundtruth_labels=groundtruth_cal,
            maximise_vals=args.cs_max,
            constraint_vals=args.cs_con,
            max_samples=args.rs_samples,
            quiet=True,
            ncpu=1)

    else:
        msg = 'Unknown option: args.thresholds = {}'.format(args.threshold)
        logging.critical(msg)
        raise ValueError(msg)

    mask_p_val, mask_proba = [], []
    predict_proba, predict_p_val = [], []

    for X_test_temp, y_test_temp in tqdm(zip(X_test, y_test)):

        # Probability scores
        logging.info('[{}] Getting probabilities for test ({})...'.format(str(idx), args.dataset if len(
            args.dataset) else args.test))
        probas_test, pred_proba_test = scores.get_svm_probs(svm, X_test_temp)

        # P-value scores

        logging.info('[{}] Computing p-values for test ({})...'.format(str(idx), args.dataset if len(
            args.dataset) else args.test))

        pred_test = svm.predict(X_test_temp)

        saved_data_name = 'p_vals_ncms_{}_svm_full_test_phase_{}.p'.format(
            args.pval_consider.replace('-', '_'), str(idx))
        saved_data_name = os.path.join(saved_data_folder, saved_data_name)

        if True:
            if args.pval_consider == 'cal-only':
                logging.info('[{}] Using calibration ncms...'.format(str(idx)))
                ncms = cal_results_dict['ncms_cal']
                groundtruth = groundtruth_cal
            else:
                raise ValueError('[{}] Unknown value: args.pval_consider={}'.format(
                    str(idx), args.pval_consider))

            logging.info(
                '[{}] Getting NCMs for test ({})...'.format(str(idx), args.dataset if len(args.dataset) else args.test))
            ncms_full_test = scores.get_svm_ncms(svm, X_test_temp, pred_test)

            p_val_test_dict = scores.compute_p_values_cred_and_conf(
                train_ncms=ncms,
                groundtruth_train=groundtruth,
                test_ncms=ncms_full_test,
                y_test=pred_test)
            data.cache_data(p_val_test_dict, saved_data_name)

        report_str = ''

        def print_and_extend(report_line):
            nonlocal report_str
            cprint(report_line, 'red')
            report_str += report_line + '\n'

        if args.thresholds == 'quartiles':
            # NOT IMPLEMENTED YET 
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

                results, _ = thresholding.test_with_rejection(
                    binary_thresholds=p_val_binary_thresholds,
                    test_scores=p_val_test_dict,
                    groundtruth_labels=y_test_temp,
                    predicted_labels=pred_test)

                report_str += thresholding.report_results(results)

                prob_binary_thresholds = {'cred': probas_thresholds[q]}
                prob_test_scores = {'cred': probas_test}

                print_and_extend('=' * 40)
                print_and_extend('[PROBS] Threshold criteria: {}'.format(q))
                print_thresholds(prob_binary_thresholds)

                results, _ = thresholding.test_with_rejection(
                    binary_thresholds=prob_binary_thresholds,
                    test_scores=prob_test_scores,
                    groundtruth_labels=y_test_temp,
                    predicted_labels=pred_proba_test)

                report_str += thresholding.report_results(results)

        elif args.thresholds in ('random-search', 'constrained-search'):

            print_and_extend('=' * 40)
            print_and_extend('[P-VALS] Threshold with random grid search')
            print_thresholds(p_val_found_thresholds)

            results, _ = thresholding.test_with_rejection(
                binary_thresholds=p_val_found_thresholds,
                test_scores=p_val_test_dict,
                groundtruth_labels=y_test_temp,
                predicted_labels=pred_test)

            report_str += thresholding.report_results(results)

            print_and_extend('=' * 40)
            print_and_extend('[PROBS] Threshold with random grid search')
            print_thresholds(prob_found_thresholds)

            prob_test_scores = {'cred': probas_test}

            results, _ = thresholding.test_with_rejection(
                binary_thresholds=prob_found_thresholds,
                test_scores=prob_test_scores,
                groundtruth_labels=y_test_temp,
                predicted_labels=pred_proba_test)

            report_str += thresholding.report_results(results)

        else:
            raise ValueError(
                'Unknown option: args.thresholds = {}'.format(args.threshold))

        mask_p_val.append(thresholding.apply_threshold(
            binary_thresholds=p_val_found_thresholds,
            test_scores=p_val_test_dict,
            y_test=pred_test))

        mask_proba.append(thresholding.apply_threshold(
            binary_thresholds=prob_found_thresholds,
            test_scores=prob_test_scores,
            y_test=pred_proba_test))

        predict_proba.append(pred_proba_test)
        predict_p_val.append(pred_test)

    response = {}

    response['mask_p_val'] = mask_p_val

    response['mask_proba'] = mask_proba

    response['predict_proba'] = predict_proba
    response['predict_p_val'] = predict_p_val

    return response


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

    return s


def get_results(preds_data, mask_data):
    k = 6

    pred_return, mask_return = [], []

    for i in trange(mask_data[0].shape[0]):
        pred_zero_counter = 0
        temp_preds = []

        for preds, mask in zip(preds_data, mask_data):
            if preds[i] == 0:
                pred_zero_counter += 1

            if mask[i]:
                temp_preds.append(preds[i])

        if pred_zero_counter > int(folds / 2):
            pred = 0
        else:
            pred = 1

        if len(temp_preds) > k and temp_preds.count(pred) > k:
            mask_return.append(1)
        else:
            mask_return.append(0)

        pred_return.append(pred)

    pred_return = np.asarray(pred_return)
    mask_return = np.asarray(mask_return)

    return pred_return, mask_return


if __name__ == '__main__':
    main()
