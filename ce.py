import logging
import os

from sklearn.svm import SVC
from termcolor import cprint

import transcend.calibration as calibration
import transcend.data as data
import transcend.scores as scores
import transcend.thresholding as thresholding
import transcend.utils as utils


def main():
    # ---------------------------------------- #
    # 0. Prelude                               #
    # ---------------------------------------- #

    utils.configure_logger()
    args = utils.parse_args()
    logging.info('Loading {} training features...'.format(args.train))
    X_train, y_train = data.load_features(args.train)
    logging.info('Loaded: {}'.format(X_train.shape, y_train.shape))

    saved_data_folder = os.path.join(
        'models', '{}-fold-{}'.format(args.folds, args.test))

    # ---------------------------------------- #
    # 1. Calibration                           #
    # ---------------------------------------- #

    logging.info('Training calibration set...')
    fold_results_list = calibration.train_calibration_set(
        X_train, y_train, args.folds, args.ncpu, saved_data_folder)
    logging.info('Concatenating calibration fold results...')
    cal_results_dict = calibration.concatenate_calibration_set_results(
        fold_results_list)

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

        p_val_found_thresholds = thresholding.find_random_search_thresholds_with_constraints(
            scores=scores_p_val_cal,
            predicted_labels=pred_cal,
            groundtruth_labels=groundtruth_cal,
            maximise_vals=args.cs_max,
            constraint_vals=args.cs_con,
            max_samples=args.rs_samples)

        scores_probas_cal = {'cred': probas_cal}

        prob_found_thresholds = thresholding.find_random_search_thresholds_with_constraints(
            scores=scores_probas_cal,
            predicted_labels=pred_proba_cal,
            groundtruth_labels=groundtruth_cal,
            maximise_vals=args.cs_max,
            constraint_vals=args.cs_con,
            max_samples=args.rs_samples,
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
    model_name = 'svm_full_test_phase.p'
    model_name = os.path.join(saved_data_folder, model_name)

    if os.path.exists(model_name):
        svm = data.load_cached_data(model_name)
    else:
        svm = SVC(probability=True, kernel='linear', verbose=True)
        svm.fit(X_train, y_train)
        data.cache_data(svm, model_name)

    # ---------------------------------------- #
    # 4. Score and Predict Test Observations   #
    # ---------------------------------------- #

    logging.info('Loading {} test features...'.format(args.test))
    X_test, y_test = data.load_features(args.test)
    logging.info('Loaded: {}'.format(X_test.shape, y_test.shape))

    # Probability scores

    logging.info('Getting probabilities for test ({})...'.format(args.test))
    probas_test, pred_proba_test = scores.get_svm_probs(svm, X_test)

    # P-value scores

    logging.info('Computing p-values for test ({})...'.format(args.test))
    pred_test = svm.predict(X_test)

    saved_data_name = 'p_vals_ncms_{}_svm_full_test_phase.p'.format(
        args.pval_consider.replace('-', '_'))
    saved_data_name = os.path.join(saved_data_folder, saved_data_name)

    if os.path.exists(saved_data_name):
        p_val_test_dict = data.load_cached_data(saved_data_name)

    else:
        if args.pval_consider == 'full-train':
            logging.info('Getting NCMs for train ({})...'.format(args.train))
            ncms = scores.get_svm_ncms(svm, X_train, y_train)
            groundtruth = y_train
        elif args.pval_consider == 'cal-only':
            logging.info('Using calibration ncms...')
            ncms = cal_results_dict['ncms_cal']
            groundtruth = groundtruth_cal
        else:
            raise ValueError('Unknown value: args.pval_consider={}'.format(
                args.pval_consider))

        logging.info('Getting NCMs for test ({})...'.format(args.test))
        ncms_full_test = scores.get_svm_ncms(svm, X_test, y_test)

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

            results = thresholding.test_with_rejection(
                binary_thresholds=p_val_binary_thresholds,
                test_scores=p_val_test_dict,
                groundtruth_labels=y_test,
                predicted_labels=pred_test)

            report_str += thresholding.report_results(results)

            prob_binary_thresholds = {'cred': probas_thresholds[q]}
            prob_test_scores = {'cred': probas_test}

            print_and_extend('=' * 40)
            print_and_extend('[PROBS] Threshold criteria: {}'.format(q))
            print_thresholds(prob_binary_thresholds)

            results = thresholding.test_with_rejection(
                binary_thresholds=prob_binary_thresholds,
                test_scores=prob_test_scores,
                groundtruth_labels=y_test,
                predicted_labels=pred_proba_test)

            report_str += thresholding.report_results(results)

    elif args.thresholds in ('random-search', 'constrained-search'):

        print_and_extend('=' * 40)
        print_and_extend('[P-VALS] Threshold with random grid search')
        print_thresholds(p_val_found_thresholds)

        results = thresholding.test_with_rejection(
            binary_thresholds=p_val_found_thresholds,
            test_scores=p_val_test_dict,
            groundtruth_labels=y_test,
            predicted_labels=pred_test)

        report_str += thresholding.report_results(results)

        print_and_extend('=' * 40)
        print_and_extend('[PROBS] Threshold with random grid search')
        print_thresholds(prob_found_thresholds)

        prob_test_scores = {'cred': probas_test}

        results = thresholding.test_with_rejection(
            binary_thresholds=prob_found_thresholds,
            test_scores=prob_test_scores,
            groundtruth_labels=y_test,
            predicted_labels=pred_proba_test)

        report_str += thresholding.report_results(results)

    else:
        raise ValueError(
            'Unknown option: args.thresholds = {}'.format(args.threshold))

    data.save_results(report_str, args)


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
    if 'cred' not in binary_thresholds and 'conf' not in binary_thresholds:
        s = 'No threshold found!'
    logging.info(s)
    return s


if __name__ == '__main__':
    main()
