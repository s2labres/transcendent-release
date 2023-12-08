import re

import scipy.sparse
from sklearn.ensemble import RandomForestClassifier
from tesseract import temporal
import ujson as json
import logging
from datetime import datetime
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.datasets import load_svmlight_file
import os
import pickle

# algorithm = 'svm-rbf'
algorithm = 'rf'

hidost_data = '/path/to/hidost'
batch = 'w08'


def main():
    # Load "train" and "test" data
    in1 = os.path.join(hidost_data, 'pdf-bin/{}-train.libsvm'.format(batch))
    X1, y1 = load_svmlight_file(in1)

    in2 = os.path.join(hidost_data, 'pdf-bin/{}-test.libsvm'.format(batch))
    X2, y2 = load_svmlight_file(in2)

    # Vectorize
    X = scipy.sparse.vstack((X1, X2))
    y = np.hstack((y1, y2))
    t = np.array(load_dates(in1) + load_dates(in2))

    # Get splits
    X_train, X_tests, y_train, y_tests, t_train, t_tests = temporal.time_aware_train_test_split(
        X, y, t, train_size=1, test_size=1, granularity='day')  #, start_date=datetime(2012, 7, 16))

    def dump(path, stuff):
        with open(path, 'wb') as f:
            pickle.dump(stuff, f)
            print(f"Dumped to {path}")


    dump("features/hidost/train_X.p", X_train)
    dump("features/hidost/test_X.p", X_tests)
    dump("features/hidost/train_y.p", y_train)
    dump("features/hidost/test_y.p", y_tests)

    # This can be used to reproduce the original 
    # Hidost experiments results 

    # # Create classifier
    # clf = {
    #     'rf': RandomForestClassifier(n_estimators=200),
    #     'svm-rbf': SVC(kernel='rbf', gamma=0.0025, C=12),
    # }[algorithm]

    # # Train
    # print('Training examples: ', X_train.shape)
    # clf.fit(X_train, y_train)

    # # Test
    # for i, (X_test, y_test) in enumerate(zip(X_tests, y_tests)):
    #     print('Test examples: ', X_test.shape)
    #     y_pred = clf.predict(X_test)
    #     f1 = f1_score(y_test, y_pred)
    #     print(f'Period {i+1}: {f1:.2f}')


def load_dates(infile):
    """
    Parses infile for any dates formatted as YYYY/MM/DD, at most one
    per line. Returns a list of datetime.date objects, in order of
    encounter.
    """
    datere = re.compile(r'\d{4}/\d{2}/\d{2}')
    dates = []
    for line in open(infile, 'r', encoding='utf-8'):
        match = re.search(datere, line)
        if match:
            dates.append(datetime(*(map(int, match.group().split('/')))))
    return dates


if __name__ == '__main__':
    main()