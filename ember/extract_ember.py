import ember
import numpy as np
from matplotlib import pyplot as plt
 
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from tesseract import evaluation, temporal, metrics, viz
import pickle 
 
"""
I apologise for the messy-ness of this script. It is an old script which was used 
for Ember data analysis as well through Tesseract. The Tesseract part is not necessary
so feel free to remove it when using it. It should be relatively straight forward to 
understand where the actual extraction is going on, that is all that is needed for the 
Transcendent experiments. 
"""
 
class Main:
    datasetPath = "/path/to/ember"
    x_final = []
    y_final = []
    t_final = []
 
    def __init__(self):
        self.process_data()
        self.plot()
 
    def process_data(self):
        '''
        Get data from dataset and load it as np array into list fields
        '''
        # ember.create_vectorized_features(self.datasetPath)
        # import data
        x_train, y_train = ember.read_vectorized_features(self.datasetPath, subset='train')
        x_test, y_test = ember.read_vectorized_features(self.datasetPath, subset='test')
 
        # re-declaring fields locally to avoid IDE issues
        x_final = []
        y_final = []
        t_final = []
 
        # position in input vector for timestamp
        timestamp_position = 626
 
        # it is significantly less expensive to append to lists and then convert them to numpy arrays after
        slice_train = len(x_train)
        slice_test = len(x_test)
 
        # filter data with wrong timestamps and is not classified (y_train[i] : -1)
        for i in range(slice_train):
            if 1514678400 > x_train[i][timestamp_position] > 1483228800 and y_train[i] != -1.0:
                x_final.append(x_train[i])
                y_final.append(y_train[i])
                t_final.append(datetime.fromtimestamp(x_train[i][timestamp_position]))
 
 
        for i in range(slice_test):
            if 1514678400 > x_test[i][timestamp_position] > 1483228800 and y_test[i] != -1.0:
                x_final.append(x_test[i])
                y_final.append(y_test[i])
                t_final.append(datetime.fromtimestamp(x_test[i][timestamp_position]))

        total_neg = len(y_final) - sum(y_final)
        total_pos = sum(y_final)
        print(f"Number of negative: {total_neg}")
        print(f"Number of positive: {total_pos}")
        print(f"Timespan: {min(t_final)} - {max(t_final)}")
        
        self.x_final = np.asarray(x_final)
        self.y_final = np.asarray(y_final)
        self.t_final = np.asarray(t_final)
 
        print(f"Total samples: {len(self.t_final)}")
 
    def plot(self):
        splits = temporal.time_aware_train_test_split(
            self.x_final, self.y_final, self.t_final,
            train_size=5, test_size=1, granularity='month',
        )
        X_train, X_test, y_train, y_test, t_train, t_test  = splits
        print(f"Training elements: {len(t_train)}")


        def dump(path, stuff):
            with open(path, 'wb') as f:
                pickle.dump(stuff, f)

        dump("features/ember_2018/train_X.p", X_train)
        dump("features/ember_2018/test_X.p", X_test)
        dump("features/ember_2018/train_y.p", y_train)
        dump("features/ember_2018/test_y.p", y_test)


        tot = 0
        for x in X_test:
            tot += len(x)

        print(f"Testing samples: {tot}")
        exit()
 
        # Perform a timeline  with the SVC and RFC
        svc_clf = LinearSVC(verbose=False)
        rfc_clf = RandomForestClassifier()
 
        svc_results = evaluation.fit_predict_update(svc_clf, *splits)
        rfc_results = evaluation.fit_predict_update(rfc_clf, *splits)
 
        metrics.print_metrics(svc_results)
        metrics.print_metrics(rfc_results)
 
        # AUT for both SVC and RFC
        svc_aut = [metrics.aut(svc_results, 'f1'), metrics.aut(svc_results, 'precision'), metrics.aut(svc_results, 'recall')]
        rfc_aut = [metrics.aut(rfc_results, 'f1'), metrics.aut(rfc_results, 'precision'), metrics.aut(rfc_results, 'recall')]
 
        print(svc_aut)
        print(rfc_aut)
 
        # Plot both results
        svc_plot = viz.plot_decay(svc_results)
        rfc_plot = viz.plot_decay(rfc_results)
 
        svc_plot.show()
        rfc_plot.show()
 
 
if __name__ == "__main__":
    process = Main()