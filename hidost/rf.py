from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

class RF(RandomForestClassifier):
    def __init__(self):
        super().__init__(n_estimators=200)

    def decision_function(self, X, y):
        n_of_trees = self.n_estimators
        
        one_preds_count = 0

        for tree in range(n_of_trees):
            if self.estimators_[tree].predict(X)[0] == 1.:
                one_preds_count += 1

        if y == 1:
            # when one_preds_count = n_of_trees (perfect consensus), then ncm = 0 (lowest)
            # when one_preds_count = 0 (lowest consesus), then ncm = 1 (highest)
            ncm = 1 - (one_preds_count / n_of_trees)
        elif y == 0:
            # when one_preds_count = n_of_trees (lowest consensus), then ncm = 1 (highest)
            # when one_preds_count = 0 (highest consensus), then ncm = 0 (lowest)
            ncm = one_preds_count / n_of_trees

        return ncm

def get_rf():
    return RF()