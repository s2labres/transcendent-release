## Reproducing Ember Experiments
The model used for the Ember experiments is a Gradient Boosting Decision Trees (GBDT) classifier. As a non-conformity measure (NCM) we use the built-in Scikit-learn `decision_function` method.
The original Transcendent codebase focused on Support Vector Machine (SVM) classifiers, and unfortunately we lacked the time needed to re-engineer the library. To replicate the Transcendent experiments
on Ember, it is sufficient to replace all instances of the SVM class in the original library. 

For example:


```python
...
svm = SVC(probability=True, kernel='linear', verbose=True)
svm.fit(X_proper_train, y_proper_train)
...
```

will then become:

```python
from sklearn.ensemble import GradientBoostingClassifier
...
gbdt = GradientBoostingClassifier(learning_rate=0.05, n_iter_no_change=1000, max_depth=15, min_samples_leaf=50, verbose=False)
gbdt.fit(X_proper_train, y_proper_train)
...
```

In `extract_ember.py` you can also find a handy extraction script which may be useful to format the dataset in a more Transcendent-friendly way.