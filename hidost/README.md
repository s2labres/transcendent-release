## Reproducing Hidost Experiments
The model used for the Hidost experiments is a Random Forest (RF). As a non-conformity measure (NCM) we use the proporition of decision trees which disagree with the ensemble.
The original Transcendent codebase focused on Support Vector Machine (SVM) classifiers, and unfortunately we lacked the time needed to re-engineer the library. This patch is what was used 
to run the RF experiments in the paper on Hidost. The `rf.py` file is what can be used to have a RF class which is suitable for the Transcendent library to operate over, it is sufficient
to replace all instances of the SVM class in the original library with the `get_rf()` method. 

For example:


```python
...
svm = SVC(probability=True, kernel='linear', verbose=True)
svm.fit(X_proper_train, y_proper_train)
...
```

will then become (depending on where you place your `rf.py` file):

```python
from transcend.rf import get_rf
...
rf = get_rf()
rf.fit(X_proper_train, y_proper_train)
...
```

In `extract_hidost.py` you can also find a handy extraction script which may be useful to format the dataset in a more Transcendent-friendly way.