import numpy as np
from hellingerdistancecriterion.hellinger_distance_criterion import HellingerDistanceCriterion
from sklearn.ensemble import RandomForestClassifier

hdc = HellingerDistanceCriterion(1, np.array([2],dtype='int64'))
clf = RandomForestClassifier(criterion=hdc, max_depth=4, n_estimators=100)
clf.fit(X_train, y_train)
print('hellinger distance score: ', clf.score(X_test, y_test))