from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score)
import numpy as np
y_true = np.array([0, 0, 1, 1, 1])
y_pred = np.array([0, 0, 1, 1, 0])
print('truth: {}'.format(y_true))
print('pred: {}'.format(y_pred))
print('accuracy: {}'.format(accuracy_score(y_true, y_pred)))
print('precision: {}'.format(precision_score(y_true, y_pred)))
print('recall: {}'.format(recall_score(y_true, y_pred)))

