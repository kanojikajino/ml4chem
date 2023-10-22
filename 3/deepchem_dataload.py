import matplotlib.pyplot as plt
import numpy as np
import deepchem as dc

featurizer = dc.feat.RDKitDescriptors()
tasks, datasets, transformers \
    = dc.molnet.load_bace_regression(featurizer)
train_set, val_set, test_set = datasets
print('train_set: \n{}'.format(
    train_set.metadata_df.iloc[:, 5:7]))
print('val_set: \n{}'.format(val_set.metadata_df.iloc[:, 5:7]))
print('test_set: \n{}'.format(test_set.metadata_df.iloc[:, 5:7]))
print('y_mean, y_std = {}, {}'.format(
    np.mean(train_set.y), np.std(train_set.y)))

plt.hist(train_set.y, bins=int(np.sqrt(len(train_set.y))))
plt.xlabel('Normalized pIC50')
plt.savefig('bace_y_train.pdf')
plt.clf()
