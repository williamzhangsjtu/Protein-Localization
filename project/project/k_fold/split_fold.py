import numpy as np
import pandas as pd 
import random
import os
from sklearn.model_selection import KFold

def kfold_split(df, outdir):
    length = len(df)
    indexes = np.arange(length)
    kf = KFold(2, True, 0)
    for idx, (train, test) in enumerate(kf.split(indexes)):

        train_set = df.values[train]
        dev_set = df.values[test]


        train_df = pd.DataFrame(train_set, columns=['id', 'label'])
        train_df.set_index('id').to_csv(os.path.join(outdir, 'train_{}.csv'.format(idx + 1)))
        test_df = pd.DataFrame(dev_set, columns=['id', 'label'])
        test_df.set_index('id').to_csv(os.path.join(outdir, 'test_{}.csv'.format(idx + 1)))

    #return train_set, dev_set



train_df = pd.read_csv("../train.csv", sep=',')
kfold_split(train_df, 'fold_2')