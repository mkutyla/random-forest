import pandas as pd
from sklearn import tree
import time
from sklearn.metrics import accuracy_score
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

if __name__ == '__main__':
    df = pd.read_csv('../data/wine.csv')
    target = 'quality'
    feature_cols = ['density', 'citric acid', 'sulphates']

    clf = tree.DecisionTreeClassifier(criterion='entropy',
                                      max_depth=len(feature_cols))
    X = df.loc[:, feature_cols]
    print(len(feature_cols))
    print(X.shape)

    Y = df.loc[:, target]
    print(Y.shape)

    start_time = time.time()
    clf = clf.fit(X, Y)
    print("--- %s seconds ---" % (time.time() - start_time))

    y_true = np.array(df[target].to_list())
    y_pred = np.array(clf.predict(X))
    print(accuracy_score(y_true, y_pred))

    print(tree.export_text(clf))

    # fig = plt.figure(figsize=(25,20))
    # _ = tree.plot_tree(clf,
    #                 feature_names=feature_cols,
    #                 class_names=,
    #                 filled=True)
