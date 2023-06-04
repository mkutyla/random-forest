import graphviz
import pandas as pd
from sklearn import tree
import time
from sklearn.metrics import accuracy_score
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import random

if __name__ == '__main__':
    df = pd.read_csv('../../data/wine.csv')
    target = 'quality'
    attributes = df.columns.tolist()
    attributes.remove(target)

    # remove empty continuous attributes if any
    for a in attributes:
        df[a] = df[a].replace(np.NaN, df[a].mean())

    # get random attrs
    amount = int(np.sqrt(len(attributes)))
    attributes = random.sample(attributes, amount)

    clf = tree.DecisionTreeClassifier(criterion='entropy',
                                      max_depth=len(attributes))
    X = df.loc[:, attributes]
    Y = df.loc[:, target]



    start_time = time.time()
    clf = clf.fit(X, Y)
    _ = tree.export_text(clf)
    print(_)
    print("--- %s seconds ---" % (time.time() - start_time))

    y_true = np.array(df[target].to_list())
    y_pred = np.array(clf.predict(X))

    print(accuracy_score(y_true, y_pred))
    dot_data = tree.export_graphviz(clf)
    print(dot_data)
    graph = graphviz.Source(dot_data)
    graph.format = "png"
    graph.render("../../tree.png")
    # fig = plt.figure(figsize=(25,20))
    # _ = tree.plot_tree(clf,
    #                 feature_names=feature_cols,
    #                 class_names=,
    #                 filled=True)
