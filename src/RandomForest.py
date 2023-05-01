import DecisionTree as DT
import pandas as pd
import numpy as np
import random
import time


class RandomForest:
    def __init__(self, df: pd.DataFrame, continuous_attributes: list[str]) -> None:
        self.df = df
        self.continuous_attributes = continuous_attributes
        self.preprocess_set()

    def preprocess_set(self):
        """
        Handles NaN values
        """
        for a in self.continuous_attributes:
            self.df[a] = self.df[a].replace(np.NaN, self.df[a].mean())

    def shuffle_sets(self) -> list[pd.DataFrame]:
        ts_size = 7
        s_size = 3
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        df_classes = [self.df[self.df[target] == c] for c in self.df[target].unique()]
        TS = None
        S = None
        for df_class in df_classes:
            splits = np.array_split(df_class, ts_size + s_size)
            for split in splits[:s_size]:
                if S is None:
                    S = split
                else:
                    S = S._append(split, ignore_index=True)
            for split in splits[s_size:]:
                if TS is None:
                    TS = split
                else:
                    TS = TS._append(split, ignore_index=True)

        return [df, S]

    def build_forest(self, target: str, N: int) -> list:
        TS, S = self.shuffle_sets()
        self.trees = list()
        attributes = self.df.columns.tolist()
        attributes.remove(target)
        predictions = pd.DataFrame()
        p = []

        for _ in range(0, N):
            T = TS.sample(replace=True, frac=1)
            amount = int(np.sqrt(len(attributes)))
            X = random.sample(attributes, amount)
            tree = DT.DecisionTree(target, attributes)
            tree.build_tree(T, X)
            self.trees.append(tree)
            pr = tree.classify(S)
            predictions[_] = pr

        most_frequent = predictions.mode(axis=1, dropna=True)
        return [S[target], list(most_frequent[0])]


if __name__ == "__main__":
    df = pd.read_csv("../data/wine.csv")
    target = "quality"
    attributes = df.columns.tolist()
    attributes.remove(target)

    rf = RandomForest(df, attributes)
    start_time = time.time()
    [expected, predicted] = rf.build_forest(target, 200)
    print("--- %s seconds ---" % (time.time() - start_time))
    l = len(expected)
    correct = len([x for (x, y) in zip(expected, predicted) if x == y])
    incorrect = l - correct
    print(f"Correct:{correct} out of {l} error={incorrect/l}")
