from Node import *
from TreeDrawer import *
import random
import numpy as np
import pandas as pd
import pickle
import statistics


class DecisionTree:
    def __init__(
        self, class_attribute: str, continuous_attr: list[str]
    ) -> "DecisionTree":
        self.cls = class_attribute
        self.continuous_attr = continuous_attr
        self.root = None

    def get_classes(self, S: pd.DataFrame) -> list:
        return list(set(S[self.cls]))

    def get_best_class(self, S: pd.DataFrame):
        # getting all classes
        classes_counters = S[self.cls].value_counts().to_dict()
        # finding most frequent class
        max_counter = max(classes_counters.values())
        # finding most frequent classes
        max_classes = [k for k, v in classes_counters.items() if v == max_counter]

        # return best class or select one of the best at random
        return max_classes[random.randint(0, len(max_classes) - 1)]

    def get_attribute_domain(self, S: pd.DataFrame, A: str) -> set:
        return set(S[A])

    def threshold(self, S: pd.DataFrame, X: list[str]) -> dict:
        information_gain = dict()
        if len(X) == 0:
            return information_gain

        src_entropy = self.entropy(S)
        for A in X:
            conditional, split = self.conditional_entropy(S, A)
            IG = src_entropy - conditional
            if IG == 0:  # thresholding no IG
                continue
            information_gain[A] = [IG, split]

        # thresholding TODO: pop worst splits

        return information_gain

    def entropy(self, S: pd.DataFrame) -> float:
        probs = list(S[self.cls].value_counts(normalize=True).to_dict().values())
        return -1 * float(np.sum(probs * np.log2(probs)))

    def conditional_entropy(self, S: pd.DataFrame, A: str) -> list[float, float]:
        if A in self.continuous_attr:
            best_split = 0
            best_entropy = float("inf")

            values = list(set(S[A].to_list()))  # eliminating duplicates
            values.sort()

            for first, second in zip(values[:-1], values[1:]):
                # split = (first + second) / 2
                # TODO: remove if previous split gets optimized
                split = statistics.mean(values)
                df = S[S[A] <= split]
                df2 = S[S[A] > split]

                p1 = self.entropy(df)
                p2 = self.entropy(df2)
                H = len(df) / len(S) * p1 + len(df2) / len(S) * p2

                if H < best_entropy:
                    best_entropy = H
                    best_split = split
                break
            # only possible if len(values)=1 <=> it's not worth to split
            if best_entropy == float("inf"):
                best_entropy = self.entropy(S)

            return [best_entropy, [best_split]]
        else:
            H = 0
            Xs = self.get_attribute_domain(S, A)
            for x in Xs:
                df = S.where(S[A] == x)
                H += len(df) / len(S) * self.entropy(S.where(S[A] == x))
            return [H, Xs]

    def add_branch(self, other_node: "Node", split_by) -> None:
        self.root.add_child_branch(other_node, split_by)

    def rand_attr_by_IG(self, attr: list[str], IGs: list[float]):
        denom = sum(IGs)
        probs = [x / denom for x in IGs]
        # random attribute, probability linear to split's IG
        selected = int(np.random.choice(np.arange(0, len(attr)), p=probs))
        # best attribute
        # best = [id for id, x in enumerate(probs) if x == max(probs)]
        # selected = int(np.random.choice(np.arange(0, len(best), 1)))
        choice = attr[selected]
        return choice

    def get_next_node(
        self, S: pd.DataFrame, Sx: pd.DataFrame, X: list[str], A: str
    ) -> Node:
        next_node = Node()
        if len(Sx) == 0:
            next_node.cls = self.get_best_class(S)
        else:
            Tx = DecisionTree(self.cls, self.continuous_attr)
            # controlling tree's splits
            next_node = Tx.build_tree(Sx, [x for x in X if x != A])

            # best for fitting almost perfectly to TS
            # next_node = Tx.build_tree(Sx, X)

        return next_node

    def build_tree(self, S: pd.DataFrame, X: list[str]) -> Node:
        self.root = Node()

        if len(c := self.get_classes(S)) == 1:
            self.root.cls = c[0]
            return self.root

        _X = self.threshold(S, X)
        if len(_X) == 0:
            self.root.cls = self.get_best_class(S)
            return self.root

        A = self.rand_attr_by_IG(list(_X.keys()), [x[0] for x in _X.values()])
        self.root.label = A
        split_by = _X[A][1]

        if len(split_by) == 1:  # continuous attribute
            x = split_by[0]

            Sx = S[S[A] <= x]
            next_node1 = self.get_next_node(S, Sx, X, A)
            Sx = S[S[A] > x]
            next_node2 = self.get_next_node(S, Sx, X, A)

            if next_node1.is_leaf() and next_node2.is_leaf():
                if next_node1.cls == next_node2.cls:
                    # remove unnecessary split, change to leaf
                    self.root.transform_to_leaf(next_node1.cls)
                    return self.root

            self.add_branch(next_node1, f"<={x}")
            self.add_branch(next_node2, f">{x}")
        else:
            for x in split_by:
                Sx = S[S[A] == x]
                next_node = self.get_next_node(S, Sx, X, A)
                self.add_branch(next_node, f"={x}")

        return self.root  # in case it's used recursively as above

    def classify(self, df: pd.DataFrame, prediction_label: str = None) -> pd.Series:
        if self.root is None:
            Exception("The tree was not trained yet!")

        if prediction_label is None:
            prediction_label = self.cls + " prediction"

        df[prediction_label] = None
        df.loc[:, prediction_label] = self.__clasify(self.root, df, prediction_label)
        prediction = df[prediction_label].copy()
        df.drop(columns=prediction_label, inplace=True)
        return prediction

    def __clasify(self, node: Node, df: pd.DataFrame, label: str):
        if node.is_leaf():
            df.loc[df[label].isna(), label] = node.cls
            return df[label]

        attr = node.label
        for branch in node.children_branches:
            condition = str(branch.label)
            if "<=" in condition:
                # cut off "<="
                val = float(condition[2:])
                cond = df[attr] <= val
                next = branch.child
                df.loc[cond, label] = self.__clasify(next, df[cond], label)
            elif ">" in condition:
                # cut off ">"
                val = float(condition[1:])
                cond = df[attr] > val
                next = branch.child
                df.loc[cond, label] = self.__clasify(next, df[cond], label)
            else:
                # discrete attribute
                # cut off "="
                val = float(condition[1:])
                cond = df[attr] == val
                for a in self.get_attribute_domain(df, attr):
                    cond = df[attr] == a
                    next = branch.child
                    df.loc[cond, label] = self.__clasify(next, df[cond], label)
        return df[label]


def run():
    df = pd.read_csv("../data/wine.csv")
    target = "quality"
    attributes = df.columns.tolist()
    attributes.remove(target)

    # remove empty continuous attributes if any
    for a in attributes:
        df[a] = df[a].replace(np.NaN, df[a].mean())

    # get random attrs
    amount = int(np.sqrt(len(attributes)))
    attributes = random.sample(attributes, amount)
    print(f"Attributes selected for classification: {attributes}")

    dt = DecisionTree(target, attributes)
    dt.build_tree(df, attributes)
    predicted = dt.classify(df)
    expected = df[target]
    l = len(expected)
    correct = len([x for (x, y) in zip(expected, predicted) if x == y])
    incorrect = l - correct
    print(f"Correct:{correct} out of {l} error={incorrect/l}")
    #
    # Multi-dimensional confusion-matrix-style errors
    #
    # correct = df[df[target + " prediction"] == df[target]].count()[target]

    # all = df[target].value_counts().to_dict()
    # invalid = df[df[target + " prediction"] !=
    #              df[target]][target].value_counts().to_dict()
    # all = dict(sorted(all.items(), key=lambda item: item[0]))
    # invalid = dict(sorted(invalid.items(), key=lambda item: item[0]))

    # error = dict()
    # print("Error rate:")
    # for k in all.keys():
    #     if k not in invalid.keys():
    #         invalid[k] = 0
    #     error[k] = "{:.2f}%".format(invalid[k] / all[k] * 100)
    #     print(k, " : ", error[k])

    # error = "{:.2f}%".format((len(df) - correct) / len(df) * 100)

    # print(f"{correct} out of {len(df)} => error={error}")

    with open("dt.pkl", "wb") as output:
        pickle.dump(dt, output)


def load():
    with open("dt.pkl", "rb") as input:
        dt = pickle.load(input)
        draw_tree(dt)


if __name__ == "__main__":
    run()
    load()
