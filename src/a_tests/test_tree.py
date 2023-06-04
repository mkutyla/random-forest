import asyncio
import pickle
import random

import numpy as np

import src.utilities.dataframes_handler as dfh
import src.utilities.dataframes_handler as ht
import src.utilities.quality as qt
import src.tree.decision_tree as dtr


class TreeTester:
    def __init__(self, set_name):
        if set_name not in ht.params.keys():
            raise Exception(f'[!] Invalid set_name! Try: {ht.params.keys()}')

        self.params = ht.params[set_name]
        self.df = ht.read_file(self.params["filename"])
        self.target = self.params["target"]
        self.quality = qt.Quality(dfh.get_attribute_domain(self.df, self.target))

        self.attributes = self.df.columns.tolist()
        self.attributes.remove(self.target)

        self.chosen_attributes = []
        self.predicted = None
        self.training_set = None
        self.test_set = None

    def get_random_attributes(self, amount: int = 0):
        if amount == 0:
            amount = int(np.sqrt(len(self.attributes)))  # default
        return random.sample(self.attributes, amount)

    def fit_and_predict(self, training_set, test_set, chosen_attributes, mode: str = 'mean'):
        dt = dtr.DecisionTree(self.target, self.params["continuous"])
        dt.fit(training_set, chosen_attributes, split_strategy=mode)
        predicted = dt.predict(test_set)
        return predicted

    def test_attribute_amount(self):

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        x, y, labels = loop.run_until_complete(self.tree_loop_amount())

        self.quality.plot_roc(x, y, labels)

    async def tree_loop_amount(self):
        tasks = []
        labels = []

        length = len(self.attributes)
        n_iterations = 1

        for amount in range(1, length+1):
            task = asyncio.ensure_future(self.tree_loop_amount_single_test(n_iterations, amount))
            tasks.append(task)
            labels.append(amount)

        results = await asyncio.gather(*tasks)

        x = [1 - item[1] / n_iterations for item in results]  # fpr
        y = [item[0] / n_iterations for item in results]

        return x, y, labels

    async def tree_loop_amount_single_test(self, n_iterations: int, amount: int):
        tasks = []
        for _ in range(n_iterations):
            task = asyncio.ensure_future(self.tree_iteration_amount(amount))
            tasks.append(task)
        results = await asyncio.gather(*tasks)
        tpr = sum([item[0] for item in results])
        tnr = sum([item[1] for item in results])

        return tpr, tnr

    async def tree_iteration_amount(self, amount):
        training_set, test_set = dfh.split_set(self.df, self.target)

        chosen_attributes = self.get_random_attributes(amount)
        predicted = self.fit_and_predict(training_set=training_set,
                                         test_set=test_set,
                                         chosen_attributes=chosen_attributes)

        quality = qt.Quality(dfh.get_attribute_domain(self.df, self.target))
        quality.set_predicted(predicted)
        quality.set_expected(test_set[self.target])
        quality.get_confusion_matrix()
        measures = quality.get_measures()

        tpr = measures["TPR"]
        tnr = measures["TNR"]

        return tpr, tnr

    @staticmethod
    def save(dt: dtr.DecisionTree):
        with open("../output/dt.pkl", "wb") as output:
            pickle.dump(dt, output)


if __name__ == "__main__":
    tester = TreeTester('wine')
    tester.test_attribute_amount()
