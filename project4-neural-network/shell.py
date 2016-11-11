import sys
from sklearn import datasets
from sklearn.cross_validation import train_test_split as tts
import pandas as pd
import numpy as np
from sklearn import preprocessing as prepros
from random import triangular as tri
import math
import matplotlib.pyplot as plt
from sklearn import neural_network as MLP


class Neuron:
    def __init__(self, num_of_attributes):
        self.weights = [tri(-1.0, 1.0) for _ in range(num_of_attributes + 1)]
        # self.weights = [.1, -.3, .2]
        self.threshold = 0
        self.bias = -1
        self.activate_value = 0
        self.error = None

    def calc_output(self, inputs):
        inputs = np.append(inputs, self.bias)
        weight_sum = 0

        for count, ele in enumerate(inputs):
            weight_sum += (self.weights[count] * ele)

        # print("h value", weight_sum)

        self.activate_value = sigmoid(weight_sum)

        return self.activate_value


def load_dataset(s):
    return s.data, s.target


def load_file(file):
    """
    Will split the dataset into the data and the targets if being read from a csv file.
    :param file: the name of the file to be read in
    :return: the data and the targets of the set
    """

    df = pd.read_csv(file)

    data = df.ix[:, df.columns != "className"]
    targets = df.ix[:, df.columns == "className"]

    # names = df.columns
    # n = names[:-1]

    return data.values, targets.values


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class Classifier:
    def __init__(self, num_layers, data, num_targets):
        self.layers = []
        self.learning_rate = .2
        # self.all_results = []

        for i in range(0, num_layers):
            self.layers.append(self.make_layer(num_layers, i, data, num_targets))
            # self.layers.append([Neuron(len(self.layers[i - 1]) if i > 0 else data.shape[1])
            #                     for _ in range(int(input("How many neurons for layer " + str(i) + "? ")))])

    def train(self, data, targets):
        num_epochs = 0
        accuracy = []
        while num_epochs < 1:
            num_epochs = int(input("Number of Epochs: "))

        for _ in range(num_epochs):
            all_results = []
            pred = []

            for data_row, target_row in zip(data, targets):
                activation = self.results(data_row)
                all_results.append(activation)
                pred.append(self.results(data_row)[-1])
                self.update(data_row, target_row, activation)

            accuracy.append(100 * sum([targets[i] == p.index(max(p)) for i, p in enumerate(pred)]) / len(targets))
            print("Accuracy for epoch", _ + 1, ":", accuracy[_])

        if input("Would you like to see the accuracy graph? (y/n): ") == 'y':
            plt.plot(range(1, num_epochs + 1), accuracy)
            plt.ylabel('ACCURACY')
            plt.xlabel('EPOCH')
            plt.show()

    def update(self, d_row, t_row, a_values):
        self.calc_error(t_row, a_values)
        self.update_all_weights(d_row, a_values)

    def calc_error(self, tar, a_values):
        for index_l, lay in reversed(list(enumerate(self.layers))):
            for index_n, neu in enumerate(lay):
                if index_l < len(self.layers) - 1:  # hidden layer
                    # print("act[", index_l, index_n, "]", a_values[index_l][index_n])
                    neu.error = self.error_hidden_node(a_values[index_l][index_n],
                                                       [n.weights[index_n] for n in self.layers[index_l + 1]],
                                                       [n.error for n in self.layers[index_l + 1]])
                    # print("hidden node error", "layer", index_l, "neuron", index_n, neu.error)
                else:  # output layer
                    # print("act[", index_l, index_n, "]", a_values[index_l][index_n])
                    # print("tar", tar)
                    neu.error = self.error_output_node(a_values[index_l][index_n], index_n == tar)
                    # print("output node error", "layer", index_l, "neuron", index_n, neu.error)

    def update_all_weights(self, d_row, a_values):
        for i, lay in enumerate(self.layers):
            for n in lay:
                self.update_neuron_weight(n, a_values[i - 1] if i > 0 else d_row.tolist())

    def update_neuron_weight(self, neuron, inputs):
        inputs = inputs + [-1]
        neuron.weights = [weight - self.learning_rate * inputs[i] * neuron.error
                          for i, weight in enumerate(neuron.weights)]

    def error_output_node(self, act, tar):
        delta = act * (1 - act) * (act - tar)
        return delta

    def error_hidden_node(self, act, weights, delta_ks):
        delta = act * (1 - act) * sum([w * d for w, d in zip(weights, delta_ks)])
        return delta

    def predict(self, data_set):
        prediction = []
        for d in data_set:
            prediction.append(self.results(d)[-1])
        return prediction

    def results(self, inputs):
        res = []
        for index, layer in enumerate(self.layers):
            res.append([neuron.calc_output(res[index - 1] if index > 0 else inputs) for neuron in layer])
        return res

    def make_layer(self, num_layers, layer_num, data, num_targets):
        # hidden
        if layer_num > 0 and layer_num < num_layers - 1:
            return [Neuron(len(self.layers[layer_num - 1]))
                    for _ in range(int(input("Num Neurons for layer " + str(layer_num) + "? ")))]
        # first or input layer
        elif layer_num == 0:
            return [Neuron(data.shape[1])
                    for _ in range(int(input("Num Neurons for layer " + str(layer_num) + "? ")))]
        # last or output layer
        else:
            return [Neuron(len(self.layers[layer_num - 1])) for _ in range(num_targets)]


def standarize(train, test):
    # standardizing logic from http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
    std_scale = prepros.StandardScaler().fit(train)
    train_std = std_scale.transform(train)
    test_std = std_scale.transform(test)

    return train_std, test_std


def get_accuracy(results, test_tar):
    number_correct = 0

    for r, tt in zip(results, test_tar):
        if tt == r.index(max(r)):
            number_correct += 1

    print("\nNumber of correct predictions:", number_correct, " of ", test_tar.size)
    print("Accuracy rate is {0:.2f}%".format((number_correct / test_tar.size) * 100))


def accuracy_for_sklearn_mlp(res, tt):
    num_cor = 0

    for r, t in zip(res, tt):
        if r == t:
            num_cor += 1

    print("\nNumber of correct predictions:", num_cor, " of ", tt.size)
    print("Accuracy rate is {0:.2f}%".format((num_cor / tt.size) * 100))


def data_processing(d_data, d_target, classifier):
    # user input for how much should be test and random state being used
    ts = 0
    while ts < .1 or ts > .5:
        ts = float(input("Percentage of data for testing (Enter value between .1 and .5): "))

    rs = 0
    while rs <= 0:
        rs = int(input("Random state for shuffling (Enter positive integer): "))

    # split the data into test and training sets after it shuffles the data
    train_data, test_data, train_target, test_target = tts(d_data, d_target, test_size=ts, random_state=rs)

    # c = MLP.MLPClassifier()
    # c.fit(train_data, train_target)
    # ans = c.predict(test_data)
    # accuracy_for_sklearn_mlp(ans, test_target)

    # normalize the data
    train_data_std, test_data_std = standarize(train_data, test_data)

    # train
    classifier.train(train_data_std, train_target)

    # compute accuracy
    get_accuracy(classifier.predict(test_data_std), test_target)


def num_of_diff_targets(targets):
    tar = []
    for t in targets:
        if t not in tar:
            tar.append(t)
    return len(tar)


def get_number_of_layers():
    lay = 0
    while lay < 1:
        lay = int(input("Number of Layers: "))
    return lay


def main(argv):
    # load the data from the database - choose which data set you want to use

    # iris data
    # data, targets = load_dataset(datasets.load_iris())

    # pima indian diabetes
    data, targets = load_file("pima.csv")

    # breast_cancer
    # data, targets = load_dataset(datasets.load_breast_cancer())

    # number of targets and list of targets
    num_targets = num_of_diff_targets(targets)

    # get the number of desired layers
    num_layers = get_number_of_layers()

    # make the classifier
    classifier = Classifier(num_layers, data, num_targets)

    data_processing(data, targets, classifier)


if __name__ == "__main__":
    main(sys.argv)
