import sys
from sklearn import datasets
from sklearn.cross_validation import train_test_split as tts
import pandas as pd
import numpy as np
from sklearn import preprocessing as prepros
from random import triangular as tri
import math


class Neuron:
    def __init__(self, num_of_attributes):
        self.weights = [tri(-1.0, 1.0) for _ in range(num_of_attributes + 1)]
        self.threshold = 0
        self.bias = -1
        self.activate_value = 0

    def calc_output(self, inputs):
        inputs = np.append(inputs, self.bias)
        weight_sum = 0

        for count, ele in enumerate(inputs):
            weight_sum += (self.weights[count] * ele)

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

    names = df.columns
    #n = names[:-1]

    return data.values, targets.values


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class Classifier:
    def __init__(self, num_layers, data, num_targets):
        self.layers = []
        self.all_results = []

        for i in range(0, num_layers):
            self.layers.append(self.make_layer(num_layers, i, data, num_targets))
            # self.layers.append([Neuron(len(self.layers[i - 1]) if i > 0 else data.shape[1])
            #                     for _ in range(int(input("How many neurons for layer " + str(i) + "? ")))])

    def train(self, data):
        for data_row in data:
            activation = self.results(data_row)
            self.all_results.append(activation)

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

    def print_results(self):
        for row in self.all_results:
            print(row[-1])

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

    # normalize the data
    train_data_std, test_data_std = standarize(train_data, test_data)

    classifier.train(train_data_std)
    get_accuracy(classifier.predict(test_data_std), test_target)

    #get_accuracy(classifier.predict(train_data_std, train_target, test_data_std), test_target)


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
    data, targets = load_dataset(datasets.load_iris())

    # pima indian diabetes
    # data, targets = load_file("pima.csv")

    # number of targets and list of targets
    num_targets= num_of_diff_targets(targets)

    # get the number of desired layers
    num_layers = get_number_of_layers()

    # make the classifier
    classifier = Classifier(num_layers, data, num_targets)

    data_processing(data, targets, classifier)


if __name__ == "__main__":
    main(sys.argv)