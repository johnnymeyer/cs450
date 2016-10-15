import sys
from sklearn import datasets
from sklearn.cross_validation import train_test_split as tts
import pandas as pd
import numpy as np
from sklearn import preprocessing as prepros
from random import triangular as tri


class Neuron:
    def __init__(self, num_of_attributes):
        self.weights = [tri(-1.0, 1.0) for _ in range(num_of_attributes + 1)]
        self.threshold = 0
        self.bias = -1

    def calc_output(self, inputs):
        inputs = np.append(inputs, self.bias)
        weight_sum = 0

        for count, ele in enumerate(inputs):
            weight_sum += (self.weights[count] * ele)

        return 0 if weight_sum < self.threshold else 1


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


class Classifier:
    def __init__(self, num_neurons, atribute_count):
        self.neurons = [Neuron(atribute_count) for _ in range(num_neurons)]

    def train(self, data_set, target_set):
        pass

    def predict(self, k, data, data_class, inputs):
        pass

    def results(self, inputs):
        return [neuron.calc_output(inputs) for neuron in self.neurons]

    def print_outputs(self, data):
        for data_row in data:
            print(self.results(data_row))


def standarize(train, test):
    # standardizing logic from http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
    std_scale = prepros.StandardScaler().fit(train)
    train_std = std_scale.transform(train)
    test_std = std_scale.transform(test)

    return train_std, test_std


def get_accuracy(results, test_tar):
    number_correct = 0

    for i in range(test_tar.size):
        if results[i] == test_tar[i]:
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

    classifier.print_outputs(train_data_std)

    #get_accuracy(classifier.predict(train_data_std, train_target, test_data_std), test_target)


def main(argv):
    # load the data from the database - choose which data set you want to use

    # iris data
    data, targets = load_dataset(datasets.load_iris())

    # pima indian diabetes
    #data, targets = load_file("pima.csv")

    # get the number of attributes
    num_column = len(data[0])

    num_neuron = 0
    while num_neuron < 1:
        num_neuron = int(input("Number of Neurons: "))

    # make the classifier
    classifier = Classifier(num_neuron, num_column)

    data_processing(data, targets, classifier)


if __name__ == "__main__":
    main(sys.argv)