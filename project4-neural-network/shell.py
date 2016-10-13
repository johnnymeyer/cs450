import sys
from sklearn import datasets
from sklearn.cross_validation import train_test_split as tts
import pandas as pd
import numpy as np
from sklearn import preprocessing as prepros


def load_dataset(s):
    return s.data, s.target


def load_file(file):
    df = pd.read_csv(file, header=None)

    ds_len_column = len(df.columns)

    data = df.loc[:, : ds_len_column - 2]
    targets = df.ix[:, ds_len_column - 1: ds_len_column - 1]

    return data.values, targets.values


class Classifier:
    def train(self, data_set, target_set):
        pass

    def predict(self, k, data, data_class, inputs):
        pass


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
            number_correct = number_correct + 1

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

    k_value = 0
    while k_value <= 0:
        k_value = int(input("k value (Enter positive integer): "))

    # split the data into test and training sets after it shuffles the data
    #train_data, test_data, train_target, test_target = tts(dataset.data, dataset.target, test_size=ts, random_state=rs)
    train_data, test_data, train_target, test_target = tts(d_data, d_target, test_size=ts, random_state=rs)

    # normalize the data
    train_data_std, test_data_std = standarize(train_data, test_data)

    get_accuracy(classifier.predict(k_value, train_data_std, train_target, test_data_std), test_target)


def main(argv):
    # load the data from the database - choose which data set you want to use

    # iris data
    data, targets = load_dataset(datasets.load_iris())

    # breast cancer data
    #data, targets = load_dataset(datasets.load_breast_cancer())

    # car data -- local file
    #data, targets = load_file('car1.csv')

    # make the classifier
    classifier = Classifier()

    data_processing(data, targets, classifier)


if __name__ == "__main__":
    main(sys.argv)