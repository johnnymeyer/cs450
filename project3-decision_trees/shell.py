import sys
from sklearn import datasets
from sklearn.cross_validation import train_test_split as tts
import pandas as pd
import numpy as np


class Node:
    def __init__(self):
        self.name = ""
        self.childNode = {}


class Classifier:
    #todo write what the classifer does
    def train(self, data_set, target_set, f_names):
        make_tree(data_set, target_set, f_names)

    def predict(self):
        pass


def all_same(items):
    return all(x == items[0] for x in items)


def calc_entropy_weighted_average(data, clas, feature):
    """
    This algorithm is based off of the calc_info_gain
    from the book, except this function just calculates
    the weighted entropy value
    :param data: the data from the dataset
    :param clas: the list of possible classes (targets)
    :param feature: the feature that we are calculating the
                    entropy for
    :return wei: the weighted average of entropy for this branch
    """
    num_data = len(data) # the number of data items
    # list of the possible values for a feature
    values = []
    for data_point in data:
        if data_point[feature] not in values:
            # get the names of the branches
            values.append(data_point[feature])

    feature_value_count = np.zeros(len(values))
    entropy = np.zeros(len(values))

    value_index = 0

    # loop for each value e.g. good, avg, and low in credit score
    for v in values:
        data_index = 0
        newClasses = []
        # loop for each row of the data
        for data_point in data:
            # e.g. if we are in good branch does the data point of a row
            #      equal good
            if data_point[feature] == v:
                feature_value_count[value_index] += 1
                newClasses.append(clas[data_index]) # e.g. newClasses = ['y','n','y','n'] for credit score branch good

            data_index += 1

        # array for the class values
        class_values = []
        for c in newClasses:
            # if the c is not in the array yet add it
            if class_values.count(c) == 0:
                class_values.append(c)

        # array containing the number of each value of the class
        num_class_values = np.zeros(len(class_values))

        class_index = 0
        # find the number of yes and no for each good in credit score
        # e.g. credit score - good branch - would contain 2 2
        for cv in class_values:
            for c in newClasses:
                if c == cv:
                    num_class_values[class_index] +=1
            class_index += 1

        # calculate the entropy getting the fractional part to put into the entropy function
        for i in range(len(class_values)):
            entropy[value_index] += calculate_entropy(float(num_class_values[i]) / sum(num_class_values))

        # weight the entropy
        weight = feature_value_count[value_index] / num_data
        entropy[value_index] = (entropy[value_index] * weight)
        value_index += 1

    return sum(entropy)


def calculate_entropy(p):
    if p != 0:
        return -p * np.log2(p)
    else:
        return 0


def make_tree(data, classes, f_names):
    num_f_names = len(f_names)
    if all_same(classes):
        n = Node()
        n.name = classes[0]
        return n
    elif num_f_names == 1:
        most_common_class = np.argmax(classes)
        n = Node()
        n.name = most_common_class
        return n
    else:
        # todo switch back to loop
        calc_entropy_weighted_average(data, classes, 0)
        #for name in range(num_f_names):
            #calc_entropy_weighted_average(data, classes, name)


def get_accuracy(results, test_tar):
    """
    Calculates the accuracy of the predictions. Will also
    display the results
    :param results: The results of the predictions
    :param test_tar: the actual values of the data
    :return: NONE
    """
    number_correct = 0

    for i in range(test_tar.size):
        if results[i] == test_tar[i]:
            number_correct = number_correct + 1

    print("\nNumber of correct predictions:", number_correct, " of ", test_tar.size)
    print("Accuracy rate is {0:.2f}%".format((number_correct / test_tar.size) * 100))


def data_processing(d_data, d_target, classifier, feature_names):
    # user input for how much should be test and random state being used
    ts = 0
    while ts < .1 or ts > .5:
        ts = float(input("Percentage of data for testing (Enter value between .1 and .5): "))

    rs = 0
    while rs <= 0:
        rs = int(input("Random state for shuffling (Enter positive integer): "))


    # split the data into test and training sets after it shuffles the data
    train_data, test_data, train_target, test_target = tts(d_data, d_target, test_size=ts, random_state=rs)

    # todo switch back to buttom classifier
    classifier.train(d_data, d_target, feature_names)
    #classifier.train(train_data, train_target, feature_names)
    #get_accuracy(classifier.predict(train_data, train_target, test_data), test_target)


def load_dataset(set):
    """
    Loads the dataset from the sklean and splits the data into
    the data and the target
    :param set: the set that is to be loaded in from sklearn
    :return: the data and the target of the set
    """
    return set.data, set.target


def load_file(file):
    """
    Will split the dataset into the data and the targets
    if being read from a csv file.
    :param file: the name of the file to be read in
    :return: the data and the targets of the set
    """

    df = pd.read_csv(file)

    data = df.ix[:, df.columns != "className"]
    targets = df.ix[:, df.columns == "className"]

    names = df.columns
    n = names[:-1]

    return data.values, targets.values, n


def main(argv):
    # load the data from the database - choose which data set you want to use

    # iris data
    #data, targets = load_dataset(datasets.load_iris())

    # breast cancer data
    #data, targets = load_dataset(datasets.load_breast_cancer())

    # car data -- local file
    data, targets, feature_names = load_file('loan.csv')

    # make the classifier
    classifier = Classifier()

    data_processing(data, targets, classifier, feature_names)


if __name__ == "__main__":
    main(sys.argv)