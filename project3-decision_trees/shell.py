import sys
from sklearn import datasets
from scipy import stats
from sklearn.cross_validation import train_test_split as tts
import pandas as pd
import numpy as np


class Node:
    def __init__(self, name="", child_node={}):
        self.name = name
        self.child_node = child_node


class Classifier:
    def train(self, data_set, target_set, f_names):
        a = make_tree(data_set, target_set, f_names)
        print("tree")
        print(a)
        return a

    def predict(self, tree, test_data, f_names):
        f_names = f_names.tolist()
        first = list(tree.keys())[0]
        first_index = f_names.index(first)

        for row in test_data:
            # row[first_index]
            pass



        # return the resluts of prediction


def get_values(info, col):
    """
    Returns a list of possible values for a certain feature
    :param info: the data set
    :param col: the col that is desired to find the possible values
    :return values: list of the possible values
    """
    values = []
    for data_point in info:
        if data_point[col] not in values:
            # get the names of the branches
            values.append(data_point[col])
    return values


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
    :return entropy: the weighted average of entropy for this branch
    """
    num_data = len(data)  # the number of data items

    # list of the possible values for a feature
    values = get_values(data, feature)

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
    num_data = len(data)

    # if all_same(classes):
    #     # n = Node(classes[0])
    #     # return n
    #     return classes[0]
    #
    # elif num_f_names == 0:
    #     # most_common_class = np.argmax(classes)
    #     # n = Node(f_names[most_common_class])
    #     # return n
    #     g = classes[np.argmax(classes)]
    #     print(classes)
    #     return g

    g = stats.mode(classes)
    g = g[0]

    if num_data == 0 or num_f_names == 0:

        #g = mode(x)
        #print(g)
        return g

    elif len(classes[0]) == num_data:
        return classes[0]

    else:
        entropy_totals = np.zeros(num_f_names) # list for all the entropies

        # loop through each feature to and calculate each entropy
        for name in range(num_f_names):
            entropy_totals[name] = calc_entropy_weighted_average(data, classes, name)

        # the lowest entropy is the best feature
        best_feature = np.argmin(entropy_totals)

        values = get_values(data, best_feature)

        #tree = Node(f_names[best_feature])
        tree = {f_names[best_feature]:{}}

        #
        new_data = []
        new_class = []

        for value in values:
            index = 0
            for d_p in data:
                if d_p[best_feature] == value:
                    #d_p = data_point.tolist()
                    if best_feature == 0:
                        data_point = d_p[1:]
                        new_names = f_names[1:]
                    elif best_feature == num_f_names:
                        data_point = d_p[:-1]
                        new_names = f_names[:-1]
                    else:
                        data_point = d_p[:best_feature]
                        if isinstance(data_point, np.ndarray):
                            data_point = data_point.tolist()
                        data_point.extend(d_p[best_feature+1:])
                        # data_point = np.append(data_point, data_point[best_feature+1:])

                        new_names = f_names[:best_feature]
                        new_names.append(f_names[best_feature+1:])

                    new_data.append(data_point)
                    new_class.append(classes[index])

                index += 1

            sub_tree = make_tree(new_data, new_class, new_names)

            #tree = Node(sub_tree.name, sub_tree.child_node)
            tree[f_names[best_feature]][value] = sub_tree

        return tree


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
    #classifier.train(d_data, d_target, feature_names)
    tree = classifier.train(train_data, train_target, feature_names)
    #get_accuracy(classifier.predict(tree, test_data, feature_names), test_target)
    classifier.predict(tree, test_data, feature_names)


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