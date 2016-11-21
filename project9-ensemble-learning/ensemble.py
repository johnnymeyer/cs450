from sklearn import ensemble
from sklearn import neural_network as MLP
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn import tree
from sklearn import svm
from sklearn.cross_validation import train_test_split as tts
from sklearn import datasets
import sys


def load_dataset(s):
    return s.data, s.target


def get_accuracy(res, tt):
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

    ans = None

    # Neural Network
    if classifier == 'n' or classifier == 'N':
        c = MLP.MLPClassifier()
        c.fit(train_data, train_target)
        ans = c.predict(test_data)
    # K Nearest Neighbors
    elif classifier == 'k' or classifier == 'K':
        knn = KNN(5)
        knn.fit(train_data, train_target)
        ans = knn.predict(test_data)
    # SVM
    elif classifier == 's' or classifier == 'S':
        s = svm.SVC()
        s.fit(train_data, train_target)
        ans = s.predict(test_target)
    # ensemble
    elif classifier == 'e' or classifier == 'E':
        print("not here yet")
    else:
        print("Invalid command\n")

    # get the accuracy
    get_accuracy(ans, test_target)


def main(argv):
    # load the data from the database - choose which data set you want to use

    # iris data
    data, targets = load_dataset(datasets.load_iris())

    classifier = input("Which classifier would you like to run?\nN: neural network\n" +
                       "K: K nearest neighbors\nT: decision tree\nS: support vector machine\n" +
                       "E: ensemble learning\n >> ")

    data_processing(data, targets, classifier)


if __name__ == "__main__":
    main(sys.argv)