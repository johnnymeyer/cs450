from sklearn import datasets
from sklearn.cross_validation import train_test_split as tts
from sklearn.neighbors import KNeighborsClassifier as knc
import pandas as pd
import numpy as np

# load the data from the database
iris = datasets.load_iris()
#car = pd.read_csv('car.csv')

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
train_data, test_data, train_target, test_target = tts(iris.data, iris.target, test_size=ts, random_state=rs)

class KNNClassifier:
    # def __init__(self, k):
    #     self.k = k
    #

    def train(self, data_set, target_set):
        # this will be where I standarize the data
        pass

    def predict(self, k, data, data_class, inputs):
        nInputs = np.shape(inputs)[0]
        closest = np.zeros(nInputs)

        for n in range(nInputs):
            distances = np.sum((data - inputs[n,:])**2, axis=1)

            indices = np.argsort(distances, axis=0)

            classes = np.unique(data_class[indices[:k]])

            if len(classes) == 1:
                closest[n] = np.unique(classes)
            else:
                counts = np.zeros(max(classes) + 1)
                for i in range(k):
                    counts[data_class[indices[i]]] += 1
                closest[n] = np.max(counts)

        return closest

def get_accuracy(results, test_tar):
    number_correct = 0

    for i in range(test_tar.size):
        if results[i] == test_tar[i]:
            number_correct = number_correct + 1

    print("\nNumber of correct predictions:", number_correct, " of ", test_tar.size)
    print("Accuracy rate is {0:.2f}%".format((number_correct / test_tar.size) * 100))

y = KNNClassifier()
#y.train(train_data, train_target)
get_accuracy(y.predict(k_value, train_data, train_target, test_data), test_target)