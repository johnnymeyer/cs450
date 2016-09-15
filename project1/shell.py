from sklearn import datasets
from sklearn.cross_validation import train_test_split as tts

# load the data from the database
iris = datasets.load_iris()

# split the data into test and training sets after it shuffles the data
train_data, test_data, train_target, test_target = tts(iris.data, iris.target, test_size=.3, random_state=15)

class HardCoded:
    def train(self, data_set, target_set):
        print("Training my very young padawon!\n\n")

    def predict(self, data_set):
        x = []
        for i in data_set:
            x.append(0)

        return x

def get_accuracy(results, test_tar):
    number_correct = 0

    for i in range(test_tar.size):
        if results[i] == test_tar[i]:
             number_correct = number_correct + 1

    print("Number of correct predictions:", number_correct, " of ", test_tar.size)
    print("Accuracy rate is {0:.2f}%".format((number_correct / test_tar.size) * 100))

y = HardCoded()
y.train(train_data, train_target)
get_accuracy(y.predict(test_data), test_target)