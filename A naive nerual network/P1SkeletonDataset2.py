import csv
import numpy as np
import random
import matplotlib.pyplot as plt


class NeuralNetwork:

    def __init__(self, X, Y):

        # Activate function has been define in this class
        self.X = X
        self.Y = Y
        self.num_of_input = self.X.shape[1]
        self.num_of_output = self.Y.shape[1]

        self.NNodes = 100  # the number of nodes in the hidden layer
        self.epochs = 200
        self.learningRate = 0.01  # Learning rate
        self.regLambda = 0.001  # a parameter which controls the importance of the regularization term

        print("NNodes: ", self.NNodes)
        print("Epochs: ", self.epochs)
        print("Learning rate: ", self.learningRate)
        print("Regularization lambda: ", self.regLambda)
        print("Trainning...")

        self.w1 = []
        self.w2 = []
        self.splitList = []

    def fit(self):
        """
        This function is used to train the model.
        Parameters
        ----------
        X : numpy matrix
            The matrix containing sample features for training.
        Y : numpy array
            The array containing sample labels for training.
        Returns
        -------
        None
        """
        # Initialize your weight matrices first.
        # (hint: check the sizes of your weight matrices first!)

        # numpy.random.uniform(low, high, size)
        # numpy.random.randn(x, y) normal distribution mean 0, variance 1
        randn_amplifier = 3
        x = self.NNodes
        y = self.num_of_input + 1
        # self.w1 = np.reshape(np.random.uniform(-2, 2, x*y), (x, y))
        self.w1 = np.random.randn(x, y) * randn_amplifier
        x = self.num_of_output
        y = self.NNodes + 1
        # self.w2 = np.reshape(np.random.uniform(-2, 2, x*y), (x, y))
        self.w2 = np.random.randn(x, y) * randn_amplifier

        # print("w1 initialize")
        # print(self.w1)
        # print("w2 initialize")
        # print(self.w2)

        # For each epoch, do
        for i in range(self.epochs):
            # For each training sample (X[i], Y[i]), do
            for j in range(self.X.shape[0]):
                # 1. Forward propagate once. Use the function "forward" here!
                self.forward(self.X[j])
                # 2. Backward progate once. Use the function "backpropagate" here!
                self.backpropagate(self.X[j], self.Y[j])

        pass

    def predict(self, X):
        """
        Predicts the labels for each sample in X.
        Parameters
        X : numpy matrix
            The matrix containing sample features for testing.
        Returns
        -------
        YPredict : numpy array
            The predictions of X.
        ----------
        """
        YPredict = self.forward(X)
        return YPredict

    def forward(self, X):
        # Perform matrix multiplication and activation twice (one for each layer).
        # (hint: add a bias term before multiplication)
        # Add bias to X
        bias = np.ones((1, 1))
        X = np.concatenate((X, bias), axis=1)
        X = np.transpose(X)

        self.net1 = np.dot(self.w1, X)
        self.z1 = self.activate(self.net1)

        # Add bias to z1
        self.z1 = np.concatenate((self.z1, bias))
        # z1 is 5 by 1
        self.net2 = np.dot(self.w2, self.z1)
        self.z2 = self.activate(self.net2)

        return self.z2

    def backpropagate(self, X, YTrue):
        # Compute loss / cost using the getCost function.
        # cost = self.getCost(YTrue, self.z2)

        # Compute gradient for each layer.
        YTrue = np.transpose(YTrue)
        diff = self.z2 - YTrue
        dloss_dw2 = np.dot(diff, np.transpose(self.z1))

        diff_trans = np.transpose(diff)
        d_activate = self.deltaActivate(self.z1)[:-1]
        bias = np.ones((1, 1))
        X = np.concatenate((X, bias), axis=1)
        dloss_dw1 = np.dot(np.multiply(np.transpose(np.dot(diff_trans, self.w2))[:-1], d_activate), X)

        # Update weight matrices.
        self.w2 = self.w2 - self.learningRate * dloss_dw2 - self.learningRate * self.regLambda * self.w2
        self.w1 = self.w1 - self.learningRate * dloss_dw1 - self.learningRate * self.regLambda * self.w1

        pass

    def getCost(self, YTrue, YPredict):
        # Compute loss / cost in terms of cross entropy.
        # (hint: your regularization term should appear here)
        cross_entropy_cost = -((1 - YTrue) * np.log2(1 - YPredict) + YTrue * np.log2(YPredict)) \
                             - (
                                     np.sum(np.multiply(self.regLambda * self.w1, self.w1))
                                     + np.sum(np.multiply(self.regLambda * self.w2, self.w2))
                             )
        return cross_entropy_cost

    # Custom function
    def activate(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def deltaActivate(self, x):
        return np.multiply(self.activate(x), (1.0 - self.activate(x)))


##################################################################################


def getData(dataDir):
    '''
    Returns
    -------
    X : numpy matrix
        Input data samples.
    Y : numpy array
        Input data labels.
    '''

    # TO-DO for this part:
    # Use your preferred method to read the csv files.
    # Write your codes here:
    csv_file = open(dataDir, 'r')
    reader = csv.reader(csv_file)

    data = []
    for row in reader:
        data.append(row)

    data = [[float(x) for x in row] for row in data]  # transfer string to float
    data = np.asmatrix(data)
    csv_file.close()
    # print(data)
    # print(data.shape)
    return data


def splitData(X, Y, K):
    '''
    Returns
    -------
    result : List[[train, test]]
        "train" is a list of indices corresponding to the training samples in the data.
        "test" is a list of indices corresponding to the testing samples in the data.
        For example, if the first list in the result is [[0, 1, 2, 3], [4]], then the 4th
        sample in the data is used for testing while the 0th, 1st, 2nd, and 3rd samples
        are for training.
    '''

    # Make sure you shuffle each train list.
    for i in range(X.shape[0] - 1):
        index1 = random.randint(0, X.shape[0] - 1)
        index2 = random.randint(0, X.shape[0] - 1)
        # Switch X
        X[[index1, index2], :] = X[[index2, index1], :]
        # Switch Y
        Y[[index1, index2], :] = Y[[index2, index1], :]

    # return List[
    #    [ [x1, y1], [x2, y2], [x3, y3], [x4, y4] , .. [datak]]
    # ]
    list = []
    N = int(X.shape[0] / K)
    # print("N", N)
    for k in range(K):
        # print(k)
        index1 = k * N
        index2 = (k + 1) * N - 1 + 1

        Xk = X[index1:index2, :]
        Yk = Y[index1:index2, :]
        list.append([Xk, Yk])

    # print(list)

    return list


def test(Xtest, Ytest, model):
    # def test(self, XTest):
    """
    This function is used for the testing phase.
    Parameters
    ----------
    XTest : numpy matrix
        The matrix containing samples features (not indices) for testing.
    model : NeuralNetwork object
        This should be a trained NN model.
    Returns
    -------
    YPredict : numpy array
        The predictions of X.
    """

    correct = 0
    YPredict = []
    for i in range(Xtest.shape[0]):
        # print(Ytest[i])
        # print(model.predict(Xtest[i]))
        Y = model.predict(Xtest[i])
        # print(Y)
        max_index = np.where(Y == np.amax(Y))
        max_index = int(max_index[0])
        # print(max_index)
        res = np.zeros((Y.shape[0], 1))
        res[max_index, 0] = 1
        YPredict.append(res)
        #print(res)
        if Ytest[i, 0] == max_index:
            correct = correct + 1

    print(Ytest.shape)
    print(correct)

    return YPredict


def plotDecisionBoundary(model, X, Y):
    """
    Plot the decision boundary given by model.
    Parameters
    ----------
    model : model, whose parameters are used to plot the decision boundary.
    X : input data
    Y : input labels
    """
    # x1_array, x2_array = np.meshgrid(np.arange(-4, 4, 0.01), np.arange(-4, 4, 0.01))
    # grid_coordinates = np.c_[x1_array.ravel(), x2_array.ravel()]
    # Z = model.predict(grid_coordinates)
    # Z = Z.reshape(x1_array.shape)
    # plt.contourf(x1_array, x2_array, Z, cmap=plt.cm.bwr)
    # plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.bwr)
    # plt.show()

    x = np.transpose(X[:, 0: 1])
    y = np.transpose(X[:, 1: 2])

    x = np.asarray(x)
    y = np.asarray(y)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Scatter Plot')

    plt.xlabel('X1')

    plt.ylabel('X2')

    for i in range(len(Y)):
        if Y[i] == 0:
            ax1.scatter(x[0][i], y[0][i], c='r', marker='o')
            pass
        if Y[i] == 1:
            ax1.scatter(x[0][i], y[0][i], c='b', marker='o')
            pass

    plt.show()


def getConfusionMatrix(YTrue, YPredict):
    """
    Computes the confusion matrix.
    Parameters
    ----------
    YTrue : numpy array
        This array contains the ground truth.
    YPredict : numpy array
        This array contains the predictions.
    Returns
    CM : numpy matrix
        The confusion matrix.
    """
    N = int(YPredict[0].shape[0])

    confuse = np.zeros((N, N))
    for i in range(YTrue.shape[0]):
        x = int(YTrue[i, 0])
        Yres = YPredict[i]
        y = int(np.where(Yres == np.max(Yres))[0])
        confuse[x, y] = confuse[x, y] + 1

    print(confuse)

    return confuse


def getPerformanceScores(confusion):
    """
    Computes the accuracy, precision, recall, f1 score.
    Parameters
    ----------
    confusion matrix = [TP, TN, FP, FN]
    Returns
    {"CM" : numpy matrix,
    "accuracy" : float,
    "precision" : float,
    "recall" : float,
    "f1" : float}
        This should be a dictionary.
    """
    total = np.sum(confusion)
    accuracy = 0
    for i in range(confusion.shape[0]):
        accuracy = accuracy + confusion[i, i]
    accuracy = accuracy / total
    print("Accuracy: ", accuracy)

    precision = 0
    precisionList = []
    sumColumn = np.sum(confusion, axis=0)
    # print(sumColumn)
    # Precision means TP/(TP+FP)
    for i in range(confusion.shape[0]):
        pre = confusion[i, i] / int(sumColumn[i])
        precisionList.append(pre)
    # print(precisionList)
    precision = sum(precisionList)
    precision = precision / confusion.shape[0]
    print("Precision: ", precision)

    recall = 0
    sumRow = np.sum(confusion, axis=1)
    # print(sumRow)
    recallList = []
    # Recall means TP/(TP+FN)
    for i in range(confusion.shape[0]):
        rec = confusion[i, i] / int(sumRow[i])
        recallList.append(rec)
    # print(recallList)
    recall = sum(recallList)
    recall = recall / confusion.shape[0]
    print("Recall: ", recall)

    F1 = 2 * precision * recall / (precision+recall)
    print("F1:", F1)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "F1": F1}


def reshapeY(Y, num_of_classes):
    """
    Transform Y[i] into an boolean array
    The array index indicates the classification of the data sample.
    :param Y:
    :param num_of_classes:
    :return: Y
    """

    Y = np.concatenate((Y, np.zeros((Y.shape[0], num_of_classes-1))), axis=1)
    # print(Y.shape)
    for i in range(Y.shape[0]):
        types = int(Y[i:i+1, 0:1])
        # print(types)
        # print(type(types))
        Y[i:i+1, 0:1] = 0
        Y[i:i+1, types:types+1] = 1

    return Y




def main():
    # Get data
    # x_path = './dataset1/LinearX.csv'
    # y_path = './dataset1/LinearY.csv'
    # x_path = './dataset1/NonlinearX.csv'
    # y_path = './dataset1/NonlinearY.csv'
    x_path = './dataset2/Digit_X_train.csv'
    y_path = './dataset2/Digit_y_train.csv'

    X = getData(x_path)
    Y = getData(y_path)

    # print(X.shape[1])
    """
        We have to reshape Y.
    """
    num_of_classes = 10
    Y = reshapeY(Y, num_of_classes)
    np.set_printoptions(threshold=np.inf)
    # print(Y)

    # Train without splitting data
    my_neural_network = NeuralNetwork(X, Y)
    my_neural_network.fit()
    Xtest_path = './dataset2/Digit_X_test.csv'
    Ytest_path = './dataset2/Digit_y_test.csv'
    Xtest = getData(Xtest_path)
    Ytest = getData(Ytest_path)

    YPredict = test(Xtest, Ytest, my_neural_network)
    # print(YPredict)
    confusion = getConfusionMatrix(Ytest, YPredict)
    performance = getPerformanceScores(confusion)
    # plotDecisionBoundary(my_neural_network, Xtest, YPredict)
    print()


main()
