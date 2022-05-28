import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


class Knn:
    def __init__(self):
        self.k = None
        self.xs = None
        self.y = None
        return

    def fit(self, xs, y, k):
        self.k = k
        self.xs = xs
        self.y = y
        return self

    def scaling(self, xs):

        return xs

    def predict_Euclidean(self, test_data):
        y_bar_list = []
        for row in test_data:
            dic = {}
            for i in range(len(self.xs)):
                distance = 0
                for j in range(len(row)):
                    distance += (row[j] - self.xs[i][j]) ** 2
                distance = math.sqrt(distance)
                dic[distance] = i
            sort_dic = sorted(dic)[0:self.k]
            vote_list = []
            for key in sort_dic:
                vote_list.append(self.y[dic[key]])
            y_bar = max(set(vote_list), key=vote_list.count)
            y_bar_list.append(y_bar)
        return y_bar_list

    def predict_Manhattan(self, test_data):
        y_bar_list = []
        for row in test_data:
            dic = {}
            for i in range(len(self.xs)):
                distance = 0
                for j in range(len(row)):
                    distance += abs(row[j] - self.xs[i][j])
                dic[distance] = i
            sort_dic = sorted(dic)[0:self.k]
            vote_list = []
            for key in sort_dic:
                vote_list.append(self.y[dic[key]])
            y_bar = max(set(vote_list), key=vote_list.count)
            y_bar_list.append(y_bar)
        return y_bar_list


def trim_data1(path):
    data = np.genfromtxt(path, delimiter=',')
    clean_data1 = []
    for row in data:
        flag = True
        for e in row:
            # delete the rows that contain ?
            if np.isnan(e):
                flag = False
        if flag:
            clean_data1.append(row)
    np_data1 = np.array(clean_data1)
    return np_data1


def trim_data2(path):
    data = np.genfromtxt(path, delimiter=',')
    clean_data2 = []
    for row in data:
        # delete the rows that contain 0 from column 12 to 16
        if 0 not in row[11:16]:
            clean_data2.append(row)
    np_data2 = np.array(clean_data2)
    return np_data2


def evaluate_acc(l1, l2):
    count = 0
    n = len(l1)
    for i in range(n):
        if l1[i] == l2[i]:
            count += 1
    return count / n


def generate_random_sample(data, x_index, y_index):
    factor = data.shape[0] // 5 * 4
    train_data = data[np.random.choice(data.shape[0], factor, replace=False), :]
    train_data = train_data.tolist()
    data = data.tolist()
    test_data = []
    for i in data:
        if i not in train_data:
            test_data.append(i)
    xs = []
    y = []
    test_xs = []
    test_y = []
    for row in train_data:
        xs.append(list(row[x_index[0]:x_index[1]]))
        y.append(row[y_index])
    for row in test_data:
        test_xs.append(list(row[x_index[0]:x_index[1]]))
        test_y.append(row[y_index])

    if test_xs == []:
        data = np.array(data)
        xs = data[:factor,x_index[0]:x_index[1]].tolist()
        y = data[:factor,y_index].tolist()
        test_xs = data[factor:,x_index[0]:x_index[1]].tolist()
        test_y = data[factor:, y_index].tolist()
        data = data.tolist()
    return [xs, y, test_xs, test_y]


def get_accuracy_Manhattan(data, x_index, y_index, k, num):
    accuracy = 0
    for i in range(num):
        xs, y, test_data, y_true = generate_random_sample(data, x_index, y_index)
        new_knn = Knn()
        new_knn.fit(k=k, xs=xs, y=y)
        y_bar = new_knn.predict_Manhattan(test_data)
        accuracy += evaluate_acc(y_bar, y_true)
    return accuracy / num * 100


def get_accuracy_Euclidean(data, x_index, y_index, k, num):
    accuracy = 0
    for i in range(num):
        xs, y, test_data, y_true = generate_random_sample(data, x_index, y_index)
        # print(len(xs))
        # print(len(test_data))
        new_knn = Knn()
        new_knn.fit(k=k, xs=xs, y=y)
        y_bar = new_knn.predict_Euclidean(test_data)
        accuracy += evaluate_acc(y_bar, y_true)
    return accuracy / num * 100


def test_data1(data1, k):
    num = 100
    a1_Euclidean = get_accuracy_Euclidean(data1, x_index=[1, data1.shape[1]], y_index=0, k=k, num=num)
    print("Euclidean: " + str(round(a1_Euclidean, 1)) + "%")
    a1_Manhattan = get_accuracy_Manhattan(data1, x_index=[1, data1.shape[1]], y_index=0, k=k, num=num)
    print("Manhattan: " + str(round(a1_Manhattan, 1)) + "%")


def test_data2(data2, k):
    num = 10
    a2_Euclidean = get_accuracy_Euclidean(data2, x_index=[0, data2.shape[1]], y_index=data2.shape[1] - 1, k=k, num=num)
    print("Euclidean: " + str(round(a2_Euclidean, 1)) + "%")
    a2_Manhattan = get_accuracy_Manhattan(data2, x_index=[0, data2.shape[1]], y_index=data2.shape[1] - 1, k=k, num=num)
    print("Manhattan: " + str(round(a2_Manhattan, 1)) + "%")


def draw_decision_boundary(data, col, colx1, colx2, target, splitter, k_range, predict):
    x = data[:, col[0]:col[1]]
    x0v = np.linspace(np.min(x[:, colx1]), np.max(x[:, colx1]), splitter)
    x1v = np.linspace(np.min(x[:, colx2]), np.max(x[:, colx2]), splitter)
    x0, x1 = np.meshgrid(x0v, x1v)
    x_all = np.vstack((x0.ravel(), x1.ravel())).T
    x_train, y_train, test_data, y_true = generate_random_sample(data, [colx1, colx2 + 1], target)
    for i in range(len(y_train)):
        y_train[i] = int(y_train[i])
    y_train = np.array(y_train)
    x_train = np.array(x_train)
    for k in range(k_range[0], k_range[1]):
        model = Knn()
        model.fit(x_train, y_train, k=k)
        y_train_prob = np.zeros((y_train.shape[0], 3))
        y_train_prob[np.arange(y_train.shape[0]), y_train] = 1
        y_prob_all = []
        if predict == "Euclidean":
            y_prob_all = model.predict_Euclidean(x_all)
        if predict == "Manhattan":
            y_prob_all = model.predict_Manhattan(x_all)
        y_train_prob_all = np.zeros((len(y_prob_all), 3))
        for i in range(len(y_prob_all)):
            if i == 1:
                y_train_prob_all[i, y_prob_all[i]] = 1
            else:
                y_train_prob_all[i, y_prob_all[i]] = 1

        plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train_prob, marker='o', alpha=1)
        plt.scatter(x_all[:, 0], x_all[:, 1], c=y_train_prob_all, marker='.', alpha=0.1)
        plt.ylabel('column '+str(colx2+1))
        plt.xlabel('column '+str(colx1+1))
        plt.show()


def scaling(data, col, method, s_factor):
    if col == "all":
        # scaling all values to [0,1]
        for i in range(data.shape[1]):
            dominator = np.max(data[:, i])
            data[:, i] = data[:, i] / dominator
    else:
        # scaling the values of some columns to [0,1]
        if method == "normalization":
            for i in col:
                dominator = np.max(data[:, i])
                data[:, i] = data[:, i] / dominator
        # scaling_up
        if method == "up":
            for i in col:
                data[:, i] = data[:, i] * s_factor
        # scaling_down
        if method == "down":
            for i in col:
                data[:, i] = data[:, i] / s_factor
    return data


if __name__ == "__main__":
    np.random.seed(10)
    data1_path = "../data/data1.csv"
    data2_path = "../data/data2.csv"
    data1 = trim_data1(data1_path)

    # if we treat 0 as an outlier from column 11-15
    data2 = trim_data2(data2_path)
    # use all rows in data2
    # data2 = np.genfromtxt(data2_path, delimiter=',')

    # knn for data1
    # data1 = scaling(data1, col=[15,16], method="down", s_factor=200)
    # data1 = scaling(data1, col=[15,16], method="down", s_factor=1000)
    # data1 = np.delete(data1, 16, axis=1)
    # data1 = np.delete(data1, 15, axis=1)
    # universal_selection1 = np.array(list(zip(data1[:,0],data1[:,16])))
    # print(universal_selection1)
    # test_data1(universal_selection1, 7)
    test_data1(data1, 7)



    # knn for data2
    # universal_selection2 = np.array(list(zip(data2[:,9],data2[:,19])))
    # test_data2(universal_selection2, 7)
    # test_data2(data2, 7)
    # data2 = scaling(data2, col=[2,3,4,5,6,7], method="down", s_factor=20)
    # data2 = scaling(data2, col=[18], method="up", s_factor=10000)
    test_data2(data2, 7)
    # data2_test = np.array(data2.tolist()[:60])
    # data2_train = np.array(data2.tolist()[60:])
    #
    # test_data2(data2_train, k=5)
    # print("--------------------------")
    # test_data2(data2_test, k=5)
    # draw_decision_boundary(data=data2, col=[0, 19], colx1=9, colx2=10, target=19,
    #                        splitter=100, k_range=[7,8], predict="Euclidean")
