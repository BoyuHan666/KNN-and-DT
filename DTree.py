import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import math


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


class Node:
    def dbg(self, indent=0):
        print('\t' * indent, self.feature_name, '<=', self.compare_value, 'cost:', self.cost_classification(), 'label:',
              self.most_frequent_label(), 'len:', len(self.X))
        if self.left_node is not None: self.left_node.dbg(indent + 1)
        if self.right_node is not None: self.right_node.dbg(indent + 1)

    def __init__(self, X, y):
        self.X = X  # Features
        self.y = y  # Class (label)
        self.feature_name = None  # name of xi
        self.compare_value = None
        self.left_node = None  # yes
        self.right_node = None  # no
        self.cost_function = self.gini_cost

    def cost_classification(self):
        if len(self.y) == 0:
            return 0
        return (len(self.y) - self.y.value_counts().max()) / len(self.y)  # (total - correct) / total rows

    def entropy_cost(self):
        prob_correct = (self.y.value_counts().max()) / len(self.y)
        prob_wrong = 1 - prob_correct
        if prob_correct == 0 or prob_wrong == 0:
            return 0
        return - prob_correct * np.log2(prob_correct) - prob_wrong * np.log2(prob_wrong)

    def gini_cost(self):
        prob_correct = (self.y.value_counts().max()) / len(self.y)
        prob_wrong = 1 - prob_correct
        return 1 - prob_correct ** 2 - prob_wrong ** 2

    def most_frequent_label(self):
        return self.y.value_counts().idxmax()


def greedy_test(node):
    # print('greedy_test...')
    best_cost = np.inf
    best_left_node = None
    best_right_node = None
    best_feature_name = None
    best_compare_value = None
    for column_name in node.X:
        for unique_value in node.X[column_name].unique():
            # print(column_name, unique_value)
            condition = node.X[column_name] <= unique_value
            left_X = node.X[condition]  # yes features
            left_y = node.y[condition]  # yes label
            left_node = Node(left_X, left_y)  # yes node

            right_X = node.X[~condition]  # no features
            right_y = node.y[~condition]  # no label
            right_node = Node(right_X, right_y)  # no node

            if len(left_X) == 0 or len(
                    right_X) == 0:  # if one child will have 0 rows after splitting, don't split this way
                continue

            split_cost = left_node.entropy_cost() * len(left_node.X) / len(
                node.X) + right_node.entropy_cost() * len(right_node.X) / len(node.X)

            if split_cost < best_cost:
                best_cost = split_cost
                best_left_node = left_node
                best_right_node = right_node
                best_feature_name = column_name
                best_compare_value = unique_value

    return best_left_node, best_right_node, best_feature_name, best_compare_value


class DecisionTree:
    def __init__(self, min_samples=1):
        self.root = None
        self.max_depth = None
        self.min_samples = min_samples

    def fit(self, X, y, max_depth):
        self.root = Node(X, y)
        self.max_depth = max_depth
        self.fit_tree(self.root, 0)

    def fit_tree(self, node, depth):
        if node is None:
            return

        node.left_node, node.right_node, node.feature_name, node.compare_value = self.greedy_test(
            node)  # return value of greedy_test function

        if not self.worth_splitting(depth, node):
            node.left_node, node.right_node, node.feature_name, node.compare_value = None, None, None, None
        else:
            self.fit_tree(node.left_node, depth + 1)
            self.fit_tree(node.right_node, depth + 1)

    def greedy_test(self, node):
        # print('greedy_test...')
        best_cost = np.inf
        best_left_node = None
        best_right_node = None
        best_feature_name = None
        best_compare_value = None
        for column_name in node.X:
            for unique_value in node.X[column_name].unique():
                condition = node.X[column_name] <= unique_value
                left_X = node.X[condition]  # yes features
                left_y = node.y[condition]  # yes label
                left_node = Node(left_X, left_y)  # yes node

                right_X = node.X[~condition]  # no features
                right_y = node.y[~condition]  # no label
                right_node = Node(right_X, right_y)  # no node

                if len(left_X) == 0 or len(
                        right_X) == 0:  # if one child will have 0 rows after splitting, don't split this way
                    continue

                split_cost = left_node.cost_function() * len(left_node.X) / len(
                    node.X) + right_node.cost_function() * len(right_node.X) / len(node.X)

                if split_cost < best_cost:
                    best_cost = split_cost
                    best_left_node = left_node
                    best_right_node = right_node
                    best_feature_name = column_name
                    best_compare_value = unique_value
        return best_left_node, best_right_node, best_feature_name, best_compare_value

    def worth_splitting(self, depth, node):
        if depth > self.max_depth:  # node is at max_depth, perform last split. left_node & right_node will not be split
            return False
        if node.left_node is None or node.right_node is None or len(node.left_node.X) == 0 or len(
                node.right_node.X) == 0:  # cannot split, no more rows in one child
            return False
        if node.cost_function() == 0:  # node cost is 0, don't need to split left & right
            return False
        if len(node.X) < self.min_samples:
            return False
        return True

    def predict(self, X):
        return X.apply(self.predict_row, axis=1)

    def predict_row(self, row):
        node = self.root  # start from root
        while node.feature_name is not None:  # has left & right, go down tree
            if [row[node.feature_name]] <= node.compare_value:  # go left
                node = node.left_node
            else:  # go right
                node = node.right_node
        return node.most_frequent_label()  # predict as most frequent label


def evaluate_acc(y, y_hat):
    cnt = 0  # count correct labels
    for i in range(len(y)):
        cnt += 1 if y.iloc[i] == y_hat.iloc[i] else 0
    return cnt / len(y)


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
    return [xs, y, test_xs, test_y]


def draw_decision_boundary(data, col, colx1, colx2, target, splitter, max_depth):
    data = data.to_numpy()
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
    x_train = pd.DataFrame(x_train, columns=[colx1, colx2])
    x_all = pd.DataFrame(x_all, columns=[colx1, colx2])
    y_train = pd.DataFrame(y_train, columns=[target])

    model = DecisionTree()
    model.fit(x_train, y_train, max_depth=max_depth)

    y_train = y_train.to_numpy()
    y_train_prob = np.zeros((y_train.shape[0], 3))
    y_train = y_train.flatten()
    y_train_prob[np.arange(y_train.shape[0]), y_train] = 1

    y_prob_all = model.predict(x_all).to_numpy().tolist()
    l = []
    for i in y_prob_all:
        l.append(i[0])
    y_prob_all = l
    y_train_prob_all = np.zeros((len(y_prob_all), 3))

    for i in range(len(y_prob_all)):
        if i == 1:
            y_train_prob_all[i, y_prob_all[i]] = 1
        else:
            y_train_prob_all[i, y_prob_all[i]] = 1

    x_train = x_train.to_numpy()
    x_all = x_all.to_numpy()
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train_prob, marker='o', alpha=1)
    plt.scatter(x_all[:, 0], x_all[:, 1], c=y_train_prob_all, marker='.', alpha=0.1)
    plt.ylabel('column ' + str(colx2+1))
    plt.xlabel('column ' + str(colx1+1))
    plt.show()


if __name__ == '__main__':
    np.random.seed(10)

    # for data1
    df = pd.read_csv('../data/data1_pd.csv')
    # print(len(df))

    df_without_missing = df[~df.eq("?").any(1)]
    # print(len(df_without_missing))

    df_without_missing = df_without_missing.apply(pd.to_numeric)
    # print(len(df_without_missing))

    data_X = df_without_missing.drop("Class", axis=1).copy()

    data_y = df_without_missing["Class"].copy()

    training_mask = np.random.rand(len(data_X)) < 0.7  # < 0.8 -> training; >= 0.8 -> test
    training_X = data_X[training_mask]
    training_y = data_y[training_mask]
    test_X = data_X[~training_mask]
    test_y = data_y[~training_mask]

    accuracies = []
    D = 10
    start = time.time()

    for min_samples in [1, 2, 5, 10, 20, 40]:
        dt = DecisionTree(min_samples)
        dt.fit(training_X, training_y, max_depth=3)
        y_hat = dt.predict(test_X)
        # print(y_hat)

        accuracies.append(evaluate_acc(test_y, y_hat))
        print('min_samples:', min_samples, 'accuracy:', accuracies[-1])
        print(time.time() - start, 'seconds')
        start = time.time()
        dt.root.dbg()

    print(accuracies)

    # for data2
    data2 = trim_data2("../data/data2.csv")
    data2 = pd.DataFrame(data2, columns=['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9',
                                         'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', "Class"])

    data_X2 = data2.drop("Class", axis=1).copy()
    # data_X2 = data2['x0'].copy()
    # 2. make a new copy of the column of data we wanna predict
    data_y2 = data2["Class"].copy()
    training_mask2 = np.random.rand(len(data_X2)) < 0.7
    training_X2 = data_X2[training_mask2]
    training_y2 = data_y2[training_mask2]
    test_X2 = data_X2[~training_mask2]
    test_y2 = data_y2[~training_mask2]

    # for max_depth in range(3, 4):
    #     dt = DecisionTree()
    #     dt.fit(training_X2, training_y2, max_depth)
    #     y_hat = dt.predict(test_X2)
    #     print('max_depth:', max_depth, 'accuracy:', evaluate_acc(test_y2, y_hat))

    # test decision boundary of data2 based on the features 10 and 11
    # draw_decision_boundary(data=data2, col=[0, 19], colx1=9, colx2=10, target=19,
    #                        splitter=100, max_depth=3)
