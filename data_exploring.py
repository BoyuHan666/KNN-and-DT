import numpy as np
import matplotlib.pyplot as plt

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

def summary_binary_data1(x,y):
    dic = {"1-1":0,"1-2":0,"2-1":0,"2-2":0} # in data1 1:False 2:True
    for k in range(len(x)):
        i = x[k]
        j = y[k]
        if i == 1 and j == 2:
            dic["1-2"] += 1
        if i == 1 and j == 1:
            dic["1-1"] += 1
        if i == 2 and j == 2:
            dic["2-2"] += 1
        if i == 2 and j == 1:
            dic["2-1"] += 1
    return dic

def summary_binary_data2(x,y):
    dic = {"0-0":0,"0-1":0,"1-0":0,"1-1":0} # in data2 0:False 1:True
    for k in range(len(x)):
        i = x[k]
        j = y[k]
        if i == 0 and j == 0:
            dic["0-0"] += 1
        if i == 0 and j == 1:
            dic["0-1"] += 1
        if i == 1 and j == 0:
            dic["1-0"] += 1
        if i == 1 and j == 1:
            dic["1-1"] += 1
    return dic


if __name__ == "__main__":
    data1_path = "../data/data1.csv"
    data2_path = "../data/data2.csv"
    data1 = trim_data1(data1_path)
    data2 = trim_data2(data2_path)

    # explore data1
    x1 = data1[:,1]
    y1 = data1[:,0]
    print(summary_binary_data1(data1[:,2],y1))
    plt.scatter(x1,y1)
    # plt.show()

    print(sum(x1)/len(x1))
    _, _, plot1 = plt.hist(x1, bins=10, align="mid")
    for p in plot1:
        x = (p._x0 + p._x1) / 2
        y = p._y1
        plt.text(x, y, p._y1)
    # plt.show()

    # explore data2
    x2 = data2[:,1]
    y2 = data2[:,19]
    print(summary_binary_data2(data2[:, 1], y2))
    plt.scatter(x2, y2)
    # plt.show()

    print(sum(x2) / len(x2))
    _, _, plot2 = plt.hist(x2, bins=10, align="mid")
    for p in plot2:
        x = (p._x0 + p._x1) / 2
        y = p._y1
        plt.text(x, y, p._y1)
    # plt.show()
