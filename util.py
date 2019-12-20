# coding=utf-8
import ast
import numpy as np
import sys


def read_file(path, split_rate):
    x, y = [], []
    with open(path, 'r') as f:
        for line in f:
            _, file_path, _, _, _, label = line.replace('\n', '').split(',')
            file = np.load(file_path)
            x.append(file)
            y.append(int(label))

    np.random.seed(1)
    x = np.array(x)
    y = np.array(y)
    random_index = [i for i in range(len(x))]
    random_index = np.random.permutation(random_index)

    train_random = random_index[:  int(len(x) * split_rate)]
    test_random = random_index[int(len(x) * split_rate):]

    x_train, y_train, x_test, y_test = x[train_random], y[train_random], x[
        test_random], y[test_random]
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = read_file('./data/patch/label.txt', 0.8)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
