'''
加载MNIST数据集：返回训练集以及测试集
X:(,1,28,28) Y:(,1)
'''
from chainer.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt


# 获取MNIST的训练集和测试集
def get_data():
    train, test = mnist.get_mnist()
    x_train, y_train = process_data(train)
    x_test, y_test = process_data(test)

    # print(x_train.shape)
    # print(y_train.shape)
    # plt.imshow(x_train[1234].reshape(28 , 28), cmap='gray')
    # plt.show()

    return x_train, y_train, x_test, y_test


# 对MNIST数据集的处理，返回array形式的数据
def process_data(data):
    x = []
    y = []
    for d in data:
        x.append(d[0])
        y.append(d[1])

    x = np.array(x)
    x = x.reshape([-1, 1, 28, 28])
    y = np.array(y)

    return x, y

