'''
神经网络搭建过程中的各种操作
'''
import chainer
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import inspect, copy
import cupy
from scipy.stats import entropy


# 定义网络的一般结构来直接进行训练或测试
class Net(ChainList):
    def __init__(self, weight=1.):
        super(Net, self).__init__()
        self.weight = weight
        self.starti = 0
        self.endi = 0

    # x为一批次放入的数据集，例如：训练过程时x为（512，1,28,28）
    # 此函数是对训练过程进行调用,self是指网络架构，其长度为网络的层数
    def __call__(self, x, test=False, starti=0, endi=None):
        h = x
        for link in self[starti:endi]:
            h = link(h)
            # if len(inspect.getfullargspec(link.__call__)[0]) == 2:
            #     h = link(h)
            # else:
            #     h = link(h,test)
        self.h = h
        return h

    # 训练过程
    def train(self, x, t, starti=0, endi=None):
        h = self(x, False, starti, endi)

        self.accuracy = F.accuracy(h, t)  # 精确度
        self.loss = F.softmax_cross_entropy(h, t)  # 损失函数
        return self.loss

    # 测试过程
    def test(self, x, starti=0, endi=None):
        h = self(x, True, starti, endi)
        return h


# 定义分支网络的结构
class Branch(ChainList):
    def __init__(self, branch, weight=1.):
        super(Branch, self).__init__()
        self.branch = branch
        self.weight = weight
        for link in branch:
            self.add_link(link)

    def cleargrads(self):
        super(SL, self).cleargrads()
        for link in self.branch:
            link.cleargrads()

    def __deepcopy__(self, memo):
        newbranches = []
        for link in self.branch:
            newbranches.append(copy.deepcopy(link, memo))
        new = type(self)(newbranches, self.weight)
        return new

    def __call__(self, x, test=False, starti=0, endi=None):
        h = x
        for link in self[starti:endi]:
            if len(inspect.getfullargspec(link.__call__)[0]) == 2:
                h = link(h)
            else:
                h = link(h, test)
        return h


# 网络层命令函数的操作
class FL(Link):
    def __init__(self, fn, *arguments, **keywords):
        super(FL, self).__init__()
        self.fn = fn
        self.arguments = arguments
        self.keywords = keywords

    def __call__(self, x, test=False):
        return self.fn(x, *self.arguments, **self.keywords)


#训练模式或测试模式的选择
class SL(Link):
    def __init__(self, fnTrain, fnTest=None):
        super(SL, self).__init__()
        self.fnTrain = fnTrain
        self.fnTest = fnTest

    def cleargrads(self):
        super(SL, self).cleargrads()
        if self.fnTrain is not None:
            self.fnTrain.cleargrads()
        if self.fnTest is not None:
            self.fnTest.cleargrads()

    # 对对象进行深度复制操作
    def __deepcopy__(self, memo):
        fnTrain = copy.deepcopy(self.fnTrain, memo)
        fnTest = copy.deepcopy(self.fnTest, memo)
        new = type(self)(fnTrain, fnTest)
        return new

    def to_gpu(self):
        if self.fnTrain is not None:
            self.fnTrain.to_gpu()
        if self.fnTest is not None:
            self.fnTest.to_gpu()

    def to_cpu(self):
        if self.fnTrain is not None:
            self.fnTrain.to_cpu()
        if self.fnTest is not None:
            self.fnTest.to_cpu()

    def __call__(self, x, test=False):
        if not test:
            return self.fnTrain(x, test)
        else:
            if self.fnTest is None:
                return x
            return self.fnTest(x, test)
        return x


class X(chainer.ChainList):
    def __init__(self, main):
        super(X, self).__init__()
        for link in main:
            self.add_link(link)
        self.weights = [1., 1.]

    def __deepcopy__(self, memo):
        newmain = []
        for link in self:
            newlink = copy.deepcopy(link)
            newlink.name = None
            newmain.append(newlink)
        new = type(self)(newmain)
        return new

    def set_weights(self, weights):
        self.weights = weights
        return

    def __call__(self, x, test):
        h = x
        for link in self:
            if len(inspect.getfullargspec(link.__call__)[0]) == 2:
                h = link(h)
            else:
                h = link(h, test)
        if x.data.shape != h.data.shape:
            xp = chainer.cuda.get_array_module(x.data)
            n, c, hh, ww = x.data.shape
            pad_c = h.data.shape[1] - c
            p = xp.zeros((n, pad_c, hh, ww), dtype=xp.float32)
            p = chainer.Variable(p, volatile=test)
            x = F.concat((p, x))
            if x.data.shape[2:] != h.data.shape[2:]:
                x = F.average_pooling_2d(x, 1, 2)
        return self.weights[0] * h + self.weights[1] * x
