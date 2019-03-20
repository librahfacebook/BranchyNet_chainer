'''
构建ResNet网络模型：由109个卷积层和1个全连接层组成
'''
from branchynet.links.links import *
from branchynet.net import BranchyNet
from networks.resblock import ResBlock
import chainer.functions as F
import chainer.links as L
import math

conv = lambda n: [L.Convolution2D(n, 32, 3, stride=1, pad=1), FL(F.relu)]
cap = lambda n: [L.Linear(n, 10)]


class ResNet:

    def build(self, percentTrainKeeps=1):
        # 第一个分支网络（由3个卷积层和1个全连接层组成）
        branch1 = [L.Convolution2D(16, 64, 5, stride=1, pad=2)] + self.norm() + conv(64) + conv(32) + cap(30 * 30 * 32)
        # 第二个分支网络（由2个卷积层和1个全连接层组成）
        branch2 = [ResBlock(16, 16), L.Linear(30 * 30 * 16, 10)]
        network = self.gen_2b(branch1, branch2)
        net = BranchyNet(network, percentTrianKeeps=percentTrainKeeps)

        return net

    # 构建主体网络(并添加两个分支网络)
    def gen_2b(self, branch1, branch2):

        w = math.sqrt(2)
        n = 18
        network = [
            L.Convolution2D(3, 16, 3, stride=1, pad=0, initialW=w),
            L.BatchNormalization(16),
            FL(F.relu),
            Branch(branch1),
        ]
        for i in range(n):
            network += [ResBlock(16, 16)]
        network += [Branch(branch2)]

        for i in range(n):
            network += [ResBlock(32 if i > 0 else 16, 32, stride=1 if i > 0 else 2)]

        for i in range(n):
            network += [ResBlock(64 if i > 0 else 32, 64, stride=1 if i > 0 else 2)]

        network += [FL(F.average_pooling_2d, 6, 1)]
        network += [Branch([L.Linear(3 * 3 * 64, 10)])]

        return network

    # ReLu->归一化操作合并
    def norm(self):
        Operation = [FL(F.relu),
                     FL(F.local_response_normalization, n=3, alpha=5e-05, beta=0.75)]
        return Operation
