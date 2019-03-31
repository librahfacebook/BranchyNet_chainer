"""
构建ResNet网络模型：由109个卷积层和1个全连接层组成
"""
from branchynet.net import BranchyNet
from branchynet.links import *
from networks.resblock import ResBlock
import chainer.functions as F
import chainer.links as L
import math

conv = lambda n: [L.Convolution2D(n, 32, 3, stride=1, pad=1), FL(F.relu)]
cap = lambda n: [L.Linear(n, 10)]


class ResNet:

    def build(self, percentTrainKeeps=1):
        # 第一个分支网络（由3个卷积层和1个全连接层组成）
        branch1 = [L.Convolution2D(16, 64, 5, stride=1, pad=2)] + self.norm() + conv(64) + conv(32) + cap(32 * 30 * 30)
        # 第二个分支网络（由2个卷积层和1个全连接层组成）
        branch2 = [ResBlock(16, 16), L.Linear(16 * 30 * 30, 10)]
        network = self.gen_2b(branch1, branch2)
        net = BranchyNet(network, percentTrianKeeps=percentTrainKeeps)

        return net

    # 构建主体网络(并添加两个分支网络)
    def gen_2b(self, branch1, branch2):

        w = math.sqrt(2)
        n = 18
        network = [
            # C1(卷积层1): Input(3*32*32),卷积核为3*3，种类为16，stride=1，pad=0,Output(16*30*30)
            L.Convolution2D(3, 16, 3, stride=1, pad=0, initialW=w),
            L.BatchNormalization(16),  # 批处理归一化
            FL(F.relu),
            # 分支网络1
            Branch(branch1),
        ]
        # 卷积层数(36): Input(16*30*30)，，卷积核为3*3，种类为16，Output(16*30*30)
        for i in range(n):
            network += [ResBlock(16, 16)]
        # 分支网络2
        network += [Branch(branch2)]

        # 卷积层数(36): Input(16*30*30)，卷积核为3*3，种类为32，Output(32*15*15)
        for i in range(n):
            network += [ResBlock(32 if i > 0 else 16, 32, stride=1 if i > 0 else 2)]

        # 卷积层数(36): Input(32*15*15)，卷积核为3*3，种类为64，Output(64*8*8)
        for i in range(n):
            network += [ResBlock(64 if i > 0 else 32, 64, stride=1 if i > 0 else 2)]

        # 平均池化层：Input(64*8*8)，池化层采样为6*6，Output(64*3*3)
        network += [FL(F.average_pooling_2d, 6, 1)]

        # 输出层：Input(576)，Output(10)
        network += [Branch([L.Linear(64 * 3 * 3, 10)])]

        return network

    # ReLu->归一化操作合并
    def norm(self):
        Operation = [FL(F.relu),
                     FL(F.local_response_normalization, n=3, alpha=5e-05, beta=0.75)]
        return Operation
