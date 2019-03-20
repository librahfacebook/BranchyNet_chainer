'''
构建AlexNet网络模型：由5个卷积层和3个全连接层组成
'''
from branchynet.links.links import *
from branchynet.net import BranchyNet

import chainer.functions as F
import chainer.links as L

conv = lambda n: [L.Convolution2D(n, 32, 3, pad=1, stride=1), FL(F.relu)]
cap = lambda n: [FL(F.max_pooling_2d, 3, 2), L.Linear(n, 10)]


class AlexNet:

    def build(self, percentTrainKeeps=1):
        #第一个分支网络（由2个卷积层和1个全连接层组成）
        branch1 = self.norm()+conv(64)+conv(32)+cap(512)
        #第二个分支网络（由1个卷积层和1个全连接层组成）
        branch2=self.norm()+conv(96)+cap(128)
        network=self.gen_2b(branch1,branch2)
        net=BranchyNet(network,percentTrianKeeps=percentTrainKeeps)

        return net

    # 构建主体网络(并添加两个分支网络)
    def gen_2b(self, branch1, branch2):
        network = [
            L.Convolution2D(3, 32, 5, pad=2, stride=1),
            FL(F.relu),
            FL(F.max_pooling_2d, 3, 2),
            FL(F.local_response_normalization, n=3, alpha=5e-05, beta=0.75),
            L.Convolution2D(32, 64, 5, pad=2, stride=1),
            Branch(branch1),
            FL(F.relu),
            FL(F.max_pooling_2d, 3, 2),
            FL(F.local_response_normalization, n=3, alpha=5e-05, beta=0.75),
            L.Convolution2D(64, 96, 3, pad=1, stride=1),
            Branch(branch2),
            FL(F.relu),
            L.Convolution2D(96, 96, 3, pad=1, stride=1),
            FL(F.relu),
            L.Convolution2D(96, 64, 3, pad=1, stride=1),
            FL(F.relu),
            FL(F.max_pooling_2d, 3, 2),
            L.Linear(1024, 256),
            FL(F.relu),
            SL(FL(F.dropout, 0.5)),
            L.Linear(256, 128),
            FL(F.relu),
            SL(FL(F.dropout, 0.5)),
            Branch([L.Linear(128, 10)])
        ]
        return network

    # ReLu->池化->归一化操作合并
    def norm(self):
        Operation = [FL(F.relu), FL(F.max_pooling_2d, 3, 2), FL(
            F.local_response_normalization, n=3, alpha=5e-05, beta=0.75
        )]
        return Operation
