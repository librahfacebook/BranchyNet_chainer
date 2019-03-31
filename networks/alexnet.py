"""
构建B-AlexNet网络模型：由5个卷积层和3个全连接层组成，
其中包括两个分支网络
"""
from branchynet.net import BranchyNet
from branchynet.links import *
import chainer.functions as F
import chainer.links as L

conv = lambda n: [L.Convolution2D(n, 32, 3, pad=1, stride=1), FL(F.relu)]  # 卷积函数操作
cap = lambda n: [FL(F.max_pooling_2d, 3, 2), L.Linear(n, 10)]  # 最大池化+全连接操作


class AlexNet:

    # 构建B-AlexNet网络，其中n代表第一个分支的卷积层数
    def build(self, n=2, percentTrainKeeps=1):
        # 第一个分支网络（由n个卷积层和1个全连接层组成）Input(64*16*16)
        if n == 0:
            branch1 = self.norm() + cap(64 * 4 * 4)
        else:
            branch1 = self.norm() + conv(64)
            for i in range(n - 1):
                branch1 += conv(32)
            branch1 += cap(512)

        # 第二个分支网络（由1个卷积层和1个全连接层组成） Input(96*8*8)
        branch2 = self.norm() + conv(96) + cap(128)
        network = self.gen_2b(branch1, branch2)
        net = BranchyNet(network, percentTrianKeeps=percentTrainKeeps)

        return net

    # 构建主体网络(并添加两个分支网络)
    def gen_2b(self, branch1, branch2):
        network = [
            # C1(卷积层1): Input(3*32*32),卷积核为5*5，种类为32，stride=1，pad=2,Output(32*32*32)
            L.Convolution2D(3, 32, 5, pad=2, stride=1),
            FL(F.relu),
            # S1(池化层1): Input(32*32*32),池化层采样为3*3，Output(32*16*16)
            FL(F.max_pooling_2d, 3, 2),
            FL(F.local_response_normalization, n=3, alpha=5e-05, beta=0.75),
            # C2(卷积层2): Input(32*16*16),卷积核为5*5，种类为64，stride=1，pad=2,Output(64*16*16)
            L.Convolution2D(32, 64, 5, pad=2, stride=1),
            Branch(branch1),
            FL(F.relu),
            # S2(池化层2): Input(64*16*16),池化层采样为3*3，Output(64*8*8)
            FL(F.max_pooling_2d, 3, 2),
            FL(F.local_response_normalization, n=3, alpha=5e-05, beta=0.75),
            # C3(卷积层3): Input(64*8*8),卷积核为3*3，种类为96，stride=1，pad=1,Output(96*8*8)
            L.Convolution2D(64, 96, 3, pad=1, stride=1),
            Branch(branch2),
            FL(F.relu),
            # C4(卷积层4): Input(96*8*8),卷积核为3*3，种类为96，stride=1，pad=1,Output(96*8*8)
            L.Convolution2D(96, 96, 3, pad=1, stride=1),
            FL(F.relu),
            # C5(卷积层5): Input(96*8*8),卷积核为3*3，种类为64，stride=1，pad=1,Output(64*8*8)
            L.Convolution2D(96, 64, 3, pad=1, stride=1),
            FL(F.relu),
            # S3(池化层3): Input(64*8*8),池化层采样为3*3，Output(64*4*4)
            FL(F.max_pooling_2d, 3, 2),
            # FC1(全连接层1): Input(1024),Output(256)
            L.Linear(1024, 256),
            FL(F.relu),
            SL(FL(F.dropout, 0.5)),
            # FC2(全连接层2): Input(256),Output(128)
            L.Linear(256, 128),
            FL(F.relu),
            SL(FL(F.dropout, 0.5)),
            # FC3(全连接层3): Input(128),Output(10)
            Branch([L.Linear(128, 10)])
        ]
        return network

    # ReLu->池化->归一化操作合并
    def norm(self):
        Operation = [FL(F.relu), FL(F.max_pooling_2d, 3, 2), FL(
            F.local_response_normalization, n=3, alpha=5e-05, beta=0.75
        )]
        return Operation
