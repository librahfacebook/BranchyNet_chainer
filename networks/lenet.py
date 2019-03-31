"""
构建B-LeNet网络模型：由三个卷积层和两个全连接层组成，
其中包括添加侧边分支的网络结构
"""
from branchynet.net import BranchyNet
from branchynet.links import *
import chainer.links as L
import chainer.functions as F


class LeNet:
    @staticmethod
    def build(percentTrainKeeps=1):
        network = [
            # C1(卷积层1): Input(1*28*28),卷积核为5*5，种类为5，stride=1，pad=3,Output(5*30*30)
            L.Convolution2D(1, 5, 5, stride=1, pad=3),

            # B1(分支网络: 由1个卷积层和1个全连接层组成)
            # B1_S1(池化层): Input(5*30*30),池化层采样为2*2，Output(5*15*15)
            # B1_C1(卷积层): Input(5*15*15),卷积核为3*3，种类为10，stride=1，pad=1,Output(10*15*15)
            # B1_S2(池化层): Input(10*13*13),池化层采样为2*2，Output(10*8*8)
            # B1_FC1(全连接层): Input(640),Output(10)
            Branch([FL(F.max_pooling_2d, 2, 2), FL(F.relu), L.Convolution2D(5, 10, 3, stride=1, pad=1),
                    FL(F.max_pooling_2d, 2, 2), FL(F.relu), L.Linear(640, 10)]),

            # S1(池化层1): Input(5*30*30),池化层采样为2*2，Output(5*15*15)
            FL(F.max_pooling_2d, 2, 2),
            FL(F.relu),

            # C2(卷积层2)：Input(5*15*15),卷积核为5*5，种类为10，stride=1，pad=3,Output(10*17*17)
            L.Convolution2D(5, 10, 5, stride=1, pad=3),

            # S2(池化层2): Input(10*17*17),池化层采样为2*2，Output(10*9*9)
            FL(F.max_pooling_2d, 2, 2),
            FL(F.relu),

            # C3(卷积层3)：Input(10*9*9),卷积核为5*5，种类为20，stride=1，pad=3,Output(20*11*11)
            L.Convolution2D(10, 20, 5, stride=1, pad=3),

            # S3(池化层3): Input(20*11*11),池化层采样为2*2，Output(20*6*6)
            FL(F.max_pooling_2d, 2, 2),
            FL(F.relu),

            # FC1(全连接层1): Input(720),Output(84)
            L.Linear(720, 84),

            # B2(分支网络2: B2_FC:Input(84),Output(10))
            Branch([L.Linear(84, 10)])
        ]
        # 构建B-LeNet
        net = BranchyNet(network, percentTrianKeeps=percentTrainKeeps)

        return net
