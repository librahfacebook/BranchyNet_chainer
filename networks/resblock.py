'''
Residual block构建：通过shortcut connection实现，通过shortcut将这个block的输入
和输出进行一个element-wise的加叠，可以大大增加模型的训练速度、提高训练效果，并且
当模型的层数加深时，能够很好的解决退化问题。
'''
from chainer.backend import cuda
import chainer.functions as F
import chainer.links as L
import math, chainer, copy


class ResBlock(chainer.Chain):

    def __init__(self, in_channels, out_channels, ksize=1, stride=1):
        w = math.sqrt(2)
        super(ResBlock, self).__init__(
            conv1=L.Convolution2D(in_channels, out_channels, 3, stride=stride, pad=1, initialW=w),
            bn1=L.BatchNormalization(out_channels),
            conv2=L.Convolution2D(out_channels, out_channels, 3, stride=1, pad=1, initialW=w),
            bn2=L.BatchNormalization(out_channels),
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ksize = ksize
        self.stride = stride

    def __deepcopy__(self, memo):
        new = type(self)(self.in_channels, self.out_channels, self.ksize, self.stride)
        return new

    def __call__(self, x):

        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        #如果residual block的输入输出维度不一致，则对增加的维度用0来填充
        if x.data.shape != h.data.shape:
            xp = cuda.get_array_module(x.data)
            n, c, hh, ww = x.data.shape #获取输入数据的数量n、通道数c、高度hh、宽度ww
            pad_c = h.data.shape[1] - c #获取需要增加的通道数
            p = xp.zeros((n, pad_c, hh, ww), dtype=xp.float32) #对增加的维度用0填充
            p = chainer.Variable(p)
            with chainer.no_backprop_mode():
                x = F.concat((p, x))
                if x.data.shape[2:] != h.data.shape[2:]:
                    x = F.average_pooling_2d(x, 1, 2)
        return F.relu(h + x)
