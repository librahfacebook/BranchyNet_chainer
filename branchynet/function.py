"""
计算gpu运行时的信息熵：作为在退出点的分类器对训练样本的置信度衡量
"""
from chainer.backends import cuda


def entropy_gpu(x):
    vec = cuda.elementwise(
        'T x',
        'T y',
        '''
            y = (x == 0) ? 0 : -x*log(x);
        ''',
        'entropy')(x.data)
    return cuda.cupy.sum(vec, 1)
