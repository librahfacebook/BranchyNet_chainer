'''
利用MNIST数据集在Lenet-5带分支网络模型上进行训练测试
'''
from branchynet.net import BranchyNet
from branchynet.links import *
from branchynet import utils, visualize
from chainer import serializers
from chainer import cuda
from networks.lenet import LeNet
from datasets import mnist
import dill
import chainer.functions as F
import chainer.links as L

# 定义网络
branchyNet = LeNet.build()
branchyNet.to_gpu()
branchyNet.training()
branchyNet.verbose=True

# 参数设置
TRAIN_BATCHSIZE = 512
TEST_BATCHSIZE = 1
TRAIN_NUM_EPOCHES = 20

# 导入MNIST数据集
X_train, Y_train, X_test, Y_test = mnist.get_data()

print(X_train.shape, Y_train.shape)

'''
# 在主网络上进行训练
main_loss,main_acc,main_time=utils.train(branchyNet,X_train,Y_train,main=True,batchsize=TRAIN_BATCHSIZE,
                                         num_epoch=TRAIN_NUM_EPOCHES)

# 可视化主网络训练结果
visualize.plot_layers(main_loss,xlabel='Epoches',ylabel='Training Loss')
visualize.plot_layers(main_acc,xlabel='Epoches',ylabel='Training Accuracy')


# 在带有分支网络层的网络上进行训练
branch_loss, branch_acc, branch_time = utils.train(branchyNet, X_train, Y_train, batchsize=TRAIN_BATCHSIZE,
                                                   num_epoch=TRAIN_NUM_EPOCHES)

# 可视化分支网络训练结果
visualize.plot_layers(list(zip(*branch_loss)), xlabel='Epoches', ylabel='Training Loss')
visualize.plot_layers(list(zip(*branch_acc)), xlabel='Epoches', ylabel='Training Accuracy')

#保存训练好的网络模型
with open("../models/lenet_mnist.bn","wb") as f:
    dill.dump(branchyNet,f)
'''

# 加载已保存好的模型
with open("../models/lenet_mnist.bn", "rb") as f:
    branchyNet = dill.load(f)
# 网络模型测试，获取主网络测试时间以及准确率
branchyNet.testing()
branchyNet.verbose = False
branchyNet.to_gpu()
g_baseacc, g_basediff, _, _ = utils.test(branchyNet, X_test, Y_test, main=True, batchsize=TEST_BATCHSIZE)
g_basediff = (g_basediff / float(len(Y_test))) * 1000.

print("g_baseacc:", g_baseacc)
print("g_basediff:", g_basediff)
# 退出点阈值设置
thresholds = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1., 2., 3., 5., 10.]

# 根据退出点阈值来获取代入单个退出点阈值的网络测试准确率、测试时间以及退出点样本数（带分支网络）
g_ts, g_accs, g_diffs, g_exits = utils.screen_branchy(branchyNet, X_test, Y_test, thresholds,
                                                      batchsize=TEST_BATCHSIZE, verbose=True)

g_diffs *= 1000.
# 显示原始网络与带有分支的网络测试精度与运行时间的关系图
visualize.plot_line_tradeoff(g_accs, g_diffs, g_ts, g_exits, g_baseacc, g_basediff,
                             all_samples=False, inc_amt=0.0001, our_label='BranchyLeNet',
                             orig_label='LeNet', xlabel='Runtime(ms)', title='LeNet Gpu',
                             output_path='../pic/lenet_gpu.png')
# 将结果保存为csv文件
utils.branchy_save_csv(g_baseacc, g_basediff, g_accs, g_diffs, g_exits, g_ts, filename='LeNet')

# 显示GPU下网络测试的结果（网络类型、准确率、运行时间、与原始网络的运行效率倍数、
# 退出点阈值和退出样本数比例）
print("GPU Results:")
utils.branchy_table_results(filename='LeNet')
