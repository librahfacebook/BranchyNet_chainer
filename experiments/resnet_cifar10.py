"""
利用cifar-10数据集在ResNet带分支网络模型上进行测试
"""
from branchynet import utils, visualize
from networks.resnet import ResNet
from datasets import cifar10
import dill

# 定义ResNet网络
branchyNet = ResNet().build()
branchyNet.to_gpu()
branchyNet.training()
branchyNet.verbose = True

# 参数设置
TRAIN_BATCHSIZE = 32
TEST_BATCHSIZE = 128
TRAIN_NUM_EPOCHES = 1
SAVE_PATH = '../pic/resnet_cifar10/'  # 实验结果图片保存路径
MODEL_NAME = '../models/resnet_cifar10(' + str(TRAIN_NUM_EPOCHES) + ').bn'  # 保存模型名称
CSV_NAME = 'resnet(' + str(TRAIN_NUM_EPOCHES) + ')'  # 输出文件名称

# 导入cifar10数据集
X_train, Y_train, X_test, Y_test = cifar10.get_data()

print("X_train:{} Y_train:{}".format(X_train.shape, Y_train.shape))
print("X_test: {} Y_test: {}".format(X_test.shape, Y_test.shape))

# 在主网络上进行训练
main_loss, main_acc, main_time = utils.train(branchyNet, X_train, Y_train, main=True, batchsize=TRAIN_BATCHSIZE,
                                             num_epoch=TRAIN_NUM_EPOCHES)
print("main_time:", main_time)

# 可视化主网络训练结果
visualize.plot_layers(main_loss, save_path=SAVE_PATH,
                      save_name='main_loss(' + str(TRAIN_NUM_EPOCHES) + ')',
                      xlabel='Epoches', ylabel='Training Loss')
visualize.plot_layers(main_acc, save_path=SAVE_PATH,
                      save_name='main_acc(' + str(TRAIN_NUM_EPOCHES) + ')',
                      xlabel='Epoches', ylabel='Training Accuracy')

# 在带有分支的网络上进行训练
branch_loss, branch_acc, branch_time = utils.train(branchyNet, X_train, Y_train, batchsize=TRAIN_BATCHSIZE,
                                                   num_epoch=TRAIN_NUM_EPOCHES)
print("branch_time:", branch_time)
# 可视化分支网络训练结果
visualize.plot_layers(list(zip(*branch_loss)), save_path=SAVE_PATH,
                      save_name='branch_loss(' + str(TRAIN_NUM_EPOCHES) + ')',
                      xlabel='Epoches', ylabel='Training Loss')
visualize.plot_layers(list(zip(*branch_acc)), save_path=SAVE_PATH,
                      save_name='branch_acc(' + str(TRAIN_NUM_EPOCHES) + ')',
                      xlabel='Epoches', ylabel='Training Accuracy')

# 保存已经训练好的网络模型
with open(MODEL_NAME, "wb") as f:
    dill.dump(branchyNet, f)

# 加载已经保存好的模型
with open(MODEL_NAME, "rb") as f:
    branchyNet = dill.load(f)

# 网络模型测试，获取基本网络测试时间和准确率
branchyNet.testing()
branchyNet.verbose = False
branchyNet.to_gpu()
g_baseacc, g_basediff, _, _, _ = utils.test(branchyNet, X_test, Y_test, main=True, batchsize=TEST_BATCHSIZE)
g_basediff = (g_basediff / float(len(Y_test))) * 1000.

print("g_baseacc:", g_baseacc)
print("g_basediff:", g_basediff)
# 退出点阈值设置
thresholds = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1., 2., 3., 5., 10.]

# 根据退出点阈值来获取代入单个退出点阈值的网络测试准确率、测试时间以及退出点样本数（带分支网络）
branchyNet.to_gpu()
g_ts, g_accs, g_diffs, g_exits, g_entropies = utils.screen_branchy(branchyNet, X_test, Y_test, thresholds,
                                                                   batchsize=TEST_BATCHSIZE, verbose=True)

g_diffs *= 1000.

# 绘制网络的总体准确率、第一个退出点样本最大信息熵与第一个退出点退出样本比例的变化关系图
visualize.plot_acc_entropy_exit(g_accs, g_exits, g_entropies, save_path=SAVE_PATH,
                                save_name='resnet_branch1(' + str(TRAIN_NUM_EPOCHES) + ')')
# 显示原始网络与带有分支的网络测试精度与运行时间的关系图
visualize.plot_line_tradeoff(g_accs, g_diffs, g_ts, g_exits, g_baseacc, g_basediff,
                             all_samples=False, inc_amt=0.0001, our_label='BranchyResNet',
                             orig_label='ResNet', xlabel='Runtime(ms)', title='ResNet Gpu',
                             output_path=SAVE_PATH, output_name='resnet_gpu(' + str(TRAIN_NUM_EPOCHES) + ')')
# 将结果保存为csv文件
utils.branchy_save_csv(g_baseacc, g_basediff, g_accs, g_diffs, g_exits, g_ts, filename=CSV_NAME)

# 显示GPU下网络测试的结果（网络类型、准确率、运行时间、与原始网络的运行效率倍数、
# 退出点阈值和退出样本数比例）
print("GPU Results:")
utils.branchy_table_results(filename=CSV_NAME)
