"""
B-AlexNet中第一个分支的卷积层数对最终分支分类精度的影响
"""
from branchynet import utils, visualize
from networks.alexnet import AlexNet
from datasets import cifar10
import dill

# 参数设置
TRAIN_BATCHSIZE = 512
TEST_BATCHSIZE = 32
TRAIN_NUM_EPOCHES = 100
NUMS = 4
SAVE_PATH = '../pic/alexnet_cifar10/'  # 实验结果图片保存路径
SAVE_NAME = 'alexnet_layers'
MODEL_PATH = '../models/alexnet_layers/'  # 保存模型路径名称


# 显示网络模型的最后一个分支网络准确率与第一个分支网络卷积层数的变化关系图
def alexnet_layers(n, MODEL_NAME):
    branchyNet = AlexNet().build(n)
    branchyNet.to_gpu()
    branchyNet.training()
    branchyNet.verbose = False

    # 在主网络上进行训练
    main_loss, main_acc, main_time = utils.train(branchyNet, X_train, Y_train, main=True, batchsize=TRAIN_BATCHSIZE,
                                                 num_epoch=TRAIN_NUM_EPOCHES)
    print("main_time:", main_time)

    # 在带有分支的网络上进行训练
    branch_loss, branch_acc, branch_time = utils.train(branchyNet, X_train, Y_train, batchsize=TRAIN_BATCHSIZE,
                                                       num_epoch=TRAIN_NUM_EPOCHES)
    print("branch_time:", branch_time)

    # 保存已经训练好的网络模型
    with open(MODEL_PATH + MODEL_NAME, "wb") as f:
        dill.dump(branchyNet, f)

    # 加载已经保存好的模型
    with open(MODEL_PATH + MODEL_NAME, "rb") as f:
        branchyNet = dill.load(f)

    # 网络模型测试，获取分支网络的准确率
    branchyNet.testing()
    branchyNet.verbose = False
    branchyNet.to_gpu()
    # 退出点阈值设置
    thresholdExits = [0.05, 0.01]
    branchyNet.thresholdExits = thresholdExits
    # 根据退出点阈值来获取分支网络的总体准确率
    overacc, _, _, accbreakdowns, _ = utils.test(branchyNet, X_test, Y_test, batchsize=TEST_BATCHSIZE)

    return accbreakdowns


if __name__ == '__main__':
    # 导入cifar10数据集
    X_train, Y_train, X_test, Y_test = cifar10.get_data()

    print("X_train:{} Y_train:{}".format(X_train.shape, Y_train.shape))
    print("X_test: {} Y_test: {}".format(X_test.shape, Y_test.shape))

    accs_exit = []
    for i in range(NUMS + 1):
        MODEL_NAME = 'B-AlexNet(' + str(i) + ').bn'
        # 第一个分支卷积层数为i的情况下最后一个退出点的准确率
        overacc, accbreakdowns = alexnet_layers(i, MODEL_NAME)
        # accs_exit.append(accbreakdowns[-1])
        accs_exit.append(overacc)
    visualize.plot_acc_layers(accs_exit, save_path=SAVE_PATH, save_name=SAVE_NAME)
