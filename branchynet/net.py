'''
BranchyNet网络模型构造
'''

from branchynet.function import *
from branchynet.links.links import *
from chainer import Variable,optimizers
import cupy
import numpy as np
import time


class BranchyNet:
    def __init__(self, network, thresholdExits=None, percentTestExits=.9, percentTrianKeeps=1., learning_rate=0.1,
                 momentum=0.9,
                 weight_decay=0.0001, alpha=0.001, opt="Adam", joint=True, verbose=False):
        self.opt = opt  # 优化器
        self.alpha = alpha  # 参数
        self.weight_decay = weight_decay  # 权重衰减
        self.momentum = momentum  # 动量
        self.learning_rate = learning_rate  # 学习率
        self.joint = joint
        self.forwardMain = None  # 主网络前向传播

        self.main = Net()
        self.models = []
        starti = 0
        curri = 0
        for link in network:
            # 如果该层不在分支层，则将其添加到分支网络
            if not isinstance(link, Branch):
                curri += 1
                self.main.add_link(link)
            else:
                net = Net(link.weight)
                net.starti = starti
                starti = curri
                net.endi = curri
                # 添加分支层前的网络
                for prevlink in self.main:
                    newlink = copy.deepcopy(prevlink)
                    newlink.name = None
                    net.add_link(newlink)
                # 添加分支网络层
                for branchlink in link:
                    newlink = copy.deepcopy(branchlink)
                    newlink.name = None
                    net.add_link(newlink)
                self.models.append(net)
        for branchlink in link:
            newlink = copy.deepcopy(branchlink)
            newlink.name = None
            self.main.add_link(newlink)

        # 优化器函数设置
        if self.opt == 'MomentumSGD':
            self.optimizer = optimizers.MomentumSGD(learning_rate=self.learning_rate, momentum=self.momentum)
        else:
            self.optimizer = optimizers.Adam(alpha=self.alpha)
        self.optimizer.setup(self.main)

        if self.opt == 'MomentumSGD':
            self.optimizer.add_hook(chainer.optimizer.WeightDecay(self.weight_decay))

        self.optimizers = []

        for model in self.models:
            if self.opt == 'MomentumSGD':
                optimizer = optimizers.MomentumSGD(learning_rate=self.learning_rate, momentum=0.9)
            else:
                optimizer = optimizers.Adam()
            optimizer.setup(model)

            if self.opt == 'MomentumSGD':
                optimizer.add_hook(chainer.optimizer.WeightDecay(self.weight_decay))

            self.optimizers.append(optimizer)

        self.percentTrainKeeps = percentTrianKeeps  # 训练集样本保持比例
        self.percentTestExits = percentTestExits  # 测试集样本退出比例
        self.thresholdExits = thresholdExits  # 退出点的阈值
        self.clearLearnedExitsThresholds()

        self.verbose = verbose
        self.gpu = False
        self.xp = np

    # 获取退出点的阈值函数
    def getLearnedExitsThresholds(self):
        return self.learnedExitsThresholds / self.learnedExitsThresholdsCount

    # 退出点阈值置零
    def clearLearnedExitsThresholds(self):
        self.learnedExitsThresholds = np.zeros(len(self.models))
        self.learnedExitsThresholdsCount = np.zeros(len(self.models))

    # 获取退出点数量
    def numexits(self):
        return len(self.models)

    # 设置模型训练
    def training(self):
        # 主网络训练
        for link in self.main:
            link.train = True
        # 分支网络训练
        for model in self.models:
            for link in model:
                link.train = True

    # 设置模型测试
    def testing(self):
        # 主网络测试
        for link in self.main:
            link.train = False
        # 分支网络测试
        for model in self.models:
            for link in model:
                link.train = False

    # 设置模型在gpu上训练
    def to_gpu(self):
        self.xp = cupy
        self.gpu = True
        self.main.to_gpu()
        for model in self.models:
            model.to_gpu()

    # 设置模型在cpu上训练
    def to_cpu(self):
        self.xp = np
        self.gpu = False
        self.main.to_cpu()
        for model in self.models:
            model.to_cpu()

    # 深化主网络上的copy操作
    def copy_main(self):
        self.main_copy = copy.deepcopy(self.main)
        return

    # 复制主网络层运行函数
    def train_main_copy(self, x, t=None):
        return self.train_model(self.main_copy, x, t)

    # 复制主网络层测试函数
    def test_main_copy(self, x, t=None):
        return self.test_model(self.main_copy, x, t)

    # 主网络运行函数
    def train_main(self, x, t=None):
        return self.train_model(self.main, x, t)

    # 单个分支网络运行函数
    def train_branch(self, i, x, t=None):
        return self.train_model(self.models[i], x, t)

    # 主网络测试函数
    def test_main(self, x, t=None):
        return self.test_model(self.main, x, t)

    # 主网络模型训练
    def train_model(self, model, x, t=None):
        self.main.cleargrads()
        loss = self.main.train(x, t)  # 获取模型训练损失函数
        accuracy = self.main.accuracy  # 获取模型训练准确率
        loss.backward()  # 通过loss损失函数进行后向传递
        self.optimizer.update()  # 利用优化器来更新参数
        if self.gpu:
            lossesdata = loss.data.get()
            accuraciesdata = accuracy.data
        else:
            lossesdata = loss.data
            accuraciesdata = accuracy.data

        # 输出损失函数值和准确率
        if self.verbose:
            print("losses: {}, accuracies: {}".format(lossesdata, accuraciesdata))

        return lossesdata, accuraciesdata

    # 主网络模型测试
    def test_model(self, model, x, t=None):
        totaltime = 0
        start_time = time.time()  # 测试开始时间
        h = self.main.test(x)  # 开始在主网络模型上进行测试
        end_time = time.time()  # 测试结束时间
        totaltime += end_time - start_time

        accuracy = F.accuracy(h, t)  # 获取测试结果的准确率
        if self.gpu:
            accuracydata = accuracy.data.get()
        else:
            accuracydata = accuracy.data

        if self.verbose:
            print("accuracies", accuracydata)

        return accuracydata, totaltime

    # 带有分支网络的神经网络训练过程
    def train(self, x, t=None):
        # 将带有分支网络的model和主网络的共有层实现参数共享
        for i, link in enumerate(self.main):
            for model in self.models:
                for j, modellink in enumerate(model[:model.endi]):
                    if i == j:
                        modellink.copyparams(link)

        # 参数重置
        self.main.cleargrads()
        [model.cleargrads() for model in self.models]

        # forwardMain不为空：在主网络中训练，获取主网络的损失函数
        if self.forwardMain is not None:
            mainLoss = self.main.train(x, t)

        # 剩余的训练样本集
        remainingXVar = x
        remainingTVar = t

        numexits = []  # 退出的样本个数
        losses = []  # 损失函数
        accuracies = []  # 准确率
        nummodels = len(self.models)  # 带有分支网络的model个数
        numsamples = x.data.shape[0]  # 训练的样本个数

        # 利用各个分支网络来对数据集进行训练，从而可以使部分训练样本提前退出
        for i, model in enumerate(self.models):

            # 剩余样本集为空，训练提前退出
            if type(remainingXVar) == None or type(remainingTVar) == None:
                break

            # 获取此时分支网络的model训练后的损失函数和准确率
            loss = model.train(remainingXVar, remainingTVar)
            losses.append(loss)
            accuracies.append(model.accuracy)

            # 如果已经到达最后一个退出点，则直接退出（此时全部样本直接退出）
            if i == nummodels - 1:
                break

            # 获取此退出点的信息熵
            softmax = F.softmax(model.h)
            if self.gpu:
                entropy_value = entropy_gpu(softmax).get()
            else:
                entropy_value = np.array([entropy(s) for s in softmax.data])
            total = entropy_value.shape[0]
            idx = np.zeros(total, dtype=bool)  # 该标志用来判断哪些样本被提前退出，True代表退出

            # 退出点阈值不为空
            if self.thresholdExits is not None:
                min_ent = 0
                # min_ent = min(entropy_value)
                # 判断退出点阈值thresholdExits是否为列表类型
                # 若退出点样本的信息熵小于其阈值，则将idx[样本位置]设置为真，表明此时该样本可以提前退出
                if isinstance(self.thresholdExits, list):
                    idx[entropy_value < min_ent + self.thresholdExits[i]] = True
                    numexit = sum(idx)
                else:
                    idx[entropy_value < min_ent + self.thresholdExits] = True
                    numexit = sum(idx)

            # 训练集退出样本比例存在且不为空，从而获取退出样本总数
            elif hasattr(self, 'percentTrainExits') and self.percentTrainExits is not None:
                if isinstance(self.percentTestExits, list):
                    numexit = int(self.percentTrainExits[i] * numsamples)
                else:
                    numexit = int(self.percentTrainExits * total)
                # 将样本集按照信息熵值从小到大排序，将前numexit个最小的idx设为True
                esorted = entropy_value.argsort()
                idx[esorted[:numexit]] = True

            # 训练集中仍然保持训练的比例存在且不为空
            else:
                if isinstance(self.percentTrainKeeps, list):
                    numkeep = (self.percentTrainKeeps[i] * numsamples)
                else:
                    numkeep = self.percentTrainKeeps * total
                numexit = int(total - numkeep)
                esorted = entropy_value.argsort()  # 返回从小到大的索引值
                idx[esorted[:numexit]] = True

            numkeep = int(total - numexit)  # 训练集中仍训练的样本数
            numexits.append(numexit)  # 将该分支网络退出的样本数加入到退出数列中

            if self.gpu:
                xdata = remainingXVar.data.get()
                tdata = remainingTVar.data.get()
            else:
                xdata = remainingXVar.data
                tdata = remainingTVar.data

            # 若此时训练网络中还有样本，则将训练集中idx为False的样本
            # 保存为剩余训练样本集，否则置为空
            if numkeep > 0:
                with chainer.no_backprop_mode():
                    remainingXVar = Variable(self.xp.array(xdata[~idx]))
                    remainingTVar = Variable(self.xp.array(tdata[~idx]))
            else:
                remainingXVar = None
                remainingTVar = None

        # 若forwardMain不为空，则主网络反向传播更新权值
        if self.forwardMain is not None:
            mainLoss.backward()

        # 各分支网络反向传播更新权值
        for i, loss in enumerate(losses):
            net = self.models[i]
            loss = net.weight * loss
            loss.backward()

        # 向主网络添加其分支网络梯度值
        if self.joint:
            if self.forwardMain is not None:
                models = self.models[:-1]
            else:
                models = self.models
            for i, link in enumerate(self.main):
                for model in models:
                    for j, modellink in enumerate(model[:model.endi]):
                        if i == j:
                            link.addgrads(modellink)
        else:
            # 从最后一个分支网络层积累梯度值
            for i, link in enumerate(self.main):
                for model in self.models[-1:]:
                    for j, modellink in enumerate(model[:model.endi]):
                        if i == j:
                            link.addgrads(modellink)
        # 优化器更新参数
        self.optimizer.update()
        [optimizer.update() for optimizer in self.optimizers]

        # 重新将带有分支网络的model和主网络的共有层实现参数共享
        for i, link in enumerate(self.main):
            for model in self.models:
                for j, modellink in enumerate(model[:model.endi]):
                    if i == j:
                        modellink.copyparams(link)

        if self.gpu:
            lossesdata = [loss.data.get() for loss in losses]
            accuraciesdata = [accuracy.data.get() for accuracy in accuracies]
        else:
            lossesdata = [loss.data for loss in losses]
            accuraciesdata = [accuracy.data for accuracy in accuracies]

        # 输出每个分支网络model的退出样本数、损失函数和准确率
        if self.verbose:
            print("numexits:{},losses:{},accuracies:{}".format(numexits, losses, accuracies))

        return lossesdata, accuraciesdata

    # 带有分支网络的模型测试函数
    def test(self, x, t=None):
        numexits = []  # 各个退出点的退出样本数列
        accuracies = []  # 各个退出点的准确率
        remainingXVar = x  # 仍然测试的样本集
        remainingTVar = t  # 仍然测试的样本标签集
        nummodels = len(self.models)  # 分支网络总数
        numsamples = x.data.shape[0]  # 总测试样本数
        totaltime = 0  # 总测试时间

        for i, model in enumerate(self.models):

            # 测试样本集或者标签集为空
            if remainingXVar is None or remainingTVar is None:
                numexits.append(0)
                accuracies.append(0)
                continue

            # 记录测试的运行时间
            start_time = time.time()
            h = model.test(remainingXVar, model.starti, model.endi)
            end_time = time.time()
            totaltime += end_time - start_time

            # 计算每个出口点的分类结果的熵，作为对预测的信心度量
            smh = model.test(h, model.endi)
            softmax = F.softmax(smh)
            if self.gpu:
                entropy_value = entropy_gpu(softmax).get()
            else:
                entropy_value = np.array([entropy(s) for s in softmax.data])

            # 判断当前出口点的熵，若小于阈值，则提前退出，否则继续到下一个退出点
            idx = np.zeros(entropy_value.shape[0], dtype=bool)
            if i == nummodels - 1:
                idx = np.ones(entropy_value.shape[0], dtype=bool)
                numexit = sum(idx)
            else:
                if self.thresholdExits is not None:
                    min_ent = 0
                    if isinstance(self.thresholdExits, list):
                        idx[entropy_value < min_ent + self.thresholdExits[i]] = True
                        numexit = sum(idx)
                    else:
                        idx[entropy_value < min_ent + self.thresholdExits] = True
                        numexit = sum(idx)
                else:
                    if isinstance(self.percentTestExits, list):
                        numexit = int((self.percentTestExits[i]) * numsamples)
                    else:
                        numexit = int(self.percentTestExits * entropy_value.shape[0])
                    esorted = entropy_value.argsort()
                    idx[esorted[:numexit]] = True

            total = entropy_value.shape[0]
            numkeep = total - numexit
            numexits.append(numexit)

            if self.gpu:
                xdata = h.data.get()
                tdata = remainingTVar.data.get()
            else:
                xdata = h.data
                tdata = remainingTVar.data

            # 若numkeep>0，则将总测试样本集中idx为false的样本保留为待测试样本
            if numkeep > 0:
                xdata_keep = xdata[~idx]
                tdata_keep = tdata[~idx]
                remainingXVar = Variable(self.xp.array(xdata_keep, dtype=x.data.dtype))
                remainingTVar = Variable(self.xp.array(tdata_keep, dtype=t.data.dtype))
            else:
                remainingXVar = None
                remainingTVar = None

            # 获取退出的测试样本集，并计算得到退出点测试样本的准确率以及信息熵
            if numexit > 0:
                xdata_exit = xdata[idx]
                tdata_exit = tdata[idx]
                exitXVar = Variable(self.xp.array(xdata_exit, dtype=x.data.dtype))
                exitTVar = Variable(self.xp.array(tdata_exit, dtype=t.data.dtype))

                with chainer.no_backprop_mode():
                    exitH = model.test(exitXVar, model.endi)
                    accuracy = F.accuracy(exitH, exitTVar)

                    if self.gpu:
                        accuracies.append(accuracy.data.get())
                    else:
                        accuracies.append(accuracy.data)
            else:
                accuracies.append(0.)
        # 计算退出点的总准确率
        overall = 0
        for i, accuracy in enumerate(accuracies):
            overall += accuracy * numexits[i]
        overall /= np.sum(numexits)

        # 输出每个退出点的退出样本数，准确率以及整个网络的总体准确率
        if self.verbose:
            print("numexits", numexits)
            print("accuracies", accuracies)
            print("overall accuracy", overall)

        return overall, accuracies, numexits, totaltime

    # 主网络运行数据统计
    def run_main(self, x):
        totaltime = 0
        start_time = time.time()
        h = self.main.test(x)
        end_time = time.time()
        totaltime += end_time - start_time
        self.num_exits = [len(x.data)]
        self.runtime = totaltime
        return h.data

    # 运行过程
    def run(self, x, t):
        hs = []
        numexits = []
        accuracies = []
        remainingXVar = x
        remainingTVar = t
        nummodels = len(self.models)
        numsamples = x.data.shape[0]

        totaltime = 0
        for i, model in enumerate(self.models):
            if isinstance(remainingXVar, type(None)) or isinstance(remainingTVar, type(None)):
                break

            start_time = time.time()
            h = model.test(remainingXVar, model.starti, model.endi)
            end_time = time.time()
            totaltime += end_time - start_time

            smh = model.test(h, model.endi)
            softmax = F.softmax(smh)
            if self.gpu:
                entropy_value = entropy_gpu(softmax).get()
            else:
                entropy_value = np.array([entropy(s) for s in softmax.data])

            # 判断当前出口点的熵，若小于阈值，则提前退出，否则继续到下一个退出点
            idx = np.zeros(entropy_value.shape[0], dtype=bool)
            if i == nummodels - 1:
                idx = np.ones(entropy_value.shape[0], dtype=bool)
                numexit = sum(idx)
            else:
                if self.thresholdExits is not None:
                    min_ent = 0
                    if isinstance(self.thresholdExits, list):
                        idx[entropy_value < min_ent + self.thresholdExits[i]] = True
                        numexit = sum(idx)
                    else:
                        idx[entropy_value < min_ent + self.thresholdExits] = True
                        numexit = sum(idx)
                else:
                    if isinstance(self.percentTestExits, list):
                        numexit = int((self.percentTestExits[i]) * numsamples)
                    else:
                        numexit = int(self.percentTestExits * entropy_value.shape[0])
                    esorted = entropy_value.argsort()
                    idx[esorted[:numexit]] = True

            total = entropy_value.shape[0]
            numkeep = total - numexit
            numexits.append(numexit)

            if self.gpu:
                xdata = h.data.get()
                tdata = remainingTVar.data.get()
            else:
                xdata = h.data
                tdata = remainingTVar.data

            if numkeep > 0:
                xdata_keep = xdata[~idx]
                tdata_keep = tdata[~idx]
                remainingXVar = Variable(self.xp.array(xdata_keep, dtype=x.data.dtype), volatile=x.volatile)
                remainingTVar = Variable(self.xp.array(tdata_keep, dtype=t.data.dtype), volatile=t.volatile)
            else:
                remainingXVar = None
                remainingTVar = None

            if numexit > 0:
                xdata_exit = xdata[idx]
                tdata_exit = tdata[idx]
                exitXVar = Variable(self.xp.array(xdata_exit, dtype=x.data.dtype), volatile=x.volatile)
                exitTVar = Variable(self.xp.array(tdata_exit, dtype=t.data.dtype), volatile=t.volatile)

                exitH = model.test(exitXVar, model.endi)
                hs.append(exitH.data)

        self.num_exits = numexits
        self.runtime = totaltime
        return np.vstack(hs)

    # 获取退出点的信息熵
    def get_SM(self, x):
        numexits = []
        accuracies = []
        nummodels = len(self.models)
        numsamples = x.data.shape[0]
        exitHs = []
        h = x

        for i, model in enumerate(self.models):
            h = model.test(h, model.starti, model.endi)
            smh = model.test(h, model.endi)
            softmax = F.softmax(smh)
            exitHs.append(softmax.data)

        return exitHs

    # 获取退出点的阈值
    def find_thresholds_entropies(self, x_train, y_train, percentTrainKeeps=0.5, batchsize=1024):
        datasize = x_train.shape[0]
        nummodels = len(self.models) - 1
        thresholds = np.zeros(nummodels)
        entropy_values = [np.array([]) for i in range(nummodels)]

        for i in range(0, datasize, batchsize):
            input_data = x_train[i:i + batchsize]
            label_data = y_train[i:i + batchsize]

            input_data = self.xp.asarray(input_data, dtype=self.xp.float32)
            label_data = self.xp.asarray(label_data, dtype=self.xp.int32)

            x = Variable(input_data)
            t = Variable(label_data)

            # 向前传递获取信息熵和滤波器
            remainingXVar = x
            remainingTVar = t
            numsamples = x.data.shape[0]
            for i, model in enumerate(self.models[:-1]):
                if isinstance(remainingXVar, type(None)) or isinstance(remainingTVar, type(None)):
                    break
                loss = model.train(remainingXVar, remainingTVar)
                softmax = F.softmax(model.h)
                if self.gpu:
                    entropy_value = entropy_gpu(softmax).get()
                else:
                    entropy_value = np.array([entropy(s) for s in softmax.data])

                entropy_values[i] = np.hstack([entropy_values[i], entropy_value])

        for i, entropy_value in enumerate(entropy_values):
            idx = np.zeros(entropy_value.shape[0], dtype=bool)
            total = entropy_value.shape[0]
            if isinstance(percentTrainKeeps, list):
                numkeep = percentTrainKeeps[i] * numsamples
            else:
                numkeep = percentTrainKeeps * total
            numexit = int(total - numkeep)
            esorted = entropy_value.argsort()
            thresholds[i] = entropy_value[esorted[numexit]]

        return thresholds.tolist(), entropy_value

    def find_thresholds(self, x_train, y_train, percentTrainKeeps=0.5, batchsize=1024):
        thresholds, _ = self.find_thresholds_entropies(x_train, y_train, percentTrainKeeps=percentTrainKeeps,
                                                       batchsize=batchsize)
        return thresholds

    def find_entropies(self, x_train, y_train, percentTrainKeeps=0.5, batchsize=1024):
        _, entropies = self.find_thresholds_entropies(x_train, y_train, percentTrainKeeps=percentTrainKeeps,
                                                      batchsize=batchsize)
        return entropies


# 网络模型结构的输出
def print_models(self):
    for model in self.models:
        print("----", model.starti, model.endi)
        for link in model:
            print(link)
    print("----", self.main.starti, model.endi)
    for link in self.main:
        print(link)
    print("----")
