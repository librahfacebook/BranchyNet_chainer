'''
数据测试及统计函数
'''
from chainer import Variable
from itertools import product
import chainer
from branchynet.augment import augmentation
import numpy as np
import time, csv


def test_suite_A(branchyNet, x_test, y_test, batchsize=10000, ps=np.linspace(0.1, 1.0, 10)):
    accs = []
    diffs = []
    num_exits = []
    for p in ps:
        branchyNet.percentTestExits = p
        acc, diff, num_exit, _ = test(branchyNet, x_test, y_test, batchsize=batchsize)
        accs.append(acc)
        diffs.append(diff)
        num_exits.append(num_exit)
    return ps, accs, diffs


# 对于退出点阈值列表中的每一个阈值分别进行测试
def test_suite_B(branchyNet, x_test, y_test, batchsize=10000, ps=np.linspace(0.1, 2.0, 10)):
    accs = []  # 准确率列表
    diffs = []  # 测试时间列表
    num_exits = []
    for p in ps:
        branchyNet.thresholdExits = p  # 设置分支网络的阈值
        # 根据代入的阈值获取测试网络的准确率、测试时间以及退出点的样本数列表
        acc, diff, num_exit, _ = test(branchyNet, x_test, y_test, batchsize=batchsize)
        accs.append(acc)
        diffs.append(diff)
        num_exits.append(num_exit)
    return ps, np.array(accs), np.array(diffs) / float(len(y_test)), num_exits


def test_augment(branchyNet, x_test, y_test=None, batchsize=10000, main=False):
    datasize = x_test.shape[0]

    overall = 0.
    totaltime = 0.
    nsamples = 0
    num_exits = np.zeros(branchyNet.numexits()).astype(int)

    sum_accuracy = 0
    num = 0
    for x, t in augmentation(x_test, y_test, batchsize):
        total = len(x)

        x = branchyNet.xp.asarray(x, dtype=branchyNet.xp.float32)
        t = branchyNet.xp.asarray(t, dtype=branchyNet.xp.int32)

        x = Variable(x, volatile=True)
        t = Variable(t, volatile=True)

        if main:
            pred = branchyNet.run_main(x)
        else:
            pred = branchyNet.run(x)

        totaltime += branchyNet.runtime

        print("pred.shape", pred.shape)
        pred = pred.mean(axis=0)
        acc = int(pred.argmax() == t.data[0])
        sum_accuracy += acc
        num += 1

        for i, exits in enumerate(branchyNet.num_exits):
            num_exits[i] += exits

    overall = sum_accuracy / num

    return overall, totaltime, num_exits


'''
神经网络测试函数：返回结果为总体测试准确率、总测试时间；
每个退出点的退出样本数以及测试准确率(带有分支网络的测试网络模型)
'''


def test(branchyNet, x_test, y_test=None, batchsize=10000, main=False):
    datasize = x_test.shape[0]  # 测试样本集个数

    overall = 0.
    totaltime = 0.  # 测试总时间
    nsamples = 0  # 总测试样本数
    num_exits = np.zeros(branchyNet.numexits()).astype(int)  # 退出点列表（即每个退出点的退出测试样本数量）
    accbreakdowns = np.zeros(branchyNet.numexits())  # 每个退出点的准确率

    for i in range(0, datasize, batchsize):
        input_data = x_test[i: i + batchsize]
        label_data = y_test[i: i + batchsize]

        input_data = branchyNet.xp.asarray(input_data, dtype=branchyNet.xp.float32)
        label_data = branchyNet.xp.asarray(label_data, dtype=branchyNet.xp.int32)

        x = Variable(input_data)
        t = Variable(label_data)

        # 在无反向传播模式下进行
        with chainer.no_backprop_mode():
            if main:
                # 在主网络下进行测试，返回结果为准确率以及测试时间
                acc, diff = branchyNet.test_main(x, t)
            else:
                # 在带有分支网络的模型下进行测试，返回结果为总体准确率，每个退出点的准确率以及测试样本数量，和该批测试样本测试时间
                acc, accuracies, test_exits, diff = branchyNet.test(x, t)
                # 获取每个退出点的退出测试样本数
                for i, exits in enumerate(test_exits):
                    num_exits[i] += exits
                # 获取每个退出点的准确率
                for i in range(branchyNet.numexits()):
                    accbreakdowns[i] += accuracies[i] * test_exits[i]

            totaltime += diff
            overall += input_data.shape[0] * acc
            nsamples += input_data.shape[0]

    overall /= nsamples

    for i in range(branchyNet.numexits()):
        if num_exits[i] > 0:
            accbreakdowns[i] /= num_exits[i]

    return overall, totaltime, num_exits, accbreakdowns


def get_SM(branchyNet, x_test, batchsize=10000):
    datasize = x_test.shape[0]
    exitHs = []

    for i in range(0, datasize, batchsize):
        input_data = x_test[i: i + batchsize]
        input_data = branchyNet.xp.asarray(input_data, dtype=branchyNet.xp.float32)
        x = Variable(input_data, volatile=True)
        exitHs.extend(branchyNet.get_SM(x))

    return exitHs


def traintest_augment(branchyNet, x_train, y_train, x_test, y_test, batchsize=10000, num_epoch=20, main=False):
    for i in range(num_epoch):
        branchyNet.training()
        plotlosses, plotaccuracies, totaltime = train_augment(branchyNet, x_train, y_train, batchsize=batchsize,
                                                              num_epoch=1, main=main)
        print("train losses", plotlosses)
        print("train accuracy", plotaccuracies)
        branchyNet.testing()
        overall, totaltime, num_exits = test_augment(branchyNet, x_test, y_test, batchsize=batchsize, main=main)
        print("test accuracy", overall)

    return plotlosses, plotaccuracies, totaltime


def traintest(branchyNet, x_train, y_train, x_test, y_test, batchsize=10000, num_epoch=20, main=False, verbose=False):
    total_time = 0.0
    losses, accs = [], []
    test_overall, test_num_exits = [], []
    test_totaltime = 0.0
    test_accbreakdowns = []
    for i in range(num_epoch):
        branchyNet.training()
        tr_loss, tr_acc, rt = train(branchyNet, x_train, y_train, batchsize=batchsize, num_epoch=1, main=main)
        if verbose:
            print("train losses", tr_loss[0])
            print("train accuracy", tr_acc[0])
        total_time += rt
        losses.append(tr_loss[0])
        accs.append(tr_acc[0])

        branchyNet.testing()
        t_acc, t_time, t_exits, t_accbreakdowns = test(branchyNet, x_test, y_test, batchsize=batchsize, main=main)
        if verbose:
            print("test accuracy", t_accbreakdowns)
            print("test exits", t_exits)
            print("test accuracy overall", t_acc)
        test_totaltime += t_time
        test_overall.append(t_acc)
        test_num_exits.append(t_exits)
        test_accbreakdowns.append(t_accbreakdowns)

    return losses, accs, total_time, test_overall, test_totaltime, test_num_exits, test_accbreakdowns


def train_augment(branchyNet, x_train, y_train, batchsize=10000, num_epoch=20, main=False):
    datasize = x_train.shape[0]

    plotepochs = range(num_epoch)
    plotlosses = []
    plotaccuracies = []

    sum_time = 0

    for epoch in plotepochs:
        sum_loss = 0
        sum_accuracy = 0
        sum_time = 0
        num = 0
        for x, t in augmentation(x_train, y_train, batchsize):
            total = len(x)
            branchyNet.training()
            losses, accuracies, totaltime = train(branchyNet, x, t, batchsize=total, num_epoch=1, main=main)
            sum_loss += losses[0] * total
            sum_accuracy += accuracies[0] * total
            sum_time += totaltime
            num += total

        avg_loss = sum_loss / num
        avg_accuracy = sum_accuracy / num
        plotlosses.append(avg_loss)
        plotaccuracies.append(avg_accuracy)

    return plotlosses, plotaccuracies, sum_time


'''神经网络训练函数：main=True时在主网络上训练，main=False时
    在带有分支的网络上进行训练。
    返回结果为损失函数列表、准确率列表以及运行时间
'''


def train(branchyNet, x_train, y_train, batchsize=10000, num_epoch=20, main=False):
    datasize = x_train.shape[0]  # 训练集总量

    losses_list = []  # 损失函数列表
    accuracies_list = []  # 准确率列表
    totaltime = 0  # 总运行时间

    for epoch in range(num_epoch):
        indexes = np.random.permutation(datasize)

        if branchyNet.verbose:
            print("Epoch{}:".format(epoch))

        losses = []
        accuracies = []

        for i in range(0, datasize, batchsize):
            input_data = x_train[indexes[i: i + batchsize]]
            label_data = y_train[indexes[i: i + batchsize]]

            input_data = branchyNet.xp.asarray(input_data, dtype=branchyNet.xp.float32)
            label_data = branchyNet.xp.asarray(label_data, dtype=branchyNet.xp.int32)

            x = Variable(input_data)
            t = Variable(label_data)

            start_time = time.time()  # 一批次开始时间
            if main:
                loss, accuracy = branchyNet.train_main(x, t)
            else:
                loss, accuracy = branchyNet.train(x, t)
            end_time = time.time()  # 一批次结束时间
            run_time = end_time - start_time  # 一批次运行时间
            totaltime += run_time  # 总时间

            losses.append(loss)
            accuracies.append(accuracy)

        # 均值化处理
        avg_losses = branchyNet.xp.mean(branchyNet.xp.asarray(losses, dtype=branchyNet.xp.float32), 0)
        avg_accuracies = branchyNet.xp.mean(branchyNet.xp.asarray(accuracies, dtype=branchyNet.xp.float32), 0)

        losses_list.append(avg_losses)
        accuracies_list.append(avg_accuracies)
    return losses_list, accuracies_list, totaltime


# 生成阈值列表(根据退出点数来分配阈值列表)
def generate_thresholds(base_ts, num_layers):
    ts = list(product(*([base_ts] * (num_layers - 1))))
    ts = [list(l) for l in ts]

    return ts


'''迭代计算获取标志性的带分支网络测试数据点列表，
    使其准确率随着运行时间的增大而增长，成正比例关系
'''


def get_inc_points(accs, diffs, ts, exits, inc_amt=-0.0005):
    idxs = np.argsort(diffs)  # 获取运行时间的升序索引
    accs = np.array(accs)
    diffs = np.array(diffs)
    inc_accs = [accs[idxs[0]]]  # 将最短运行时间下的准确率放入准确率列表
    inc_rts = [diffs[idxs[0]]]  # 最短运行时间放入运行时间列表
    inc_ts = [ts[idxs[0]]]  # 最短时间下所对应的退出点阈值放入退出点阈值列表
    inc_exits = [exits[idxs[0]]]  # 最短时间下的退出点样本数放入退出点样本数列表
    # 除最短时间外按照时间由低到高排序，如果其准确率高于inc_accs[-1]+inc_amr，则将其放入对应列表中
    for i, idx in enumerate(idxs[1:]):
        if accs[idx] > inc_accs[-1] + inc_amt:
            inc_accs.append(accs[idx])
            inc_rts.append(diffs[idx])
            inc_ts.append(ts[idx])
            inc_exits.append(exits[idx])

    return inc_accs, inc_rts, inc_ts, inc_exits


# 根据退出点阈值来获取代入单个退出点阈值的网络测试准确率、测试时间以及退出点样本数（带分支网络）
def screen_branchy(branchyNet, x_test, y_test, base_ts, batchsize=1, enumerate_ts=True, verbose=False):
    # 生成退出点的阈值列表
    if enumerate_ts:
        ts = generate_thresholds(base_ts, branchyNet.numexits())
    else:
        ts = base_ts

    # 对于阈值列表中的每一个阈值分别进行测试
    ts, accs, diffs, exits = test_suite_B(branchyNet, x_test, y_test, batchsize=batchsize, ps=ts)

    return ts, accs, diffs, exits


# 输出网络测试结果
def branchy_table_results(filepath='../results/',filename=''):

    #csv文件读取
    data=csv.reader(open(filepath+filename+'.csv','r'))
    for i,d in enumerate(data):
        if i==0:
            print("{:>15}{:>15}{:>15}{:>15}{:>15}{:>15}".format(
                    d[0], d[1], d[2], d[3], d[4], d[5]))
        elif i==1:
            print("{:>15}{:>14.2f}{:>14.2f}{:>17.2f}{:>11}{:>15}".format(
                    d[0], float(d[1]), float(d[2]), float(d[3]), d[4], d[5]))
        else:
            print("{:>15}{:>14.2f}{:>14.2f}{:>17.2f}{:>15}{:>20}".format(
                d[0], float(d[1]), float(d[2]), float(d[3]), d[4], d[5]))


# 将网络测试结果保存为csv格式文件
def branchy_save_csv(baseacc,basediff,accs,diffs,exits,ts, filepath='../results/', filename=''):
    print_lst = lambda xs: '{' + ', '.join(map(str, xs)) + '}'
    data=list()
    #输出结果名字列表
    result_name=['Network','Acc.(%)','Time(ms)','Gain','Thrshld.T','Exit(%)']
    data.append(result_name)
    #原始网络结果
    base_result=[filename,baseacc*100.,basediff,1.00,'-','-']
    data.append(base_result)
    #带分支网络结果
    for i,(acc,diff,exit,t) in enumerate(zip(accs,diffs,exits,ts)):
        branch_result = ['B-'+filename,acc*100.,diff,basediff/diff,print_lst(t),
                         print_lst(round((100.*(e/float(sum(exit)))),2) for e in exit)]
        data.append(branch_result)

    # 打开文件，写入w
    out_file = open(filepath + filename+'.csv', 'w', newline='')
    # 设定写入模式
    csv_write = csv.writer(out_file, dialect='excel')
    #写入具体内容
    for d in data:
        csv_write.writerow(d)
    print("write csv over")

def compute_network_times(exits, branch_times):
    total_times = []
    for exit in exits:
        total_time = 0
        for e, t in zip(exit, branch_times):
            total_time += e * t

        total_time /= float(np.sum(exit))
        total_time *= 1000.
        total_times.append(total_time)

    return total_times


def compute_branch_times(net, x_test, y_test, batchsize=1, num_samples=200):
    thresholds = [10.]
    branch_times = []
    for i in range(len(net.models)):
        net.thresholdExits = thresholds
        _, branch_time, _ = test(net, x_test[:num_samples], y_test[:num_samples],
                                 batchsize=batchsize, main=False)
        branch_time /= float(len(y_test[:num_samples]))
        branch_times.append(branch_time)
        thresholds = [0.] + thresholds

    return branch_times


def get_ent_setdiff(old_ent_vals, ent_vals, percentExit=0.5):
    p_total = 0.
    diffs = []
    exited_old = np.empty(0)
    exited_new = np.empty(0)
    for i, (vo, vn) in enumerate(zip(old_ent_vals, ent_vals)):
        old_idxs = np.argsort(vo)
        new_idxs = np.argsort(vn)

        if i != 0:
            old_idxs = np.setdiff1d(old_idxs, exited_old)
            new_idxs = np.setdiff1d(new_idxs, exited_new)

        start_idx = 0
        end_idx = int(len(old_idxs) * percentExit)

        old_exits = old_idxs[start_idx:end_idx]
        new_exits = new_idxs[start_idx:end_idx]
        exited_old = np.concatenate((exited_old, old_exits))
        exited_new = np.concatenate((exited_new, new_exits))

        num_diff = len(np.setdiff1d(old_exits, new_exits))
        pct_diff = num_diff / float(end_idx - start_idx)
        diffs.append(pct_diff)

    return diffs
