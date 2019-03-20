'''
数据以及运行结果的可视化表现
'''
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from chainer import Variable
import chainer.function as F

from scipy.stats import entropy
from branchynet import utils

#绘制网络层训练的准确率或损失函数随迭代次数变化的折线图
def plot_layers(values, save_name=None, xlabel='', ylabel=''):
    font = {'family': 'sans-serif',
            'weight': 'normal',
            'size': 18}
    matplotlib.rc('font', **font)

    plt.figure(figsize=(12, 10))
    if type(values[0]) in [tuple, list]:
        for i, val in enumerate(values):
            plt.plot(val, label='Layer ' + str(i), linewidth=4.0)
    else:
        plt.plot(values, label='Last Layer', linewidth=4.0)

    plt.legend(loc='best')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if save_name != None:
        plt.title(save_name)
        plt.savefig(save_name + '.png')
    plt.show()


def plot_tradeoff(ps, accs, diffs, baseacc, basediff):
    baseaccs = [baseacc] * len(ps)
    basediffs = [basediff] * len(ps)

    font = {'family': 'sans-serif',
            'weight': 'normal',
            'size': 18}
    matplotlib.rc('font', **font)

    fig = plt.figure(figsize=(12, 10))
    lns = []
    ax = fig.add_subplot(111)
    l = ax.plot(ps, np.array(accs) * 100., '-g', label='Accuracy', linewidth=5.0)
    lns += l
    l = ax.plot(ps, baseaccs, '--g', label='Baseline Accuracy', linewidth=5.0)
    lns += l
    ax.set_xlabel('Percentage Exit Factor')
    ax.set_ylabel('Accuracy')

    ax2 = ax.twinx()
    l = ax2.plot(ps, diffs, '-r', label='Time', linewidth=5.0)
    lns += l
    l = ax2.plot(ps, basediffs, '--r', label='Baseline Time', linewidth=5.0)
    lns += l
    ax2.set_ylabel('Runtime (s)')

    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='best')


def plot_roc(ps, accs, diffs, baseacc, basediff):
    font = {'family': 'sans-serif',
            'weight': 'normal',
            'size': 18}
    matplotlib.rc('font', **font)

    plt.figure(figsize=(12, 10))

    accs = np.array(accs)

    idxs = np.argsort(diffs)
    max_acc = accs[idxs][0]
    keep_idxs = [idxs[0]]

    for i, acc in enumerate(accs[idxs]):
        if acc > max_acc:
            max_acc = acc
            keep_idxs.append(i)

    keep_idxs = np.array(keep_idxs)

    plt.plot(diffs, accs, linewidth=4, label='Our Method')
    plt.plot(basediff, baseacc, 'D', linewidth=4, label='Baseline')
    plt.xlabel('Runtime (s)')
    plt.ylabel('Overall Accuracy')
    plt.legend(loc=0)

# 可视化原始网络与带有分支的网络（不同退出点阈值）测试精度与运行时间的关系图
def plot_line_tradeoff(accs, diffs, ps, exits, baseacc, basediff, orig_label='Baseline', title=None,
                       our_label='Our Method',
                       xlabel='Runtime (s)', ylabel='Classification Accuracy', all_samples=False, knee_idx=None,
                       xlim=None, ylim=None, inc_amt=-0.0005, output_path=None):
    matplotlib.rcParams.update({'axes.labelsize': 10,
                                'text.fontsize': 18,
                                'legend.fontsize': 15,
                                'xtick.labelsize': 13,
                                'ytick.labelsize': 13,
                                'axes.labelsize': 18,
                                'text.usetex': False,
                                'figure.figsize': [4.5, 3.5]})

    matplotlib.rcParams['legend.numpoints'] = 1
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    #获取准确率与运行时间成正比例关系的数据列表
    inc_accs, inc_rts, _, _ = utils.get_inc_points(accs, diffs, ps, exits, inc_amt=inc_amt)
    #若其为True，则绘制整个测试网络下的数据图
    if all_samples:
        plt.plot(diffs, accs, 'go', linewidth=4, label=our_label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    #绘制关键点
    if knee_idx:
        plt.plot(inc_rts[0], np.array(inc_accs[0]) * 100.,
                 '-o', color='#5163d3', ms=6, linewidth=4, label=our_label)
        plt.plot(inc_rts, np.array(inc_accs) * 100.,
                 '-', color='#5163d3', ms=6, linewidth=4)
        plt.plot(np.delete(inc_rts, knee_idx), np.delete(np.array(inc_accs) * 100., knee_idx),
                 'o', color='#5163d3', ms=6)
        plt.plot(inc_rts[knee_idx], np.array(inc_accs)[knee_idx] * 100., '*', color='#50d350', ms=16,
                 label=our_label + ' knee')

    else:
        plt.plot(inc_rts, np.array(inc_accs) * 100.,
                 '-o', color='#5163d3', ms=6, linewidth=4, label=our_label)
    #绘制原始网络测试数据点
    plt.plot(basediff, baseacc * 100., 'D', color='#d3515a', ms=8, label=orig_label)

    plt.legend(loc='best')

    if title:
        plt.title(title)

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.show()


def plot_layer_entropy(leakyNet, x):
    font = {'family': 'sans-serif',
            'weight': 'normal',
            'size': 18}
    matplotlib.rcParams['legend.numpoints'] = 1
    matplotlib.rc('font', **font)

    leakyNet.to_cpu()
    x = leakyNet.xp.asarray(x, dtype=leakyNet.xp.float32)
    h = Variable(x, volatile=True)
    ents = []
    for model in leakyNet.models:
        h = model.test(h, model.starti, model.endi)
        smh = model.test(h, model.endi)
        softmax = F.softmax(smh)
        entropy_value = np.array([entropy(s) for s in softmax.data])
        ents.append(entropy_value)

    plt.figure(figsize=(12, 10))
    for i, ent in enumerate(ents):
        plt.plot(sorted(ent, reverse=False), linewidth=4, label='Exit ' + str(i + 1))

    plt.legend(loc='best')
    # plt.yscale('log')
    plt.xlabel('Sorted Samples by Entropy')
    plt.ylabel('Entropy')


def plot_exits(g_exits, g_accs, g_exits2, g_accs2, i=0, labels=['Joint', 'Separate']):
    matplotlib.rcParams.update({'axes.labelsize': 10,
                                'text.fontsize': 18,
                                'legend.fontsize': 15,
                                'xtick.labelsize': 13,
                                'ytick.labelsize': 13,
                                'axes.labelsize': 18,
                                'text.usetex': False,
                                'figure.figsize': [4.5, 3.5]})

    matplotlib.rcParams['legend.numpoints'] = 1
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    plt.plot(np.array(g_exits)[:, i], g_accs * 100, linewidth=4, label=labels[0])
    plt.plot(np.array(g_exits2)[:, i], g_accs2 * 100, linewidth=4, label=labels[1])
    plt.legend(loc='best')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Num Exits')


def plot_runtimes(g_diffs, g_accs, g_diffs2, g_accs2, i=0, labels=['Joint', 'Separate']):
    matplotlib.rcParams.update({'axes.labelsize': 10,
                                'text.fontsize': 18,
                                'legend.fontsize': 15,
                                'xtick.labelsize': 13,
                                'ytick.labelsize': 13,
                                'axes.labelsize': 18,
                                'text.usetex': False,
                                'figure.figsize': [4.5, 3.5]})

    matplotlib.rcParams['legend.numpoints'] = 1
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    plt.plot(g_diffs, g_accs * 100, linewidth=4, label=labels[0])
    plt.plot(g_diffs2, g_accs2 * 100, linewidth=4, label=labels[1])
    plt.legend(loc='best')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Runtime (ms)')
