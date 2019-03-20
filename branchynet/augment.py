'''
模型训练名称定义以及参数名设置
'''
from skimage.io import imsave
from scipy.misc import imresize
from multiprocessing import Process, Queue, Pool
import os, argparse, six, time
import numpy as np


# 图片格式的裁剪与转换
class Transform(object):
    cropping_size = 24
    scaling_size = 28

    def __call__(self, img):
        imgs = []
        for offset_y in six.moves.range(0, 12, 4):
            for offset_x in six.moves.range(0, 12, 4):
                im = img[offset_y:offset_y + self.cropping_size,
                     offset_x:offset_x + self.cropping_size]
                im = im.astype(np.float)
                im -= im.reshape(-1, 3).mean(axis=0)
                im -= im.reshape(-1, 3).std(axis=0) + 1e-5
                imgs.append(im)
                imgs.append(np.fliplr(im))

        for offset_y in six.moves.range(0, 6, 2):
            for offset_x in six.moves.range(0, 6, 2):
                im = img[offset_y:offset_y + self.scaling_size,
                     offset_x:offset_x + self.scaling_size]
                im = imresize(im, (self.cropping_size, self.cropping_size),
                              'nearest')
                im = im.astype(np.float)
                im -= im.reshape(-1, 3).mean(axis=0)
                im -= im.reshape(-1, 3).std(axis=0) + 1e-5
                imgs.append(im)
                imgs.append(np.fliplr(im))
        imgs = np.asarray(imgs, dtype=np.float32)

        return imgs


def augmentation_helper(aug_queue, data, label, batchsize):
    trans = Transform()
    np.random.seed(int(time.time()))
    perm = np.random.permutation(data.shape[0])
    for i in six.moves.range(0, data.shape[0], batchsize):
        chosen_ids = perm[i:i + batchsize]
        augs = []
        labels = []
        for j in chosen_ids:
            aug = trans(data[j])
            augs.append(aug)
            lb = np.repeat(label[j], len(aug))
            labels.append(lb)
        augs = np.vstack(augs).transpose((0, 3, 1, 2))
        labels = np.hstack(labels)

        x = np.asarray(augs, dtype=np.float32)
        t = np.asarray(labels, dtype=np.int32)

        aug_queue.put(x, t)
    aug_queue.put(None)
    return


def augmentation(data, label, batchsize):
    data = data.transpose((0, 2, 3, 1))

    aug_queue = Queue()
    aug_worker = Process(target=augmentation_helper,
                         args=(aug_queue, data, label, batchsize))
    aug_worker.start()

    while True:
        datum = aug_queue.get()
        if datum is None:
            break
        x, t = datum
        yield (x, t)
    aug_worker.join()
