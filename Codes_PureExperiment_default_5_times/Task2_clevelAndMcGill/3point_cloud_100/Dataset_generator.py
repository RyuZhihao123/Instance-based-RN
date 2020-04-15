import cv2
import os
import numpy as np
import shutil
from Configure import Config, MakeDir
import math

config = Config()

train_num = 600
val_num   = 200
test_num  = 200

original_num = 100  # initial number of points.

def AddNoise(_img):
    img = _img.copy()
    noises = np.random.uniform(0, 0.05, (config.image_height,config.image_width,1))
    img += noises
    _min = img.min()
    _max = img.max()
    img -= _min
    img /= (_max - _min)

    return img

def GenerateOneChart(origin_num = 10, max_added_num = 10):
    #print(origin_num)
    _pool = np.arange(0, config.image_width*config.image_height)  # [0,..., w*h]

    _added_num = np.random.randint(0, max_added_num+1)              # number of added points.
    _pts = np.random.choice(_pool, origin_num + _added_num, False)  # sampling w/o replacement

    imgA = np.ones(shape=(config.image_width, config.image_width, 1), dtype='float32')
    imgB = np.ones(shape=(config.image_width, config.image_width, 1), dtype='float32')

    for i in range(origin_num):   # origin_num
        x = _pts[i] % config.image_height
        y = _pts[i] // config.image_height
        imgA[x, y, 0] = 0.
        imgB[x, y, 0] = 0.

    for i in range(_added_num):   # added_num.
        x = _pts[i + origin_num] % config.image_height
        y = _pts[i + origin_num] // config.image_height
        imgB[x, y, 0] = 0.

    imgA = AddNoise(imgA)
    imgB = AddNoise(imgB)

    return imgA, imgB, _added_num/10.


def Normalize(arr):
    return arr / np.sum(arr)

def ClearDir(path):
    if os.path.exists(path):
        print("Resetting the folder.....",path)
        shutil.rmtree(path=path)
    os.mkdir(path)

def GetPaths(type):
    if type == 0:
        return config.dir_Charts_train, \
               config.dir_subCharts_train, \
               config.path_groundTruth_train, \
               config.path_pair_groundTruth_train, \
               train_num
    if type == 1:
        return config.dir_Charts_val, \
               config.dir_subCharts_val, \
               config.path_groundTruth_val, \
               config.path_pair_groundTruth_val, \
               val_num
    if type == 2:
        return config.dir_Charts_test, \
               config.dir_subCharts_test, \
               config.path_groundTruth_test, \
               config.path_pair_groundTruth_test, \
               test_num

if __name__ == '__main__':
    MakeDir(config.base_dir)
    ClearDir(config.base_dir+"datasets_train")
    ClearDir(config.base_dir+"datasets_val")
    ClearDir(config.base_dir+"datasets_test")

    for i in range(3):

        dir_charts, dir_subCharts, path_groundTruth, path_pair_groundTruth, image_num = GetPaths(i)

        print("Generating:", dir_charts)

        ClearDir(dir_charts)

        file_gt = open(path_groundTruth, 'w')
        _count = 0
        while _count < image_num:

            i1, i2, label = GenerateOneChart(origin_num=original_num)
            if _count % 200 == 0:
                print(" Generated {}/{} charts".format(_count, image_num))

            cv2.imwrite(dir_charts + config.subChartName.format(_count, 0), i1 * 255)
            cv2.imwrite(dir_charts + config.subChartName.format(_count, 1), i2 * 255)
            file_gt.write("%.6f\n" % (label))  # ground truth
            _count += 1



def GenerateDatasetVGG(flag, image_num):

    print("Generating {} Dataset: {} ----------------".format(str.upper(flag),image_num))
    _images = []
    _labels = []

    _count = 0
    while _count < image_num:

        _, i2, label = GenerateOneChart(origin_num=original_num)

        if _count % 5000 == 0:
            print("   id {}".format(_count))

        image = np.concatenate((i2,i2,i2), axis=-1)

        _images.append(image)
        _labels.append(label)

        _count += 1

    _images = np.array(_images,dtype='float32')
    _labels = np.array(_labels,dtype='float32')

    # print(_images[0])
    # print(_images[0].shape)
    # print(_labels.shape)
    # print(_labels[0:1])
    # cv2.imshow('aaa', np.hstack([_images[i] for i in range(1)]))
    # cv2.waitKey(0)

    print('x_shape: ', _images.shape)
    print('y_shape: ', _labels.shape)

    return _images,_labels


def GenerateDatasetIRN(flag, image_num):
    print("Generating {} Dataset: {} ----------------".format(str.upper(flag), image_num))

    _images1 = []
    _images2 = []
    _labels = []

    _count = 0
    while _count < image_num:

        i1, i2, label = GenerateOneChart(origin_num=original_num)

        if _count % 5000 == 0:
            print("   id {}".format(_count))

        _images1.append(i1)
        _images2.append(i2)
        _labels.append(label)

        _count += 1

    _images1 = np.array(_images1,dtype='float32')
    _images2 = np.array(_images2,dtype='float32')
    _labels = np.array(_labels,dtype='float32')

    # print(_images1[0])
    # print(_images1.shape, _images2.shape)
    # print(_labels.shape)
    # print(_labels[0:1])
    # cv2.imshow('a', np.vstack([np.hstack([_images1[i] for i in range(1)]),
    #                            np.hstack([_images2[i] for i in range(1)])]))
    # cv2.waitKey(0)

    print('x1_shape: ', _images1.shape)
    print('x2_shape: ', _images2.shape)
    print('y_shape: ', _labels.shape)

    return _images1, _images2, _labels