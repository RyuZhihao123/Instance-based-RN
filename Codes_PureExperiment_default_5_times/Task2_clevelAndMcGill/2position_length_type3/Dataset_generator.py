import cv2
import os
import numpy as np
import shutil
from Configure import Config, MakeDir
import math
from ClevelAndMcGill.Figure4 import Figure4

config = Config()

EXPERIMENT = "Figure4.data_to_type3"

train_num = 600
val_num   = 200
test_num  = 200


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
        ClearDir(dir_subCharts)


        file_gt = open(path_groundTruth, 'w')
        _count = 0
        while _count < image_num:
            data, label = Figure4.generate_datapoint()
            try:
                i0, i1, i2 = eval(EXPERIMENT)(data)
            except:
                continue
            i = _count
            if i % 1000 == 0:
                print(" Num={}/{}".format(i,image_num))

            cv2.imwrite(dir_charts + config.chartName.format(i), i0 * 255)

            cv2.imwrite(dir_subCharts + config.subChartName.format(i, 0), i1 * 255)
            cv2.imwrite(dir_subCharts + config.subChartName.format(i, 1), i2 * 255)
            file_gt.write("%.6f\n" % (label))  # ground truth
            _count += 1


def GenerateDatasetVGG(flag, image_num):

    print("Generating {} Dataset: {} ----------------".format(str.upper(flag),image_num))
    _images = []
    _labels = []

    _count = 0
    while _count < image_num:

        data, label = Figure4.generate_datapoint()
        try:
            i0, i1, i2 = eval(EXPERIMENT)(data)
        except:
            continue

        if _count % 5000 == 0:
            print("   id {}".format(_count))

        image = i0[..., np.newaxis]
        image = np.concatenate((image,image,image), axis=-1)

        _images.append(image)
        _labels.append([label])

        _count += 1

    _images = np.array(_images,dtype='float32')
    _labels = np.array(_labels,dtype='float32')

    # print(_images[0])
    # print(_images[0].shape)
    # print(_labels.shape)
    # print(_labels[0:5])
    # cv2.imshow('aaa', np.hstack([_images[i] for i in range(5)]))
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

        data, label = Figure4.generate_datapoint()
        try:
            i0, i1, i2 = eval(EXPERIMENT)(data)
        except:
            continue

        if _count % 5000 == 0:
            print("   id {}".format(_count))

        i1 = i1[..., np.newaxis]
        i1 = np.concatenate((i1,i1,i1), axis=-1)

        i2 = i2[..., np.newaxis]
        i2 = np.concatenate((i2,i2,i2), axis=-1)

        _images1.append(i1)
        _images2.append(i2)
        _labels.append([label])

        _count += 1

    _images1 = np.array(_images1,dtype='float32')
    _images2 = np.array(_images2,dtype='float32')
    _labels = np.array(_labels,dtype='float32')

    # print(_images1[0])
    # print(_images1.shape, _images2.shape)
    # print(_labels.shape)
    # print(_labels[0:5])
    # cv2.imshow('a', np.vstack([np.hstack([_images1[i] for i in range(5)]),
    #                            np.hstack([_images2[i] for i in range(5)])]))
    # cv2.waitKey(0)

    print('x1_shape: ', _images1.shape)
    print('x2_shape: ', _images2.shape)
    print('y_shape: ', _labels.shape)

    return _images1, _images2, _labels