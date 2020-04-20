import cv2
import os
import numpy as np
import shutil
from Configure import Config, MakeDir
import math
from ClevelAndMcGill.Figure3 import Figure3

config = Config()

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
        file_pair_gt = open(path_pair_groundTruth,'w')


        for i in range(image_num):

            data, labels = Figure3.generate_datapoint()

            image, subImages = Figure3.data_to_barchart(data)

            featureVector = np.roll(labels, np.where(data == np.max(data))[0])

            if i % 500 == 0:
                print("   id {} (obj_num = {})".format(i, len(subImages)))

            cv2.imwrite(dir_charts + config.chartName.format(i), image * 255)
            for t in range(len(subImages)):
                cv2.imwrite(dir_subCharts + config.subChartName.format(i, t), subImages[t] * 255)

            for t in range(len(featureVector)):
                file_gt.write("%.6f\t" % (featureVector[t]))
            file_gt.write("\n")

            for t in range(len(featureVector) - 1):
                file_pair_gt.write("{} {} {}\n".format(config.subChartName.format(i, t),
                                                       config.subChartName.format(i, t + 1),
                                                       featureVector[t+1] / featureVector[t]))
        file_gt.close()
        file_pair_gt.close()

def GenerateDatasetVGG(flag, image_num):

    print("Generating {} Dataset: {} ----------------".format(str.upper(flag),image_num))
    _images = []
    _labels = []

    for i in range(image_num):

        data, labels = Figure3.generate_datapoint()
        image, subImages = Figure3.data_to_barchart(data)
        featureVector = np.roll(labels, np.where(data == np.max(data))[0])


        if i % 5000 == 0:
            print("   id {} (obj_num = {})".format(i, featureVector.shape[0]))

        image = image[..., np.newaxis]
        image = np.concatenate((image,image,image), axis=-1)

        _images.append(image)
        _labels.append(featureVector)

    _images = np.array(_images,dtype='float32')
    _labels = np.array(_labels,dtype='float32')

    # print(_images[0])
    # print(_images[0].shape)
    # print(_labels[0])
    # cv2.imshow('aaa', _images[0])
    # cv2.waitKey(0)

    print('x_shape: ', _images.shape)
    print('y_shape: ', _labels.shape)

    return _images,_labels


def GenerateDatasetIRNm(flag, image_num):
    print("Generating {} Dataset: {} ----------------".format(str.upper(flag), image_num))

    _images = np.ones((config.max_obj_num, image_num, config.image_height, config.image_width, 1), dtype='float32')
    _labels = []


    for i in range(image_num):

        data, labels = Figure3.generate_datapoint()
        image, subimages = Figure3.data_to_barchart(data)
        featureVector = np.roll(labels, np.where(data == np.max(data))[0])

        if i % 5000 == 0:
            print("   id {} (obj_num = {})".format(i, featureVector.shape[0]))

        subimages = [subimages[t][...,np.newaxis] for t in range(config.max_obj_num)]
        # subimages = [np.concatenate((img, img, img), axis=-1) for img in subimages]

        for t in range(config.max_obj_num):
            _images[t][i] = subimages[t]

        _labels.append(featureVector)

    _labels = np.array(_labels, dtype='float32')

    # print(_images[0][2])
    # print(_labels[2])
    # cv2.imshow('aaa', np.hstack([_images[t][2] for t in range(config.max_obj_num)]))
    # cv2.waitKey(0)

    print('x_shape: ', _images.shape)
    print('y_shape: ', _labels.shape)

    return _images, _labels