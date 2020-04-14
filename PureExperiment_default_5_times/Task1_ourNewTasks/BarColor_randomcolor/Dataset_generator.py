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

#  set the RANGE of object NUMBER for both training and testing set respectively
min_train_obj = 3
max_train_obj = config.max_obj_num    # 6
min_test_obj = 3
max_test_obj = config.max_obj_num     # 6

def Normalize(arr):
    return arr / np.sum(arr)

# generate a chart with number=num objects.
def GenerateOneBarChart(num, size = config.image_width):

    colors = np.random.uniform(0.0, 0.9,size = (config.max_obj_num,3))

    image = np.ones(shape=(size, size, 3))
    subImages = [np.ones(shape=(size,size,3)) for i in range(config.max_obj_num)]
    heights = np.random.randint(10,80,size=(num))

    barWidth = int( (size-3*(num+1)-3)//num * (np.random.randint(50,100)/100.0) )
    barWidth = max(barWidth, 4)
    spaceWidth = (size-(barWidth)*num)//(num+1)

    sx = (size - barWidth*num - spaceWidth*(num-1))//2
    for i in range(num):

        sy = size - 1
        ex = sx + barWidth
        ey = sy - heights[i]

        cv2.rectangle(image,(sx,sy),(ex,ey),colors[i],-1)
        cv2.rectangle(subImages[i],(sx,sy),(ex,ey),colors[i],-1)
        sx = ex + spaceWidth

    # add noise
    noises = np.random.uniform(0, 0.05, (size, size,3))
    image = image + noises
    _min = 0.0
    _max = image.max()
    image -= _min
    image /= (_max - _min)

    for i in range(len(subImages)):
        noises = np.random.uniform(0, 0.05, (size, size, 3))
        subImages[i] = subImages[i] + noises
        _min = 0.0
        _max = subImages[i].max()
        subImages[i] -= _min
        subImages[i] /= (_max - _min)
    #
    heights = heights.astype('float32')
    max_height = max(heights)

    for i in range(len(heights)):
        heights[i] /= max_height
    return image, subImages, heights

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

        min_num_obj = min_train_obj if i == 0 else min_test_obj
        max_num_obj = max_train_obj if i == 0 else max_test_obj

        for i in range(image_num):
            image, subImages, featureVector = GenerateOneBarChart(
                num=np.random.randint(min_num_obj, max_num_obj + 1))

            if i % 200 == 0:
                print("   id {} (obj_num = {})".format(i, len(subImages)))

            cv2.imwrite(dir_charts + config.chartName.format(i), image * 255)
            for t in range(len(subImages)):
                cv2.imwrite(dir_subCharts + config.subChartName.format(i, t), subImages[t] * 255)

            for t in range(len(featureVector)):
                file_gt.write("%.6f\t" % (featureVector[t]))
            for t in range(config.max_obj_num - len(featureVector)):
                file_gt.write("0.00\t")
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

    min_num_obj = min_train_obj if flag == 'train' else min_test_obj
    max_num_obj = max_train_obj if flag == 'train' else max_test_obj


    for i in range(image_num):

        image, _, featureVector = GenerateOneBarChart(
                num=np.random.randint(min_num_obj, max_num_obj + 1))
        featureVector = np.array(featureVector)

        if i % 5000 == 0:
            print("   id {} (obj_num = {})".format(i, featureVector.shape[0]))

        label = np.zeros(config.max_obj_num,dtype='float32')
        label[:len(featureVector)] = featureVector

        _images.append(image)
        _labels.append(label)

    _images = np.array(_images,dtype='float32')
    _labels = np.array(_labels,dtype='float32')

    # print(_images[0])
    # print(_labels[0])
    # cv2.imshow('aaa', _images[0])
    # cv2.waitKey(0)

    print('x_shape: ', _images.shape)
    print('y_shape: ', _labels.shape)

    return _images,_labels


def GenerateDatasetIRNm(flag, image_num):
    print("Generating {} Dataset: {} ----------------".format(str.upper(flag), image_num))

    _images = np.ones((config.max_obj_num, image_num, config.image_height, config.image_width, 3), dtype='float32')
    _labels = []

    min_num_obj = min_train_obj if flag == 'train' else min_test_obj
    max_num_obj = max_train_obj if flag == 'train' else max_test_obj


    for i in range(image_num):

        _, subimages, featureVector = GenerateOneBarChart(
                num=np.random.randint(min_num_obj, max_num_obj + 1))
        featureVector = np.array(featureVector)

        if i % 5000 == 0:
            print("   id {} (obj_num = {})".format(i, featureVector.shape[0]))

        for t in range(config.max_obj_num):
            _images[t][i] = subimages[t]
        label = np.zeros(config.max_obj_num, dtype='float32')
        label[:len(featureVector)] = featureVector
        _labels.append(label)

    _labels = np.array(_labels, dtype='float32')

    # print(_images[0][2])
    # print(_labels[2])
    # cv2.imshow('aaa', np.hstack([_images[t][2] for t in range(config.max_obj_num)]))
    # cv2.waitKey(0)

    print('x_shape: ', _images.shape)
    print('y_shape: ', _labels.shape)

    return _images, _labels