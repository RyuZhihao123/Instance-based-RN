import cv2
import os
import numpy as np
import shutil
from Configure import Config, MakeDir
import math

config = Config()

train_num = 60000
val_num   = 20000
test_num  = 20000

# The colors used in training set.
train_colors = np.array( [(0.541176,0.270588,0.341176),(0.486275,0.709804,0.88549),(0.564706,0.89549,0.490196),
          (0.894706,0.639216,0.364706),(0.505882,0.521569,0.883725),(0.891961,0.831373,0.337255)])

#  set the RANGE of object NUMBER for both training and testing set respectively
min_train_obj = 3
max_train_obj = config.max_obj_num    # 6
min_test_obj = 3
max_test_obj = config.max_obj_num     # 6

def Normalize(arr):
    return arr / np.sum(arr)

# generate a chart with number=num objects.
def GenerateOneBarChart(num, size = config.image_width, random_color = True):

    colors = train_colors if random_color==False else np.random.uniform(0.0, 0.9,size = (config.max_obj_num,3))
    if random_color == False:
        np.random.shuffle(colors)

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

        isRandomColor = False if i==0 else True

        for i in range(image_num):
            image, subImages, featureVector = GenerateOneBarChart(
                num=np.random.randint(min_num_obj, max_num_obj + 1), random_color=isRandomColor)

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
