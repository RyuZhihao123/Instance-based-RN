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



#  set the RANGE of line width for training and testing set respectively
thickness_train = [1]
thickness_test  = [2, 3]


#  set the RANGE of object NUMBER for both training and testing set respectively
min_train_obj = 3
max_train_obj = config.max_obj_num
min_test_obj = 3
max_test_obj = config.max_obj_num


def Normalize(arr):
    return arr / np.sum(arr)

# generate a chart with number=num objects.
def GenerateOnePieChart(num, size = 100, thickness = 1):
    r = np.random.randint(25,45)        # Radii of the pie. (random)
    thickness = thickness  # line thickness.   (random)

    center = (int(size/2),int(size/2))  #
    image = np.ones(shape=(size, size, 1))
    subImages = [np.ones(shape=(size,size,1)) for i in range(config.max_obj_num)]
    angles = Normalize(np.random.randint(10,60,size=(num)))

    start_angle = 90 - np.random.randint(0,360*angles[0])/2.0
    _cur_start_angle = start_angle
    cv2.circle(image,center,r,0,thickness)

    for i in range(num):
        _cur_end_angle = _cur_start_angle + angles[i] * 360.0
        cv2.line(image,center, (50-int(r*math.sin(math.pi*_cur_start_angle/180.0)),
                                50-int(r*math.cos(math.pi*_cur_start_angle/180.0))),0,thickness)

        cv2.line(subImages[i], center, (50-int(r*math.sin(math.pi*_cur_start_angle/180.0)),
                                        50-int(r*math.cos(math.pi*_cur_start_angle/180.0))),0,thickness)
        cv2.line(subImages[i], center, (50 - int(r * math.sin(math.pi * _cur_end_angle / 180.0)),
                                        50 - int(r * math.cos(math.pi * _cur_end_angle / 180.0))), 0, thickness)
        cv2.ellipse(subImages[i],center,(r,r),270,-_cur_start_angle,-_cur_end_angle,0,thickness)
        _cur_start_angle = _cur_end_angle

    # add noise
    noises = np.random.uniform(0, 0.05, (size, size,1))
    image = image + noises
    _min = image.min()
    _max = image.max()
    image -= _min
    image /= (_max - _min)

    for i in range(len(subImages)):
        noises = np.random.uniform(0, 0.05, (size, size, 1))
        subImages[i] = subImages[i] + noises
        _min = subImages[i].min() if i < num else 0   # since 'i>=num' are empty images.
        _max = subImages[i].max()
        subImages[i] -= _min
        subImages[i] /= (_max - _min)
    #

    max_height = max(angles)

    for i in range(len(angles)):
        angles[i] /= max_height
    return image, subImages, angles

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

        thickness_set = thickness_train if i==0 else thickness_test

        for i in range(image_num):
            image, subImages, featureVector = GenerateOnePieChart(
                num=np.random.randint(min_num_obj, max_num_obj + 1), thickness=thickness_set[np.random.randint(len(thickness_set))])

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
