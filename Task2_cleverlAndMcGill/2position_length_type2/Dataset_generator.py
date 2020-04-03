import cv2
import os
import numpy as np
import shutil
from Configure import Config, MakeDir
import math
from ClevelAndMcGill.Figure4 import Figure4

config = Config()

EXPERIMENT = "Figure4.data_to_type2"

train_num = 60000
val_num   = 20000
test_num  = 20000


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
