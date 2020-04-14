import numpy as np
import os
import shutil

class Config:
    # The folder of datasets.
    base_dir = os.path.abspath('./dataset')+"/"

    # image configuration
    image_width = 100
    image_height = 100
    max_obj_num = 12      # The MAXIMUM object number (also the output-dim of networks.)

    # Do not change the following settings
    chartName = "img_{}.png"
    subChartName = "sub_{}_{}.png"

    # training set
    dir_Charts_train = base_dir + "datasets_train/charts/"
    dir_subCharts_train = base_dir + "datasets_train/subcharts/"
    path_groundTruth_train = base_dir + "datasets_train/ground_truth.txt"
    path_pair_groundTruth_train = base_dir + "datasets_train/pair_ground_truth.txt"

    # testing set
    dir_Charts_test = base_dir + "datasets_test/charts/"
    dir_subCharts_test = base_dir + "datasets_test/subcharts/"
    path_groundTruth_test = base_dir + "datasets_test/ground_truth.txt"
    path_pair_groundTruth_test = base_dir + "datasets_test/pair_ground_truth.txt"

    # validation set
    dir_Charts_val = base_dir + "datasets_val/charts/"
    dir_subCharts_val = base_dir + "datasets_val/subcharts/"
    path_groundTruth_val = base_dir + "datasets_val/ground_truth.txt"
    path_pair_groundTruth_val = base_dir + "datasets_val/pair_ground_truth.txt"

# Drawing the processing bar
def GetProcessBar(bid, batch_num,dot_num = 40):
    ratio = (bid+1)/batch_num
    delta = 40-(int(ratio*dot_num) + int((1-ratio)*dot_num))
    return '['+'='*int(ratio*dot_num) + '>'+"."*int((1-ratio)*dot_num+delta)+']'


def NormalizeNp(arr):
    max = np.sum(arr)
    return np.array(arr/max)


def GetAllFiles(dirpath):
    filepaths = []
    for root, dirs, files in os.walk(dirpath):

        for f in files:
            filepaths.append(os.path.join(root, f))
    return filepaths

def GetFileCountIn(dirpath):
    for root, dirs, files in os.walk(dirpath):
        return len(files)



def ReadLinesInFile(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    file.close()
    return lines

def ClearDir(path):
    if os.path.exists(path):
        print("Resetting the folder.....",path)
        shutil.rmtree(path=path)
    os.mkdir(path)

def MakeDir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def RemoveDir(path):
    if os.path.exists(path):
        os.remove(path)

