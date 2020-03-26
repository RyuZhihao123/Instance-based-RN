from Configure import Config, ClearDir
import numpy as np
import cv2

config = Config()

isRandomBaseNum = False
min_base_num = 1000
max_base_num = 1000

def AddNoise(_img):
    img = _img.copy()
    # 添加噪音
    noises = np.random.uniform(0, 0.05, (config.image_height,config.image_width,1))
    img += noises
    _min = img.min()
    _max = img.max()
    img -= _min
    img /= (_max - _min)

    img *= 255
    return img

def GenerateOneChart(origin_num = 10, max_added_num = 10):
    _pool = np.arange(0, config.image_width*config.image_height)  # 抽奖池!!!
    _added_num = np.random.randint(0, max_added_num+1)  # 新增加的点数目
    _pts = np.random.choice(_pool, origin_num + _added_num, False)  # 不放回抽奖

    imgA = np.ones(shape=(config.image_width, config.image_width, 1), dtype='float32')
    imgB = np.ones(shape=(config.image_width, config.image_width, 1), dtype='float32')
    for i in range(origin_num):
        x = _pts[i] % config.image_height
        y = _pts[i] // config.image_height
        imgA[x, y, 0] = 0.
        imgB[x, y, 0] = 0.

    for i in range(_added_num):
        x = _pts[i + origin_num] % config.image_height
        y = _pts[i + origin_num] // config.image_height
        imgB[x, y, 0] = 0.

    imgA = AddNoise(imgA)
    imgB = AddNoise(imgB)
    return imgA, imgB, _added_num/10.

if __name__ == '__main__':

    # 生成训练集
    ClearDir(config.dir_barCharts)
    file = open(config.path_groundTruth, 'w')


    for i in range(config.image_num):
        baseNum = min_base_num
        if isRandomBaseNum:
            baseNum = np.random.randint(min_base_num, max_base_num)
        imageA, imageB, gt = GenerateOneChart(baseNum)

        cv2.imwrite(config.dir_barCharts+config.barChartName.format(i, 0), imageA)
        cv2.imwrite(config.dir_barCharts+config.barChartName.format(i, 1), imageB)
        # print(config.dir_barCharts+config.barChartName.format(i,0))

        if i % 1000 == 0:
            print("已经加载: {}/{}".format(i,config.image_num))
        file.write(config.barChartName.format(i, 0) + '\t' + config.barChartName.format(i, 1) + '\t' + str(gt) + '\n')
    file.close()

    # 生成测试集
    ClearDir(config.dir_barCharts_test)
    file = open(config.path_groundTruth_test, 'w')

    for i in range(config.image_num_test):
        baseNum = min_base_num
        if isRandomBaseNum:
            baseNum = np.random.randint(min_base_num, max_base_num)

        imageA, imageB, gt = GenerateOneChart(baseNum)

        cv2.imwrite(config.dir_barCharts_test+config.barChartName.format(i, 0), imageA)
        cv2.imwrite(config.dir_barCharts_test+config.barChartName.format(i, 1), imageB)
        # print(config.dir_barCharts+config.barChartName.format(i,0))

        if i % 1000 == 0:
            print("已经加载: {}/{}".format(i,config.image_num_test))
        file.write(config.barChartName.format(i, 0) + '\t' + config.barChartName.format(i, 1) + '\t' + str(gt) + '\n')
    file.close()









