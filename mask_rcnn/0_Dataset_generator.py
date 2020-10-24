import cv2
import os
import numpy as np
import shutil
from Configure import Config, MakeDir
import math
import shutil
from WRITE_JSON import WRITE_JSON

config = Config()

train_num = 3


# The colors used in training set.
train_colors = np.array( [(0.541176,0.270588,0.341176),(0.486275,0.709804,0.88549),(0.564706,0.89549,0.490196),
          (0.894706,0.639216,0.364706),(0.505882,0.521569,0.883725),(0.891961,0.831373,0.337255)])

label_colors = np.array([(0.0, 0.0, 0.5), (0.0, 0.5, 0.0)])

#  set the RANGE of object NUMBER for both training and testing set respectively
min_test_obj = 3
max_test_obj = config.max_obj_num     # 12

def Normalize(arr):
    return arr / np.sum(arr)

# generate a chart with number=num objects.
def GenerateOneBarChart(num, size = config.image_width, random_color = True):

    colors = train_colors if random_color==False else np.random.uniform(0.0, 0.9,size = (config.max_obj_num,3))
    if random_color == False:
        np.random.shuffle(colors)

    image = np.ones(shape=(size, size, 3))
    mask_image = np.zeros(shape=(size,size,3))
    # subImages = [np.ones(shape=(size,size,3)) for i in range(config.max_obj_num)]
    heights = np.random.randint(10,80,size=(num))

    barWidth = int( (size-3*(num+1)-3)//num * (np.random.randint(50,100)/100.0) )
    barWidth = max(barWidth, 4)
    spaceWidth = (size-(barWidth)*num)//(num+1)

    points = []

    sx = (size - barWidth*num - spaceWidth*(num-1))//2
    for i in range(num):

        sy = size - 1
        ex = sx + barWidth
        ey = sy - heights[i]

        cv2.rectangle(image,(sx,sy),(ex,ey),colors[i],-1)
        cv2.rectangle(mask_image,(sx,sy),(ex,ey),label_colors[0],-1)
        points.append([[float(sx),float(sy)],[float(sx),float(ey)],[float(ex),float(ey)],[float(ex),float(sy)]])
        sx = ex + spaceWidth

    # add noise
    noises = np.random.uniform(0, 0.05, (size, size,3))
    image = image + noises
    _min = 0.0
    _max = image.max()
    image -= _min
    image /= (_max - _min)

    #
    heights = heights.astype('float32')
    max_height = max(heights)

    for i in range(len(heights)):
        heights[i] /= max_height
    return image, mask_image, heights, points

def ClearDir(path):
    if os.path.exists(path):
        print("Resetting the folder.....",path)
        shutil.rmtree(path=path)
    os.mkdir(path)


if __name__ == '__main__':
    ClearDir(config.base_dir)


    MakeDir(config.base_dir + "pic/")
    MakeDir(config.base_dir + "json/")
    MakeDir(config.base_dir + "cv2_mask/")
    MakeDir(config.base_dir + "labelme_json/")

        


    isRandomColor = True

    for i in range(train_num):

        image, _, featureVector, points = GenerateOneBarChart(
            num=np.random.randint(min_test_obj, max_test_obj + 1), random_color=isRandomColor)

        # print(points)

        if i % 200 == 0:
            print("   id {} (obj_num = {})".format(i, len(featureVector)))

        image = (image*255).astype(np.uint8)

        cv2.imwrite(config.base_dir + "pic/{}.png".format(i), image)

        WRITE_JSON.SAVE_TO_FILE(config.base_dir + "json/{}.json".format(i), image, i, points,"bar")


