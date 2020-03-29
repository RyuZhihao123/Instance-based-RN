import numpy as np
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten,Input
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
import cv2
import os
import sklearn
from  openpyxl import Workbook
from sklearn.metrics import mean_squared_error
from Configure import Config, GetProcessBar,ReadLinesInFile, ClearDir, MakeDir, RemoveDir
import time, argparse, copy
from utils.non_local import non_local_block


""" some important configuration """
train_num = 60000             # image number.
val_num   = 20000
test_num  = 20000

m_epoch = 100                # epoch
m_batchSize = 32            # batch_size
m_print_loss_step = 15      # print once after how many iterations.



""" processing command line """
parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=0.0001, type = float)  # learning rate
parser.add_argument("--gpu", default='3')                  # gpu id
parser.add_argument("--savedir", default= 'IRN_p')         # saving path.
parser.add_argument("--backup", default=False, type=bool)   # whether to save weights after each epoch.
                                                           # (If True, it will cost lots of memories)
a = parser.parse_args()

m_optimizer = Adam(a.lr)
os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu

config = Config()

# create save folder.
MakeDir("./results/")
dir_results   = os.path.abspath(".") + "/results/{}/".format(a.savedir)    # backup path. 'results/IRN_p'




""" Load the dataset    """
max_img_id_train = -1
max_img_id_test  = -1
max_img_id_val   = -1

def Split_Image_Bar_ID(s):   # Extract the imageID and subID from the filename 'sub_{}_{}.png'
    _name = s.split(".")[-2].split("_")
    _imgID = int(_name[-2])
    _barID = int(_name[-1])
    return _imgID,_barID

def GetInformation(flag):
    if flag == 'train':
        return train_num, config.dir_subCharts_train, config.path_pair_groundTruth_train
    if flag == 'val':
        return val_num, config.dir_subCharts_val, config.path_pair_groundTruth_val
    if flag == 'test':
        return test_num, config.dir_subCharts_test, config.path_pair_groundTruth_test

def LoadDataset(flag = 'train'):    # must be 'train' or 'val' or 'test'

    global max_img_id_train, max_img_id_test, max_img_id_val

    __max_load_num, __dir_subcharts, __path_pair_groundTruth = GetInformation(flag)

    lines = ReadLinesInFile(__path_pair_groundTruth)

    old_curpath = os.path.abspath(".")
    os.chdir(__dir_subcharts)    # change current path to load images faster.

    images1 = []  # The first  input image size =（max_load_num, h, w, 3)
    images2 = []  # The second input image size =（max_load_num, h, w, 3)
    labels = []   # groundTruth  size = (max_load_num, 1)
    count = 0

    __max_img_id = -1
    data_infos = [[] for i in range(__max_load_num)]
    for line in lines:
        elements = line.split()
        if len(elements) < 3:
            continue

        images1.append(cv2.imread(elements[0]))
        images2.append(cv2.imread(elements[1]))
        labels.append(float(elements[2]))

        _img_id,_ = Split_Image_Bar_ID(elements[0])
        data_infos[_img_id].append(labels[-1])

        __max_img_id = max(__max_img_id,_img_id)

        if count % 10000 == 0:
            print("Loaded: {}/{}".format(count,__max_load_num))

        count += 1
        if count >= __max_load_num:
            break


    images1 = np.array(images1, dtype= 'float32') /255.0
    images2 = np.array(images2, dtype= 'float32') /255.0
    labels = np.array(labels , dtype= 'float32')

    if flag == 'train': max_img_id_train = __max_img_id
    if flag == 'test' : max_img_id_test = __max_img_id
    if flag == 'val' : max_img_id_val = __max_img_id

    os.chdir(old_curpath)
    #print("cur path", os.path.abspath(os.curdir))

    print("---------------------------")
    print("[Process] x1: ", images1.shape)
    print("[Process] x2: ", images2.shape)
    print("[Process] y: ", labels.shape, config.max_obj_num)

    return images1, images2, labels, data_infos


# Level1 module is to extract the individual features from one instance.
# Level1 has two NON-LOCAL block.
def Level1_Module():
    input = Input(shape=(config.image_height, config.image_width, 3))
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = non_local_block(x)    # non local block
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = non_local_block(x)    # non local block
    return Model(inputs=input, outputs=x)

# IRN_p module is to estimate the ratio between one pair.
def Build_IRN_p_Network():
    inputA = Input(shape=(config.image_height, config.image_width, 3))
    inputB = Input(shape=(config.image_height, config.image_width, 3))

    level1 =Level1_Module()    # build a level1 module
    a = level1(inputA)         # extract two individual features.
    b = level1(inputB)

    combined = keras.layers.concatenate([a, b])   # concatenate them.
    z = Conv2D(64, (3, 3), activation='relu',padding='same')(combined)
    z = Conv2D(64, (3, 3), activation='relu', padding='same')(z)
    z = non_local_block(z)        # non local block
    #
    z = Flatten()(z)
    z = Dense(256, activation="relu")(z)
    z = Dropout(0.5)(z)
    z = Dense(1, activation="linear")(z)    # output the ratio of this pair.

    return Model(inputs=[inputA, inputB], outputs=z)


""" normalize the pairwise ratios into the final multi-dim ratios"""
""" Since IRN_p directly output a scalar, whearas what we finally need is the normalized multi-dim ratio vectors"""
""" I think.. perhaps this part of code is hard to read.....(*^*)    """
""" So you can directly move to the 'main()' function in the following.  """
def transfer_predict_vec_to_6dim_vec(line):
    res = [0.0 for i in range(config.max_obj_num)]
    if len(line) == 0:
        return res

    ids = [len(line)-i for i in range(0,len(line))] # len=3 (i=0,1,2) id=(3,2,1)
    for i in ids:    # i = 3,2,1  i-1 = (2,1,0)
        if i==ids[0]:
            res[i] = 1.0
        res[i-1] = line[i-1]*res[i]

    return np.array(res).reshape((-1)).astype('float32')

def Get_transfered_datasets(data_infos, predictY, flag = 'train'):
    predict_infos = copy.deepcopy(data_infos)
    count = 0
    for i in range(len(data_infos)):
        for j in range(len(data_infos[i])):
            predict_infos[i][j] = predictY[count]
            count += 1

    if flag == 'train':
        __max_img_id = max_img_id_train
    if flag == 'val':
        __max_img_id = max_img_id_val
    if flag == 'test':
        __max_img_id = max_img_id_test

    gts = []
    prs = []
    for i in range(__max_img_id - 1):
        vec_gt = transfer_predict_vec_to_6dim_vec(data_infos[i])
        vec_predict = transfer_predict_vec_to_6dim_vec(predict_infos[i])

        max1 = np.max(vec_gt)
        max2 = np.max(vec_predict)
        vec_gt /= max1
        vec_predict /= max2
        gts.append(vec_gt)
        prs.append(vec_predict)

    return np.array(gts), np.array(prs)

def GetMSEloss(data_infos, predictY, flag = 'train'):
    gts, prs = Get_transfered_datasets(data_infos,predictY,flag)

    return mean_squared_error(gts, prs)

def SavePredictedResult(x1,x2,y,data_infos, model, flag = 'train'):
    predictY = model.predict(x=[x1,x2], batch_size=1)
    gts, prs = Get_transfered_datasets(data_infos, predictY,flag)

    predictFile = open(dir_results + flag + "_predict_results.txt",'w')
    predictFile_norm = open(dir_results + flag +"_predict_results_norm.txt",'w')
    for i in range(x1.shape[0]):
        predictFile.write(str(y[i])+'\t'+str(predictY[i][0]) + '\n')
    for i in range(gts.shape[0]):

        for k in range(gts.shape[1]):
            predictFile_norm.write(str(gts[i,k]) + "\t")
        predictFile_norm.write("\n")
        for k in range(prs.shape[1]):
            predictFile_norm.write(str(prs[i,k]) + "\t")
        predictFile_norm.write("\n")
    predictFile.close()

    MLAE = np.log2(sklearn.metrics.mean_absolute_error( prs * 100, gts * 100) + .125)

    return MLAE




if __name__ == '__main__':

    ClearDir(dir_results)
    ClearDir(dir_results+"backup")

    x1_train, x2_train, y_train, data_infos_train = LoadDataset('train')
    x1_val, x2_val, y_val, data_infos_val = LoadDataset('val')
    x1_test, x2_test, y_test, data_infos_test = LoadDataset('test')

    ##  normalize to [-0.5, 0.5]
    x1_train -= .5
    x2_train -= .5
    x1_val -= .5
    x2_val -= .5
    x1_test -= .5
    x2_test -= .5

    print("----------Build Network----------------")
    model = Build_IRN_p_Network()
    model.compile(optimizer=m_optimizer, loss='mse')

    print("----------Training-------------------")
    history_batch = []
    history_iter = []
    batch_amount = train_num // m_batchSize

    # information of the best model on validation set.
    best_val_loss = 99999.99999
    best_model_name = "xxxxx"
    best_train_loss = 99999.99999

    for iter in range(m_epoch):

        # shuffle the training set
        index = [i for i in range(train_num)]
        np.random.shuffle(index)

        for bid in range(batch_amount):

            # using the shuffle index...
            x1_batch = x1_train[index[bid * m_batchSize : (bid + 1) * m_batchSize]]
            x2_batch = x2_train[index[bid * m_batchSize : (bid + 1) * m_batchSize]]
            y_batch = y_train[index[bid * m_batchSize : (bid + 1) * m_batchSize]]

            model.train_on_batch(x=[x1_batch,x2_batch], y= y_batch)   # training on batch

            if bid % m_print_loss_step == 0:
                logs = model.evaluate(x=[x1_batch,x2_batch], y= y_batch, verbose=0, batch_size=m_batchSize)
                print("iter({}/{}) Batch({}/{}) {} : Pair_loss={}".format(iter, m_epoch, bid, batch_amount,
                                                                     GetProcessBar(bid, batch_amount), logs))
                history_batch.append([iter, bid, logs])

        # one epoch is done. Do some information collections.
        epoch_loss_pair_train = model.evaluate(x=[x1_train, x2_train], y=y_train, verbose=0, batch_size=m_batchSize)
        pred_y_train = model.predict(x=[x1_train, x2_train], verbose=0, batch_size=m_batchSize)
        epoch_loss_norm_train = GetMSEloss(data_infos_train, pred_y_train,'train')

        pred_y_val = model.predict(x=[x1_val, x2_val], verbose=0, batch_size=m_batchSize)
        epoch_loss_norm_val = GetMSEloss(data_infos_val, pred_y_val, 'val')
        print("----- epoch({}/{}) pair_loss={} norm_train_loss={} norm_val_loss = {}".format(iter, m_epoch, epoch_loss_pair_train, epoch_loss_norm_train, epoch_loss_norm_val))
        history_iter.append([iter, epoch_loss_pair_train, epoch_loss_norm_train, epoch_loss_norm_val])

        if a.backup == True:   # whether to save the weight of each epoch
            model.save_weights(dir_results+"backup/model_{}_{}.h5".format(iter, epoch_loss_norm_val))

        if epoch_loss_norm_val < best_val_loss:  # save the best model on Validation set.
            RemoveDir(best_model_name)
            best_model_name = dir_results + "model_irnp_{}.h5".format(epoch_loss_norm_val)
            model.save_weights(best_model_name)
            best_val_loss = epoch_loss_norm_val
            best_train_loss = epoch_loss_norm_train


    # test on the testing set.
    model.load_weights(best_model_name)   # using the best model

    test_pair_loss = model.evaluate(x=[x1_test, x2_test], y=y_test, verbose=0, batch_size=m_batchSize)
    pred_y_test = model.predict(x=[x1_test, x2_test], verbose=0, batch_size=m_batchSize)
    test_norm_loss = GetMSEloss(data_infos_test, pred_y_test, 'test')


    # Save the predicted results and ground truth.
    MLAE_train = SavePredictedResult(x1_train,x2_train,y_train,data_infos_train,model,'train')
    MLAE_val = SavePredictedResult(x1_val,x2_val,y_val,data_infos_val,model,'val')
    MLAE_test = SavePredictedResult(x1_test,x2_test,y_test,data_infos_test,model,'test')


    # save the training information.
    wb = Workbook()
    ws1 = wb.active
    ws2 = wb.create_sheet("iter loss")

    ws1.append(["Iter ID", "Batch ID", "Pair MSE Loss"])
    ws2.append(["Iter ID", "Pair Loss", "Norm Train Loss", "Norm Test Loss"])
    for i in range(len(history_batch)):
        ws1.append(history_batch[i])
    for i in range(len(history_iter)):
        ws2.append(history_iter[i])
    ws2.append(["Train MSE", best_train_loss])
    ws2.append(["Val MSE", best_val_loss])
    ws2.append(['Test MSE',test_norm_loss])
    ws2.append(["Train MLAE", MLAE_train])
    ws2.append(["val MLAE", MLAE_val])
    ws2.append(["Test MLAE", MLAE_test])

    wb.save(dir_results + "train_info.xlsx")


    print("Training MSE:", best_train_loss)
    print("Validat. MSE:", best_val_loss)
    print("Testing MSE:", test_norm_loss)
    print("Training MLAE:", MLAE_train)
    print("Validat. MSE:", MLAE_val)
    print("Testing MLAE:", MLAE_test)
