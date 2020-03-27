# Instance-based-RN

We designed the **IRN_m network** for those multi-objects ratio tasks, and **IRN_p network** for pair-ratio estimation tasks. 

Since I didn't tidy up my codes before, if you meet any bug, please tell me (liuzh96@outlook.com)  thanks (^,^).

## How to use.

### 1. Environments:

* **Recommended Env.** Python3.6, Tensorflow1.14.0, Keras 2.2.4.

* **Neccessary libaries:** numpy, opencv-python, argparse, scikit-learn,openpyxl

### 2. Example:

Here, I'll take **Task1.1 PieNumber** [(codes)](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task1_ourNewTasks/PieNumber)  as an example to show how to use the codes.

* **First, create your own datasets.** The following command will create a `./datasets` folder in current path.

```
python3.6 Dataset_generator.py
```

* **Second, run a network.** This command will train and test the `IRN_m` network. When training step is finished, the training informations(MLAE and MSE etc.), best model(on val sets) and the predicted results can be obtained in folder `./results/IRN_m`.

```
python3.6 Net_IRNm.py --gpu 2      # GPU ID
```

**Details:**

(1) Train/val/test sets contains 60000/20000/20000 charts respectively. We use Adam optimizer (lr =0.0001). 

(2) During training, we shuffle the datasets for each epoch, and save the best model which gets the lowest mse loss on `validation set`.

(3) Noises were directly added during dataset generalization. And `Position-length` and `Point cloud` got the most different values from the obvious results when using Adam optimizer.

**Notice:** 

In rare cases (very low probability), if the loss of some network doesn't decrease obviously after the first epoch, please kill it and restart the program, because Adam could usually get a way low loss value even during the first epoch.

## Experiments1: Our new tasks. 
(These experiments focus on verifying the generalization ability of networks.)

> Notice: Under normal circumstances, if a network can work well on both training and testing set, we choose to save the best model on their `validation sets`. However, for pieNumber and pieLineWidth, whose training and testing sets have different distribution, VGG and RN only works well on training set, so this time we save their best model on training sets. Because if a network can't work well on testing set, its validation loss would tend to jump up and down, so that perhaps the network haven't yet converged on training set but it gives the lowest validation loss (For example, obatin the lowest validation loss just at the begining).

### Task1.1: PieNumber.

[Codes](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task1_ourNewTasks/PieNumber) 

* `The range of object number` are different between training and testing sets. By default, the pie charts in training sets contain 3 to 6 pie sectors, while those in testing sets contain 7 to 9 pie sectors. For VGG, RN and IRNm, all the outputs are 9-dim vector.

![Example Image](https://github.com/RyuZhihao123/Instance-based-RN/blob/master/image/PieNumber.png)

![Example Image](https://github.com/RyuZhihao123/Instance-based-RN/blob/master/image/Sample.png)

* **We found only our IRN_m and IRN_p can get a good result on testing set.** 

| MSE(MLAE) | VGG | VGG_seg | RN | IRN_p| IRN_m (!!!) |
| ----- | ----- | ----- | ----- | -----| ----- |
| Train set | 0.00023(0.15) | 0.00025(0.17) | 0.00287(1.69) | 0.00015(-0.56) | **0.00010(-0.57)** |
| Test set | 0.13811(4.59) | 0.15428(4.81) | 0.16186(4.85) | 0.00087(0.97) | **0.00058(0.81)** |


### Task1.2: PieLineWidth.

[Codes](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task1_ourNewTasks/PieLineWidth) 

* `The line width` are different between training and testing sets in this task. By default, the line width of the piechart in training sets is 1, while the width of those in testing sets is 2 or 3.

![Example Image](https://github.com/RyuZhihao123/Instance-based-RN/blob/master/image/PieLineWidth.png)

* **PieLineWidth is unlike PieNumber whose training and testing sets both have different line width (The appearence domain are same). In PieLineWidth, even segmented objects, look different between the training sets and testing set. However, the result is surprising!! We found that both IRN_p and IRN_m can get a good result in testing set, and IRN_m performs better. That means if we segmeneted objects in advance and directly using CNN to extract their individual features, it does make great effect.** 

| MSE(MLAE) | VGG | RN | IRN_p| IRN_m (!!!) |
| ----- | -----  | ----- | -----| ----- |
| Train set | 0.00038(0.73)  | 0.00441(2.27) | -(-) | **-(-)** |
| Test set | 0.09503(4.53)  | 0.08425(4.32) | -(-) | **-(-)** |


### Task1.3: Pie3_12.

[Codes](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task1_ourNewTasks/Pie3_12) 

* In both training and testing sets, each pie chart contains 3 to 12 pie sectors. This task is to test the performance when the maximun object number is large and the number changes greatly. I think 12 is large enough since if we use a larger number, the chart would looks messy.

![Example Image](https://github.com/RyuZhihao123/Instance-based-RN/blob/master/image/Pie3_12.png)

## Experiments2: ClevelandMcGill
(The experiments that are same as Daniel's paper.)

### Task2.1: Position-Angle.

Codes: 
[[Bar charts]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task2_cleverlAndMcGill/1position_angle_Bar) 
[[Pie charts]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task2_cleverlAndMcGill/1position_angle_Pie)

![Example Image](https://github.com/RyuZhihao123/Instance-based-RN/blob/master/image/Position_angle.png)

### Task2.2: Position-Length.

Codes: 
[[MULTI]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task2_cleverlAndMcGill/2position_length_multi) 
[[Type1]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task2_cleverlAndMcGill/2position_length_type1)
[[Type2]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task2_cleverlAndMcGill/2position_length_type2)
[[Type3]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task2_cleverlAndMcGill/2position_length_type3)
[[Type4]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task2_cleverlAndMcGill/2position_length_type4)
[[Type5]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task2_cleverlAndMcGill/2position_length_type5)

![Example Image](https://github.com/RyuZhihao123/Instance-based-RN/blob/master/image/Position_multi.png)

### Task2.3: Point-Cloud.

Codes: 
[[Num10]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task2_cleverlAndMcGill/3point_cloud_10) 
[[Num100]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task2_cleverlAndMcGill/3point_cloud_100)
[[Num1000]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task2_cleverlAndMcGill/4point_cloud_1000)


## Experiments3: Supplement 

### Task3.1: The effect of Non_local_block.
### Task3.2: Can orignal RN be improved by changing its structure?
### Task3.3: How would RN perform when the objects are segmented directly rather than extracted by CNN.


