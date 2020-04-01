# Instance-based-RN

We designed a **IRN_m network** for multi-objects ratio tasks, and **IRN_p network** for pair-ratio estimation tasks. 

if you meet any problem or bug, please tell me (liuzh96@outlook.com)  thanks (^,^).

## Usage.

### 1. Environments:

* **Recommended Env.** Python3.6, Tensorflow1.14.0, Keras 2.2.4.

* **Neccessary libaries:** numpy, opencv-python, argparse, scikit-learn,openpyxl

### 2. Example:

I'll take **Task1.1 PieNumber** [(codes)](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task1_ourNewTasks/PieNumber)  as an example to show how to use the codes.

* **First, create your own datasets.** The following command will create a `./datasets` folder in current path.

```
python3.6 Dataset_generator.py
```

* **Second, run a network.** This command will train and test the `IRN_m` network. When training step is finished, the training informations(MLAE, MSE and histories etc.), best model(on val sets) and the predicted results can be obtained in folder `./results/IRN_m`.

```
python3.6 Net_IRNm.py --gpu 2 (--savedir abcdef)      # GPU ID ( target directory: ./results/abcdef)
```

<div align=center><img width="650" src="https://github.com/RyuZhihao123/Instance-based-RN/blob/master/image/OutputFiles.png"/></div>

### 3. Our network structure:

To make the generalization ability of our network more powerful, we redesgin the IRN_m network, as shown in the following figure. It makes great improvements on the conditions that (1) the training and testing set are different, e.g., `Task1.1 PieNumber`, or (2) the object number is large, for example `task1.3 Pie3_12`.

<div align=center><img width="900" src="https://github.com/RyuZhihao123/Instance-based-RN/blob/master/image/IRN_structure.png"/></div>

**Details:**

(1) Train/val/test sets contains 60000/20000/20000 charts respectively. And We use Adam optimizer (lr=0.0001) to train the network. 

(2) During training, we first shuffle the datasets before each epoch, and save the `best model` which can get the lowest MSE loss on `validation set`.

(3) Noises were directly added during dataset generation. And `Position-length` and `Point cloud` got the most different values from the obvious results in Daniel's paper when I use Adam optimizer.

## Experiments1: Our new tasks. 

(These experiments focus on verifying the generalization ability of networks.)

Note that: **For most tasks, we use the best model on validation set to compute its final MSE and MLAE etc**. However, VGG, VGG_seg and RN don't have strong generalization abilibity so that they can not deal with the validation/testing sets in `PieNumber` and `PieLineWidth`. It may happen that the network has obtained the lowest loss on validation sets but it still doesn't converage on training set. Therefor, to evaluate them better, **only for VGG and RN in PieNumber and PieLineWidth tasks**, we use the best model on training set instead of on validation set to compute the MSE on training set, while we still use the best model on validation set to compute its MSE on testing sets.

### Task1.1: PieNumber.

[[Codes]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task1_ourNewTasks/PieNumber) 

* `The range of object number` are different between training and testing sets. By default, the pie charts in training sets contain 3 to 6 pie sectors, while those in testing sets contain 7 to 9 pie sectors. For VGG, RN and IRNm, all the outputs are `9-dim vector`.

<div align=center><img width="350" src="https://github.com/RyuZhihao123/Instance-based-RN/blob/master/image/PieNumber.png"/></div>

<div align=center><img width="350" src="https://github.com/RyuZhihao123/Instance-based-RN/blob/master/image/Sample.png"/></div>

* **Only our IRN_m and IRN_p can get a good result on testing set. (1) Our network can deal with the condition that training and testing sets have different object number. (2) Our network seems converage faster than VGG and RN.** It seems that the validation loss of our IRN_m network has a stronger fluctuation than VGG, but it's not true. Because the order of magnitudes (数量级) of their validation loss are too much different. 

| MSE(MLAE) | VGG | VGG_seg | RN | IRN_p| IRN_m (!!!) |
| ----- | ----- | ----- | ----- | -----| ----- |
| Train set | 0.00023(0.18) | 0.00022(0.11) | 0.00289(1.70) | 0.00015(-0.56) | **0.00010(-0.57)** |
| Test set | 0.13354(4.56) | 0.14972(4.79) | 0.15874(4.84) | 0.00087(0.97) | **0.00058(0.81)** |

![Pie Number: Loss](https://github.com/RyuZhihao123/Instance-based-RN/blob/master/image/PieNumberLoss.png)

### Task1.2: PieLineWidth.

[[Codes]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task1_ourNewTasks/PieLineWidth) 

* `The line width` are different between training and testing sets in this task. By default, the line width of the piechart in training sets is 1, while the width of those in testing sets is 2 or 3. In addition, the output of networks is 6-dim vector, and each chart contains 3 to 6 pie sectors.

<div align=center><img width="350" src="https://github.com/RyuZhihao123/Instance-based-RN/blob/master/image/PieLineWidth.png"/></div>

* Due to different line width, PieLineWidth is unlike PieNumber whose training and testing sets have same appearence domain. However, the result is surprising. **We found that both IRN_p and IRN_m can get a good result in testing set.**. That means if we segmeneted objects in advance and directly using CNN to extract their individual features, it does make some effect.

* 在IRN_m的曲线中，val loss随着train loss是一起降低的，并且能持续相当一段时间，这证明分割是有帮助的。但是随着训练轮数增大到一定程度，此时网络在trainset上已经学的很好了，但是由于网络并没有见过val set中的图形，而网络又一直只在train set上进行学习，导致慢慢它开始不能适用于val set中的图像了。虽然这里我写作overfitting，但其实也不算是overfitting，因为毕竟两个数据集的appearence是完全独立的。这个现象是很正常的，因为随着训练越来越多，网络倾向于更加拟合它的训练集，但是在训练集达到0.00018时，也就是已经拟合training set的情况下，在测试集上我们也可以得到相当好的loss=0.00032。

| MSE(MLAE) | VGG | RN | IRN_p| IRN_m (!!!) |
| ----- | -----  | ----- | -----| ----- |
| Train set | 0.00036(0.69)  | 0.00429(2.26) | 0.00065(0.59) | **0.00018(0.01)** |
| Test set | 0.06459(4.26)  | 0.05459(4.08) | 0.00160(1.27) | **0.00032(0.33)** |

![Pie LineWidth: Loss](https://github.com/RyuZhihao123/Instance-based-RN/blob/master/image/PieLineWidthLoss.png)

### Task1.3: PieColor

[[FixedTrain Codes]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task1_ourNewTasks/PieColor_fixedtrain)  [[RandomColor Codes]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task1_ourNewTasks/PieColor_randomcolor) 

* We design two tasks in PieColor. (1) **FixedTrain**: the training set only uses 6 colors, while testing set use random colors. (2) **RandomColor**: Both training and testing set use random colors.

![](https://github.com/RyuZhihao123/Instance-based-RN/blob/master/image/PieColor.png)

| FixedTrain | VGG | RN | IRN_m (!!!) |
| ----- | -----  | ----- | ----- |
| Train set | 0.00040(0.76)  | 0.00443(2.28)  | **0.00014(-0.24)** |
| Test set | 0.06982(3.90)  | 0.08715(4.38)  | **0.00480(1.45)** |

| RandomColor | VGG | RN |  IRN_m (!!!) |
| ----- | -----  | ----- | ----- |
| Train set | 0.00051(0.86)  | 0.00492(2.36)| **0.00015(-0.22)** |
| Test set | 0.00095(0.95)  | 0.00599(2.41)  | **0.00015(-0.21)** |

![Pie Number: Loss](https://github.com/RyuZhihao123/Instance-based-RN/blob/master/image/PieColor_fixedTrain.png)

![Pie Number: Loss](https://github.com/RyuZhihao123/Instance-based-RN/blob/master/image/PieColor_random.png)

### Task1.4: Pie3_6 and Pie3_12

[[Pie3_6 Codes]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task1_ourNewTasks/Pie3_6)  [[Pie3_12 Codes]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task1_ourNewTasks/Pie3_12) 

* This task is to test the performance when the maximun object number is large and the number changes greatly. The object number in both training and testing sets is 3 to 6 (12) for task Pie3_6(Pie3_12).  I think 12 is large enough since if we use a larger number, the chart would looks messy.

<div align=center><img width="350" src="https://github.com/RyuZhihao123/Instance-based-RN/blob/master/image/Pie3_6_12.png"/></div>

* **It's clear that VGG and RN performs worse when the number of objects increased, as the following results proved. Whereas, our network, IRN_m can still perform very nice**. `It seems that the MLAE of VGG increases more significantly than MSE.`

| MSE(MLAE) | VGG | RN | IRN_p| IRN_m (!!!) |
| ----- | -----  | ----- | -----| ----- |
| Pie3_6: Train set | 0.00036(0.67)  | 0.00435(2.26) | 0.00016(-0.25) | **0.00012(-0.29)** |
| Pie3_6: Test set  | 0.00038(0.70)  | 0.00438(2.26) | 0.00017(-0.22) | **0.00012(-0.28)** |
| Pie3_12: Train set | 0.00089(1.24)  | 0.00705(2.54) | 0.00033(0.12) | **0.00023(0.00)** |
| Pie3_12: Test set  | 0.00098(1.29)  | 0.00727(2.56) | 0.00041(0.25) | **0.00024(0.02)** |

<div align=center><img width="650" src="https://github.com/RyuZhihao123/Instance-based-RN/blob/master/image/Pie3_6_12Loss.png"/></div>


## Experiments2: ClevelandMcGill
(The experiments that are same as Daniel's paper.)

For the following experiments, I only show the MSE and MLAE on testing sets.

钟发海在帮忙跑这些，跑完后再贴上（目前进度：按顺序到position-length的type3）。

### Task2.1: Position-Angle.

Codes: 
[[Bar charts]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task2_cleverlAndMcGill/1position_angle_Bar) 
[[Pie charts]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task2_cleverlAndMcGill/1position_angle_Pie)

<div align=center><img width="350" src="https://github.com/RyuZhihao123/Instance-based-RN/blob/master/image/Position_angle.png"/></div>

* Although the MSE of VGG and IRN_m seems similar, the MLAE are different a lot.

| MSE(MLAE) | VGG | RN | IRN_m (!!!) |
| ----- | -----  |  -----| ----- |
| Bar chart | 0.00016(0.21)  | 0.00394(2.34)  | **0.00014(-0.31)** |
| Pie chart | -(-)  | -(-)  | **-(-)** |

### Task2.2: Position-Length.

Codes: 
[[MULTI]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task2_cleverlAndMcGill/2position_length_multi) 
[[Type1]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task2_cleverlAndMcGill/2position_length_type1)
[[Type2]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task2_cleverlAndMcGill/2position_length_type2)
[[Type3]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task2_cleverlAndMcGill/2position_length_type3)
[[Type4]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task2_cleverlAndMcGill/2position_length_type4)
[[Type5]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task2_cleverlAndMcGill/2position_length_type5)

<div align=center><img width="350" src="https://github.com/RyuZhihao123/Instance-based-RN/blob/master/image/Position_multi.png"/></div>

### Task2.3: Point-Cloud.

Codes: 
[[Num10]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task2_cleverlAndMcGill/3point_cloud_10) 
[[Num100]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task2_cleverlAndMcGill/3point_cloud_100)
[[Num1000]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task2_cleverlAndMcGill/4point_cloud_1000)


## Experiments3: Supplement 

### Task3.1: The effect of Non_local_block.
### Task3.2: Can orignal RN be improved by changing its structure?
### Task3.3: How would RN perform when the objects are segmented directly rather than extracted by CNN.


