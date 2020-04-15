# Instance-based-RN

We designed an **IRN_m network** for multi-objects ratio tasks, and an **IRN_p network** for pair-ratio estimation tasks. 

if you meet any problem or bug, please tell me (liuzh96@outlook.com)  thanks (^,^).

## Usage.

### 1. Environments:

* **Recommended Env.** Python3.6, Tensorflow1.14.0, Keras 2.2.4.

* **Neccessary libaries:** numpy, opencv-python, argparse, scikit-learn, openpyxl, pickle.

### 2. Example:

We provide 2 versions of code: one is the [Code_RunOnce](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Codes_RunOnce) that only run the experiments **once**; the other is [Code_PureExperiments](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Codes_PureExperiment_default_5_times) that could automatically run the network **5 times by default** and **compute the average and SD of MSE and MLAE**. For quick experiments, we could only focus on [Code_PureExperiments](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Codes_PureExperiment_default_5_times). 

I'll take **Task1.1 Pie3_6** [(codes)](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Codes_PureExperiment_default_5_times/Task1_ourNewTasks/Pie3_6) as an example to show how to use the codes. Dataset will re-generated before training each time, 

* ** ** 

```
python Net_VGG.py    --gpu 0 (--times 5)
python Net_RN.py     --gpu 1 (--times 5)
python Net_RN_seg.py --gpu 2 (--times 5)
python Net_IRNm.py   --gpu 3 (--times 5)  or python Net_IRNp.py   --gpu 3 (--times 5)

```

* **Second, run a network.** This command will train and test the `IRN_m` network. When training step is finished, the training informations(MLAE, MSE and histories etc.), best model(on val sets) and the predicted results can be obtained in folder `./results/IRN_m`.

```
python3.6 Net_IRNm.py --gpu 2 (--savedir IRNm_03)      # GPU ID ( target directory: ./results/IRNm_03)
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

### Task1.1: Pie3_6 and Pie3_12

[[Pie3_6 Codes]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task1_ourNewTasks/Pie3_6)  [[Pie3_12 Codes]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task1_ourNewTasks/Pie3_12) 

* This task is to test the performance when the maximun object number is large and the number changes greatly. The object number in both training and testing sets is 3 to 6 (12) for task Pie3_6(Pie3_12).  I think 12 is large enough since if we use a larger number, the chart would looks messy.

<div align=center><img width="750" src="https://github.com/RyuZhihao123/Instance-based-RN/blob/master/image/Pie3_6_12.png"/></div>

* **It's clear that VGG and RN performs worse when the number of objects increased, as the following results proved. Whereas, our network, IRN_m can still perform very nice**. `It seems that the MLAE of VGG increases more significantly than MSE.`

| MSE(MLAE) | VGG | RN | IRN_p| IRN_m (!!!) |
| ----- | -----  | ----- | -----| ----- |
| Pie3_6: Train set | 0.00036(0.67)  | 0.00435(2.26) | 0.00016(-0.25) | **0.00012(-0.29)** |
| Pie3_6: Test set  | 0.00038(0.70)  | 0.00438(2.26) | 0.00017(-0.22) | **0.00012(-0.28)** |
| Pie3_12: Train set | 0.00089(1.24)  | 0.00705(2.54) | 0.00033(0.12) | **0.00023(0.00)** |
| Pie3_12: Test set  | 0.00098(1.29)  | 0.00727(2.56) | 0.00041(0.25) | **0.00024(0.02)** |

<div align=center><img width="650" src="https://github.com/RyuZhihao123/Instance-based-RN/blob/master/image/Pie3_6_12Loss.png"/></div>

### Task1.4: PieColor

[[FixedTrain Codes]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task1_ourNewTasks/PieColor_fixedtrain)  [[RandomColor Codes]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task1_ourNewTasks/PieColor_randomcolor) 

* We design two tasks in PieColor. (1) **FixedTrain**: the training set only uses 6 colors, while testing set use random colors. (2) **RandomColor**: Both training and testing set use random colors.

<div align=center><img width="750" src="https://github.com/RyuZhihao123/Instance-based-RN/blob/master/image/PieColor.png"/></div>

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

### Task1.2: PieNumber.

[[Codes]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task1_ourNewTasks/PieNumber) 

* `The range of object number` are different between training and testing sets. By default, the pie charts in training sets contain 3 to 6 pie sectors, while those in testing sets contain 7 to 9 pie sectors. For VGG, RN and IRNm, all the outputs are `9-dim vector`.

<div align=center><img width="750" src="https://github.com/RyuZhihao123/Instance-based-RN/blob/master/image/PieNumber.png"/></div>

<div align=center><img width="350" src="https://github.com/RyuZhihao123/Instance-based-RN/blob/master/image/Sample.png"/></div>

* **Only our IRN_m and IRN_p can get a good result on testing set. (1) Our network can deal with the condition that training and testing sets have different object number. (2) Our network seems converage faster than VGG and RN.** It seems that the validation loss of our IRN_m network has a stronger fluctuation than VGG, but it's not true. Because the order of magnitudes (数量级) of their validation loss are too much different. 

| MSE(MLAE) | VGG | VGG_seg | RN | IRN_p| IRN_m (!!!) |
| ----- | ----- | ----- | ----- | -----| ----- |
| Train set | 0.00023(0.18) | 0.00022(0.11) | 0.00289(1.70) | 0.00015(-0.56) | **0.00010(-0.57)** |
| Test set | 0.13354(4.56) | 0.14972(4.79) | 0.15874(4.84) | 0.00087(0.97) | **0.00058(0.81)** |

![Pie Number: Loss](https://github.com/RyuZhihao123/Instance-based-RN/blob/master/image/PieNumberLoss.png)

### Task1.3: PieLineWidth.

[[Codes]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task1_ourNewTasks/PieLineWidth) 

* `The line width` are different between training and testing sets in this task. By default, the line width of the piechart in training sets is 1, while the width of those in testing sets is 2 or 3. In addition, the output of networks is 6-dim vector, and each chart contains 3 to 6 pie sectors.

<div align=center><img width="750" src="https://github.com/RyuZhihao123/Instance-based-RN/blob/master/image/PieLineWidth.png"/></div>

* Due to different line width, PieLineWidth is unlike PieNumber whose training and testing sets have same appearence domain. However, the result is surprising. **We found that both IRN_p and IRN_m can get a good result in testing set.**. That means if we segmeneted objects in advance and directly using CNN to extract their individual features, it does make some effect.

* For our IRN_m network, val_loss declines with train_loss in the early stage and also keeps for many epochs. We could see that IRN_m network is able to perform very well on val_set as on train_set. However, because we always opitimize the network using train_set, so it's okay and normal that val_loss would become bad on val_set when the network try to get much better results on train_set.

| MSE(MLAE) | VGG | RN | IRN_p| IRN_m (!!!) |
| ----- | -----  | ----- | -----| ----- |
| Train set | 0.00036(0.69)  | 0.00429(2.26) | 0.00065(0.59) | **0.00018(0.01)** |
| Test set | 0.06459(4.26)  | 0.05459(4.08) | 0.00160(1.27) | **0.00032(0.33)** |

![Pie LineWidth: Loss](https://github.com/RyuZhihao123/Instance-based-RN/blob/master/image/PieLineWidthLoss.png)


## Experiments2: ClevelandMcGill
(The experiments that are same as Daniel's paper.)

For the following experiments, I only show the MSE and MLAE on testing sets.


### Task2.1: Position-Angle.

Codes: 
[[Bar charts]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task2_cleverlAndMcGill/1position_angle_Bar) 
[[Pie charts]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task2_cleverlAndMcGill/1position_angle_Pie)

<div align=center><img width="350" src="https://github.com/RyuZhihao123/Instance-based-RN/blob/master/image/Position_angle.png"/></div>

| MSE(MLAE) | VGG | RN | IRN_m (!!!) |
| ----- | -----  |  -----| ----- |
| Bar chart | 0.00016(0.21)  | 0.00394(2.34)  | **0.00014(-0.31)** |
| Pie chart | 0.00028(0.57)  | 0.00390(2.34)  | **0.00021(0.11)** |

### Task2.2: Position-Length.

Codes: 
[[MULTI]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task2_cleverlAndMcGill/2position_length_multi) 
[[Type1]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task2_cleverlAndMcGill/2position_length_type1)
[[Type2]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task2_cleverlAndMcGill/2position_length_type2)
[[Type3]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task2_cleverlAndMcGill/2position_length_type3)
[[Type4]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task2_cleverlAndMcGill/2position_length_type4)
[[Type5]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task2_cleverlAndMcGill/2position_length_type5)

<div align=center><img width="350" src="https://github.com/RyuZhihao123/Instance-based-RN/blob/master/image/Position_multi.png"/></div>


| MSE(MLAE) | VGG | RN | IRN_p (!!!) |
| ----- | -----  |  -----| ----- |
| Type1 | **0.000004(-1.77)**  | 0.000546(0.80)  | 0.000008(-1.49) |
| Type2 | **0.000005(-1.66)**  | 0.000485(0.72)  | 0.000007(-1.57) |
| Type3 | **0.000006(-1.63)**  | 0.000524(0.78)  | 0.000007(-1.54) |
| Type4 | **0.000004(-1.80)**  | 0.000494(0.74)  | 0.000010(-1.34) |
| Type5 | **0.000004(-1.77)**  | 0.000509(0.77)  | 0.000009(-1.42) |
| Multi | 0.000011(-1.41)  | 0.000507(0.76)  | **0.000008(-1.49)** |

### Task2.3: Point-Cloud.

Codes: 
[[Num10]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task2_cleverlAndMcGill/3point_cloud_10) 
[[Num100]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task2_cleverlAndMcGill/3point_cloud_100)
[[Num1000]](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task2_cleverlAndMcGill/3point_cloud_1000)

| MSE(MLAE) | VGG | RN | IRN_p (!!!) |
| ----- | -----  |  -----| ----- |
| Base10 | 0.000099(-0.17)  | 0.002772(2.06)  | **0.000016(-1.26)** |
| Base100 | 0.099914(4.77)  | 0.005228(2.56)  | **0.000045(-0.65)** |
| Base1000 | 0.101107(4.79)  | 0.022654(3.58)  | **0.000894(1.29)** |

## Experiments3: Supplement 

### Task3.1: The effect of Non_local_block.
### Task3.2: Can orignal RN be improved by changing its structure?
### Task3.3: How would RN perform when the objects are segmented directly rather than extracted by CNN.


