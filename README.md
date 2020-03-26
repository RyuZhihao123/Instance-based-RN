# Instance-based-RN


## How to use.

### 1. Environments:

* **Recommended Env.** Python3.6, Tensorflow1.14.0

* **Neccessary libaries:** numpy, opencv-python, argparse, scikit-learn,openpyxl

### 2. Example:

Here, I'll take **Task1.1 PieNumber** [(codes)](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task1_ourNewTasks/PieNumber)  as an example to show how to use the codes.

First, create your own datasets. The following command will create a 'datasets' folder in current path.

```
python3.6 Dataset_generator.py
```

Second, run a network. This command will train and test the 'IRN_m' network. When training step is finished, the training informations(MLAE and MSE etc.), best model(on val sets) and the predicted results can be obtained in folder './results/IRN_m'.

```
python3.6 Net_IRNm.py
```

**Details**: (1) Train/val/test sets contains 60000/20000/20000 charts respectively. We use Adam optimizer (lr =0.0001). (2) During training, we shuffle the datasets for each epoch, and only save the best model which gets the lowest mse loss on validation set.

**Notice**: In rare cases (very low probability), if the loss of some network doesn't decrease obviously after the first epoch, please kill it and restart the program, because Adam could usually get a way low loss value even during the first epoch.

## Experiments1: Our new tasks. 
(These experiments focus on verifying the generalization ability of networks.)

* **Task1.1: PieNumber.** [Codes](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task1_ourNewTasks/PieNumber) 

> `The range of object number` are different between training and testing sets in this task. By default, each pie chart in training sets contains 3 to 6 pie sectors, while the charts in testing sets contain 7 to 9 pie sectors.

* **Task1.2: PieLineWidth.** [Codes](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task1_ourNewTasks/PieLineWidth) 

> `The line width` are different between training and testing sets in this task. By default, the line width of the piechart in training sets is 1, while the width of those in testing sets is 2 or 3.

* **Task1.3: Pie3_12.**  [Codes](https://github.com/RyuZhihao123/Instance-based-RN/tree/master/Task1_ourNewTasks/Pie3_12) 

> In both training and testing sets, each pie chart contains 3 to 12 pie sectors. This task is to test the performance when the maximun object number is large and the number changes greatly. I think 12 is large enough since if we use a larger number, the chart would looks messy.


## Experiments2: ClevelandMcGill
(The experiments that are same as Daniel's paper.)

* **Task2.1: Position-Angle.**
* **Task2.2: Position-Length.**
* **Task2.3: Point-Cloud.**

## Experiments3: Supplement 

* **Task3.1: The effect of Non_local_block.**
* **Task3.2: Can orignal RN be improved by changing its structure?**
*


