# Instance-based-RN


### How to use.

* **environments:** Python3.6, Tensorflow1.14.0

* **Neccessary python libaries:** numpy, opencv-python, argparse, scikit-learn,openpyxl

Here, I'll take **Task1.1 PieNumber** as an example to show how to use the codes.

First, create your own datasets. The following command will create a 'datasets' folder in current path.

```
python3.6 Dataset_generator.py
```

Second, run a network. This command will train and test the 'IRN_m' network. When training step is finished, the training informations, best model and the predicted results can be obtained in folder './results/IRN_m'.

```
python3.6 Net_IRNm.py
```

**Details**: (1) Train/val/test sets contains 60000/20000/20000 charts respectively. We use Adam optimizer (lr =0.0001). (2) During training, we shuffle the datasets for each epoch, and only save the best model which gets the lowest mse loss on validation set.

**Notice**: In rare cases, if the loss of some network doesn't decrease obviously after the first epoch, please kill it and restart the program, because Adam can get a way low loss value even during the first epoch.

### Experiments1: Our new tasks. 
(These experiments focus on verifying the generalization ability of networks.)

* **Task1.1: PieNumber.**

> `The range of object number` are different between training and testing sets in this task. By default, each pie chart in training sets contains 3 to 6 pie sectors, while the charts in testing sets contain 7 to 9 pie sectors.

* **Task1.2: PieLineWidth.**

> `The line width` are different between training and testing sets in this task. By default, the line width of the piechart in training sets is 1, while the width of those in testing sets is 2 or 3.

* **Task1.3: Pie3_12.**

> In both training and testing sets, each pie chart contains 3 to 12 pie sectors. This task is to test the performance when the maximun object number is large and the number changes greatly. I think 12 is large enough since if we use a larger number, the chart would looks messy.


### Experiments2: ClevelandMcGill
(The experiments that are same as Daniel's paper.)

* **Task2.1: Position-Angle.**
* **Task2.2: Position-Length.**
* **Task2.3: Point-Cloud.**


