#!/bin/bash

CRTDIR=$1
gpuid=$2
times=$3
echo $CRTDIR, $gpuid



basicpath="./Task2_clevelAndMcGill/"

cd $CRTDIR; cd $basicpath; cd "3point_cloud_10/" ; python3.6 Net_IRNp.py --gpu $gpuid --times $times
cd $CRTDIR; cd $basicpath; cd "3point_cloud_100/" ; python3.6 Net_IRNp.py --gpu $gpuid --times $times
cd $CRTDIR; cd $basicpath; cd "3point_cloud_1000/" ; python3.6 Net_IRNp.py --gpu $gpuid --times $times

cd $CRTDIR; cd $basicpath; cd "1position_angle_Pie/" ; python3.6 Net_IRNm.py --gpu $gpuid --times $times

cd $CRTDIR; cd "Task1_ourNewTasks/"; cd "Bar3_12/" ; python3.6 Net_VGG.py --gpu $gpuid --times $times

echo " Omedetou !!  [GPU8] has done all the works."
echo " Omedetou !!  [GPU8] has done all the works."
echo " Omedetou !!  [GPU8] has done all the works."
echo " Omedetou !!  [GPU8] has done all the works."