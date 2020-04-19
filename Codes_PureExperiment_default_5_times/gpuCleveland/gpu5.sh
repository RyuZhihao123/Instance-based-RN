#!/bin/bash

CRTDIR=$1
gpuid=$2
times=$3
echo $CRTDIR, $gpuid



basicpath="./Task2_clevelAndMcGill/"


cd $CRTDIR; cd $basicpath; cd "2position_length_type3/" ; python3.6 Net_VGG.py --gpu $gpuid --times $times
cd $CRTDIR; cd $basicpath; cd "2position_length_type4/" ; python3.6 Net_VGG.py --gpu $gpuid --times $times
cd $CRTDIR; cd $basicpath; cd "2position_length_type5/" ; python3.6 Net_VGG.py --gpu $gpuid --times $times

cd $CRTDIR; cd $basicpath; cd "3point_cloud_10/" ; python3.6 Net_VGG.py --gpu $gpuid --times $times
cd $CRTDIR; cd $basicpath; cd "3point_cloud_100/" ; python3.6 Net_VGG.py --gpu $gpuid --times $times
cd $CRTDIR; cd $basicpath; cd "3point_cloud_1000/" ; python3.6 Net_VGG.py --gpu $gpuid --times $times



echo " Omedetou !!  [GPU5] has done all the works."
echo " Omedetou !!  [GPU5] has done all the works."
echo " Omedetou !!  [GPU5] has done all the works."
echo " Omedetou !!  [GPU5] has done all the works."