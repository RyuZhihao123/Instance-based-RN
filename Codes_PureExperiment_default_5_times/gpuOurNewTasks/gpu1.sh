#!/bin/bash

CRTDIR=$1
gpuid=$2
times=$3
echo $CRTDIR, $gpuid



basicpath="./Task1_ourNewTasks/"

cd $CRTDIR; cd $basicpath; cd "Bar3_6/" ; python3.6 Net_VGG.py --gpu $gpuid --times $times


cd $CRTDIR; cd $basicpath; cd "BarColor_fixedtrain/" ; python3.6 Net_VGG.py --gpu $gpuid --times $times
cd $CRTDIR; cd $basicpath; cd "BarColor_randomcolor/" ; python3.6 Net_VGG.py --gpu $gpuid --times $times

cd $CRTDIR; cd $basicpath; cd "BarNumber/" ; python3.6 Net_VGG.py --gpu $gpuid --times $times





echo " Omedetou !!  [GPU1] has done all the works."
echo " Omedetou !!  [GPU1] has done all the works."
echo " Omedetou !!  [GPU1] has done all the works."
echo " Omedetou !!  [GPU1] has done all the works."