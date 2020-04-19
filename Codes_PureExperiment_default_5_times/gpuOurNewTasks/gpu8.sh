#!/bin/bash

CRTDIR=$1
gpuid=$2
times=$3
echo $CRTDIR, $gpuid



basicpath="./Task1_ourNewTasks/"



cd $CRTDIR; cd $basicpath; cd "PieNumber/" ; python3.6 Net_IRNm.py --gpu $gpuid --times $times
cd $CRTDIR; cd $basicpath; cd "PieColor_fixedtrain/" ; python3.6 Net_VGG.py --gpu $gpuid --times $times
cd $CRTDIR; cd $basicpath; cd "PieColor_randomcolor/" ; python3.6 Net_VGG.py --gpu $gpuid --times $times


cd $CRTDIR; cd $basicpath; cd "Pie3_6/" ; python3.6 Net_IRNm.py --gpu $gpuid --times $times


echo " Omedetou !!  [GPU8] has done all the works."
echo " Omedetou !!  [GPU8] has done all the works."
echo " Omedetou !!  [GPU8] has done all the works."
echo " Omedetou !!  [GPU8] has done all the works."