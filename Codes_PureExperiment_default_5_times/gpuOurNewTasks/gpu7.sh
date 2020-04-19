#!/bin/bash

CRTDIR=$1
gpuid=$2
times=$3
echo $CRTDIR, $gpuid



basicpath="./Task1_ourNewTasks/"


cd $CRTDIR; cd $basicpath; cd "BarNumber/" ; python3.6 Net_IRNm.py --gpu $gpuid --times $times

cd $CRTDIR; cd $basicpath; cd "PieLineWidth/" ; python3.6 Net_VGG.py --gpu $gpuid --times $times
cd $CRTDIR; cd $basicpath; cd "PieNumber/" ; python3.6 Net_VGG.py --gpu $gpuid --times $times


echo " Omedetou !!  [GPU7] has done all the works."
echo " Omedetou !!  [GPU7] has done all the works."
echo " Omedetou !!  [GPU7] has done all the works."
echo " Omedetou !!  [GPU7] has done all the works."