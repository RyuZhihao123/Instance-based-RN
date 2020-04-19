#!/bin/bash

CRTDIR=$1
gpuid=$2
times=$3
echo $CRTDIR, $gpuid



basicpath="./Task1_ourNewTasks/"

cd $CRTDIR; cd $basicpath; cd "Bar3_6/" ; python3.6 Net_IRNm.py --gpu $gpuid --times $times


cd $CRTDIR; cd $basicpath; cd "BarColor_fixedtrain/" ; python3.6 Net_IRNm.py --gpu $gpuid --times $times
cd $CRTDIR; cd $basicpath; cd "BarColor_randomcolor/" ; python3.6 Net_IRNm.py --gpu $gpuid --times $times

cd $CRTDIR; cd $basicpath; cd "BarLineWidth/" ; python3.6 Net_IRNm.py --gpu $gpuid --times $times












echo " Omedetou !!  [GPU4] has done all the works."
echo " Omedetou !!  [GPU4] has done all the works."
echo " Omedetou !!  [GPU4] has done all the works."
echo " Omedetou !!  [GPU4] has done all the works."