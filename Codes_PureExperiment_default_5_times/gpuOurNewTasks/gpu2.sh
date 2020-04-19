#!/bin/bash

CRTDIR=$1
gpuid=$2
times=$3
echo $CRTDIR, $gpuid



basicpath="./Task1_ourNewTasks/"

cd $CRTDIR; cd $basicpath; cd "Bar3_6/" ; python3.6 Net_RN.py --gpu $gpuid --times $times


cd $CRTDIR; cd $basicpath; cd "BarColor_fixedtrain/" ; python3.6 Net_RN.py --gpu $gpuid --times $times
cd $CRTDIR; cd $basicpath; cd "BarColor_randomcolor/" ; python3.6 Net_RN.py --gpu $gpuid --times $times

cd $CRTDIR; cd $basicpath; cd "BarLineWidth/" ; python3.6 Net_RN.py --gpu $gpuid --times $times
cd $CRTDIR; cd $basicpath; cd "BarNumber/" ; python3.6 Net_RN.py --gpu $gpuid --times $times


cd $CRTDIR; cd $basicpath; cd "Pie3_6/" ; python3.6 Net_RN.py --gpu $gpuid --times $times


cd $CRTDIR; cd $basicpath; cd "PieColor_fixedtrain/" ; python3.6 Net_RN.py --gpu $gpuid --times $times
cd $CRTDIR; cd $basicpath; cd "PieColor_randomcolor/" ; python3.6 Net_RN.py --gpu $gpuid --times $times

cd $CRTDIR; cd $basicpath; cd "PieLineWidth/" ; python3.6 Net_RN.py --gpu $gpuid --times $times
cd $CRTDIR; cd $basicpath; cd "PieNumber/" ; python3.6 Net_RN.py --gpu $gpuid --times $times

cd $CRTDIR; cd $basicpath; cd "Pie3_6/" ; python3.6 Net_RN_seg.py --gpu $gpuid --times $times

cd $CRTDIR; cd $basicpath; cd "PieColor_fixedtrain/" ; python3.6 Net_RN_seg.py --gpu $gpuid --times $times




echo " Omedetou !!  [GPU2] has done all the works."
echo " Omedetou !!  [GPU2] has done all the works."
echo " Omedetou !!  [GPU2] has done all the works."
echo " Omedetou !!  [GPU2] has done all the works."