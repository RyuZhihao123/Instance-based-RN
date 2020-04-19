#!/bin/bash

CRTDIR=$1
gpuid=$2
times=$3
echo $CRTDIR, $gpuid



basicpath="./Task1_ourNewTasks/"



cd $CRTDIR; cd $basicpath; cd "BarNumber/" ; python3.6 Net_RN_seg.py --gpu $gpuid --times $times
cd $CRTDIR; cd $basicpath; cd "PieColor_randomcolor/" ; python3.6 Net_RN_seg.py --gpu $gpuid --times $times

cd $CRTDIR; cd $basicpath; cd "PieLineWidth/" ; python3.6 Net_RN_seg.py --gpu $gpuid --times $times

echo " Omedetou !!  [GPU5] has done all the works."
echo " Omedetou !!  [GPU5] has done all the works."
echo " Omedetou !!  [GPU5] has done all the works."
echo " Omedetou !!  [GPU5] has done all the works."