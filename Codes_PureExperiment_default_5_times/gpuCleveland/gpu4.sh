#!/bin/bash

CRTDIR=$1
gpuid=$2
times=$3
echo $CRTDIR, $gpuid



basicpath="./Task2_clevelAndMcGill/"

cd $CRTDIR; cd $basicpath; cd "1position_angle_Bar/" ; python3.6 Net_IRNm.py --gpu $gpuid --times $times

cd $CRTDIR; cd $basicpath; cd "2position_length_type2/" ; python3.6 Net_IRNp.py --gpu $gpuid --times $times
cd $CRTDIR; cd $basicpath; cd "2position_length_multi/" ; python3.6 Net_IRNp.py --gpu $gpuid --times $times
cd $CRTDIR; cd $basicpath; cd "2position_length_type1/" ; python3.6 Net_IRNp.py --gpu $gpuid --times $times
cd $CRTDIR; cd $basicpath; cd "Bar3_12/" ; python3.6 Net_IRNm.py --gpu $gpuid --times $times




echo " Omedetou !!  [GPU4] has done all the works."
echo " Omedetou !!  [GPU4] has done all the works."
echo " Omedetou !!  [GPU4] has done all the works."
echo " Omedetou !!  [GPU4] has done all the works."