#!/bin/bash

CRTDIR=$1
gpuid=$2
times=$3
echo $CRTDIR, $gpuid



basicpath="./Task2_clevelAndMcGill/"

cd $CRTDIR; cd $basicpath; cd "1position_angle_Bar/" ; python3.6 Net_RN_seg.py --gpu $gpuid --times $times


cd $CRTDIR; cd $basicpath; cd "2position_length_multi/" ; python3.6 Net_RN_seg.py --gpu $gpuid --times $times
cd $CRTDIR; cd $basicpath; cd "2position_length_type1/" ; python3.6 Net_RN_seg.py --gpu $gpuid --times $times


cd $CRTDIR; cd $basicpath; cd "Bar3_12/" ; python3.6 Net_RN_seg.py --gpu $gpuid --times $times


echo " Omedetou !!  [GPU3] has done all the works."
echo " Omedetou !!  [GPU3] has done all the works."
echo " Omedetou !!  [GPU3] has done all the works."
echo " Omedetou !!  [GPU3] has done all the works."