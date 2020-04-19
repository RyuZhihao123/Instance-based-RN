#!/bin/bash

CRTDIR=$1
gpuid=$2
times=$3
echo $CRTDIR, $gpuid



basicpath="./Task2_clevelAndMcGill/"

cd $CRTDIR; cd $basicpath; cd "Pie3_12/" ; python3.6 Net_RN.py --gpu $gpuid --times $times

cd $CRTDIR; cd $basicpath; cd "1position_angle_Pie/" ; python3.6 Net_RN_seg.py --gpu $gpuid --times $times


cd $CRTDIR; cd $basicpath; cd "2position_length_type3/" ; python3.6 Net_IRNp.py --gpu $gpuid --times $times
cd $CRTDIR; cd $basicpath; cd "2position_length_type4/" ; python3.6 Net_IRNp.py --gpu $gpuid --times $times
cd $CRTDIR; cd $basicpath; cd "2position_length_type5/" ; python3.6 Net_IRNp.py --gpu $gpuid --times $times
cd $CRTDIR; cd $basicpath; cd "Pie3_12/" ; python3.6 Net_IRNm.py --gpu $gpuid --times $times


echo " Omedetou !!  [GPU7] has done all the works."
echo " Omedetou !!  [GPU7] has done all the works."
echo " Omedetou !!  [GPU7] has done all the works."
echo " Omedetou !!  [GPU7] has done all the works."