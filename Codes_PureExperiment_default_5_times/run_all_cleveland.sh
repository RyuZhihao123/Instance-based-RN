#!/bin/bash

CRTDIR=$(pwd)

bash ./gpuCleveland/gpu1.sh $CRTDIR 0 $1 & 
bash ./gpuCleveland/gpu2.sh $CRTDIR 1 $1 & 
bash ./gpuCleveland/gpu3.sh $CRTDIR 2 $1 &
bash ./gpuCleveland/gpu4.sh $CRTDIR 3 $1 & 
bash ./gpuCleveland/gpu5.sh $CRTDIR 4 $1 & 
bash ./gpuCleveland/gpu6.sh $CRTDIR 5 $1 &
bash ./gpuCleveland/gpu7.sh $CRTDIR 6 $1 & 
bash ./gpuCleveland/gpu8.sh $CRTDIR 7 $1 & 



