#!/bin/bash

CRTDIR=$(pwd)

bash ./gpuOurNewTasks/gpu1.sh $CRTDIR 0 $1 & 
bash ./gpuOurNewTasks/gpu2.sh $CRTDIR 1 $1 & 
bash ./gpuOurNewTasks/gpu3.sh $CRTDIR 2 $1 &
bash ./gpuOurNewTasks/gpu4.sh $CRTDIR 3 $1 & 
bash ./gpuOurNewTasks/gpu5.sh $CRTDIR 4 $1 & 
bash ./gpuOurNewTasks/gpu6.sh $CRTDIR 5 $1 &
bash ./gpuOurNewTasks/gpu7.sh $CRTDIR 6 $1 & 
bash ./gpuOurNewTasks/gpu8.sh $CRTDIR 7 $1 & 



