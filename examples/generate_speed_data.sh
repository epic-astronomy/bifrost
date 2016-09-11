#!/bin/bash
for i in `seq 1 10`;
do
        python fft_speed_test.py $i
        nvidia-smi | grep python | cut -c 10-17 | xargs -n 1 kill -9
done
