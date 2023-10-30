#!/bin/bash

python lstm/train.py --learning-rate 0.00006 --batch-size 64  --output-window 1 --loss-weight 3
