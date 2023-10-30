#!/bin/bash

python bespoketart/train.py --description "" --learning-rate 0.0003 --batch-size 32  --output-window 5 --loss-weight 1 --overwrite "true"
