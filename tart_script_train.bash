#!/bin/bash

python bespoketart/train.py --description "using a transformer after BERT embedding" --learning-rate 0.003 --batch-size 64  --output-window 5 --loss-weight 5
