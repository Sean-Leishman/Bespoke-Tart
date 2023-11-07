#!/bin/bash

python cbespoketart/train.py --description "using a transformer after BERT embedding. predicting tokens" --learning-rate 0.003 --batch-size 8  --output-window 5 --loss-weight 5
