#!/bin/bash

# shellcheck disable=SC1101
python gbespoketart/train.py --description "removed padding from target to decoder" \
--learning-rate 0.0002 --batch-size 6  --output-window 5 --bert-pretraining "bert-base-uncased" \
--bert-finetuning "true"
