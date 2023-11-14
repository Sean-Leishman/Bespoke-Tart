#!/bin/bash

# shellcheck disable=SC1101
python gbespoketart/train.py --description "decrease lr to 5e-5 and allocate additional training data to 90/10 split" \
--learning-rate 0.00001 --batch-size 16  --output-window 5 --bert-pretraining "bert-base-uncased" \
--bert-finetuning "true" --overwrite "true"
