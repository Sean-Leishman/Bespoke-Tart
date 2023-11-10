#!/bin/bash

# shellcheck disable=SC1101
python mbespoketart/train.py --description "pretraining with masked ml" \
--learning-rate 0.00005 --batch-size 6  --output-window 5 --bert-pretraining "bert-base-uncased" \
--bert-finetuning "true"
