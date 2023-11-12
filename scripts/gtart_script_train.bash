#!/bin/bash

# shellcheck disable=SC1101
python gbespoketart/train.py --description "decrease lr (0.0002 -> 0.0001), update generation config" \
--learning-rate 0.0001 --batch-size 6  --output-window 5 --bert-pretraining "bert-base-uncased" \
--bert-finetuning "true" --overwrite "false"
