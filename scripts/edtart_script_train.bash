#!/bin/bash

# shellcheck disable=SC1101
python edbespoketart/train.py --description "back to encoder decoder structure" \
--learning-rate 0.00001 --batch-size 8  --output-window 5 --bert-pretraining "bert-base-uncased" \
--bert-finetuning "true" --overwrite "false"
