#!/bin/bash

# shellcheck disable=SC1101
python gptonly/train.py --description "gpt" \
--learning-rate 0.00001 --batch-size 6  --output-window 5 --bert-pretraining "gpt2" \
--bert-finetuning "true" --overwrite "false"
