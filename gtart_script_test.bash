#!/bin/bash

# shellcheck disable=SC1101
python gbespoketart/train.py --description "using BertGeneration and finetuning decoder for token prediction" \
--learning-rate 0.0002 --batch-size 6  --output-window 5 --bert-pretraining "bert-base-uncased" \
--bert-finetuning "true" --evaluate "true" --load-model "true" --load-path "trained_model/2023-11-09:16-08-45"