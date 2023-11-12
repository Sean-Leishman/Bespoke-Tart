#!/bin/bash

# shellcheck disable=SC1101
python gptonly/train.py --description "gptonly" \
--learning-rate 0.0001 --batch-size 6  --output-window 5 --bert-pretraining "gpt2" \
--bert-finetuning "true" --evaluate "true" --load-model "true" --load-path "trained_model/2023-11-10:18-52-29"
