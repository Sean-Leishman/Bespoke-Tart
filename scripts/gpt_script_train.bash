#!/bin/bash

# shellcheck disable=SC1101
python gptonly/train.py --description "increase switchboard size" \
--learning-rate 0.00002 --batch-size 4  --pretrained "gpt2" \
--finetune --cuda --datasets "switchboard" --speaker-tokens --projection-labels
