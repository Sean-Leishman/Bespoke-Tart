#!/bin/bash

# shellcheck disable=SC1101
python gptonly/train.py --description "gpt. increase lr to 1e-5. Change split to ~" \
--learning-rate 0.00002 --batch-size 4  --pretrained "gpt2" \
--finetune --cuda --datasets "switchboard" "fisher"
