# Models

Mimicing TurnGPT with only Switchboard dataset.

## TODO
- Try new learning rate increases
- Experiment learning rates with TurnGPT and see if similar results are achieved by only using Switchboard
- Find out if handling of punctuation is correct as the TRP graph shows punctuation tokens
- Check whether turn shift token handling is correct as loss is extremely high for using pretrained weights and the output is only the newly added token
- Maybe add Fisher 
- Use the same dataset with BERT as well 
 
## 2023-11-10:17-12-09
Epoch 1:
- train loss: 5.6191

## 2023-11-19
Large amount of changes in order to align with TurnGPT:
- Changed dataset generation so that the same max length of input is used without considering output
- Updated step between consecutive sequences as previously each index corresponded to the next token in a sequence now consider overla of 10 tokens between two sequences
- Using Wandb to plot results and plot probability of TRP
- Updated tokenization such that `eos_token` is now `[SEP]` for a turn shift and a `pad_token` is `<end_of_text>`.
- 