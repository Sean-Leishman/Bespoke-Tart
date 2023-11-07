# Models

## 2023-10-28:17-17-07
Main changes surround learning rate and weight loss increase from 0.0002 and weight loss decrease from 1.5
- Learning rate: 0.002
- Weight loss: 1.355

### Epoch 1
- f1: 0.7907
- average valid correct: 86.5683
- train loss: 0.6300

### Epoch 10

- f1: 0.7882
- average valid correct: 91.3770
- train loss: 0.6240

## 2023-10-28:17-58-17
Main changes is that `token_type_ids` are added to the model which show the speaker tokens of each part of the utterance.
Rechecking the model does not use `token_type_ids` in the `forward` method of the  model
### Epoch 1
- f1: 0.8190
- average valid correct: 90.3538
- train loss: 0.6299

### Epoch 10
- f1: 0.8207
- average valid correct: 91.1025
- train loss: 0.6238

## 2023-10-29:11-23-06
Do the change mentioned above properly

### Epoch 1
- f1: 0.8193
- average valid correct: 90.5478
- train loss: 0.6300

### Epoch 10
- f1: 0.8210
- average valid correct: 91.4216
- train loss: 0.6213

## 2023-10-29:14-48-03
Using BERT and finetuning its weights resulted in worse performance so perhaps this task is not suitable for finetuning a large LM and as such some multitask approach would be more suitable.

### Epoch 5
- f1: 0.8067
- average valid correct: 86.568
- train_loss: 0.6453

## 2023-10-29:22-28-13
Not finetuning BERT model. Using a smaller output window of size 5 so that classes are even. Turns out results as above were false as most predictions were 1 and the class labels were incorrect.
Added other metrics to examine these properties: precision and recall

### Epoch 9 (Highest Validation f1)
- f1: 0.5506
- precision: 0.5248
- recall: 0.5970
- train loss: 0.6571

## 2023-10-30:00-15-22
Using new data turn system where a context window of two turns is extracted for each index.

### Epoch 1
- f1: 0.5211
- precision: 0.5570
- recall 0.5069
- train loss 0.6671

### Epoch 7
- f1: 0.5107
- precision: 0.5704
- recall: 0.4790

## 2023-10-30:18-29-47
Fixed training data where continuous TRPs by the same speaker now form one utterance. As such number of turn shifts has decreased massively

### Epoch 1 
- f1: 0.2368
- precision: 0.2738
- recall: 0.2670
- train loss: 0.9608

### Epoch 2
- f1: 0.2834
- precision: 0.3220
- recall: 0.2856
- train-loss: 0.9359


## 2023-10-30:19-50-35
Fixed token-type-ids bug so now the token_type_ids correctly correspons to the speaker of token X. 
Also added additional bAcc metric ((TPR+FPR)/2) used within TurnGPT. 
High class weight of 5.0 

### Epoch 1
- f1: 0.3385
- accuracy: 0.4133
- precision: 0.2154
- recall: 0.8347 
- bAcc: 0.5768

### Epoch 10
- f1: 0.3398
- accuracy: 0.5096
- precision: 0.2334
- bAcc: 0.5973

### Epoch 8
- f1: 0.3482
- accuracy: 0.6345
- precision: 0.2617
- recall: 0.5475
- bAcc: 0.6007




