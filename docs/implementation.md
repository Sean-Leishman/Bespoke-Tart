# Implementation

## TODO
- [x] Download data - 19/9/2023
- [x] Organise Dataset and Dataloader classes - 20/9/2023
- [x] Have some sort of default class structure for training model - 20/9/2023
- [x] Reparse data in order to have each speaker's utterance be a single sentence 
- [x] Within dataset separate speaker utterances so that non turn shift utterances are used 
- [ ] Implement other baselines and embeddings 
- [ ] Implement a thourough classifier 
- [ ] Use other datasets 
- [ ] Develop an evaluation framework 
    - [x] Create a validation/test function
    - [ ] Use metrics aside from accuracy
- [ ] Generate documentation as experiments will be carried out
- [x] Save and load model checkpoints for stopping training 
- [ ] Implement validation to determine threshold of classification
- [ ] Put more consideration into choice of learning rate 
- [ ] Show results within ipynb via graphs
- [ ] Integrate some system that separates model results based on date/type -> could just be done manually or by default with datetime. 

## Data
### HCRC
1. Download data to transcripts
2. Write script to generate train.txt, text.txt with indexes of appropriate files shuffled

### Taskmaster
1. use hugging face to download data as is with TurnGPT as a guide

## Notes
### 25/9/2023
Currently running the model for 10 epochs with batch size of 128 and learning rate 0.0002. 
Achieves (format: START->END):
- train_loss=0.6654->0.6198
- validation_loss=0.6550->0.6149
- validation_correct=0.713654->0.90 

Appears to be a good level of prediction. Not a huge amount of loss improvement.
Have to compare against some baseline or distribution-based approximation. 
Threshold of prediction could be improved through validation 
Data is currently split into a ~0.7:1 ratio of nontrp to trp but in real life 
nontrps would be the vast majority of samples 

Architecture: BERT -> Fully Connected Layer (BERT hidden, 1) -> Sigmoid (0,1) 
