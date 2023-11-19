# Dataset Managers

## TranscriptDM
Written originally for use in the classification style model where inputs are fed into a BERT
encoder and the output is fed into a linear layer. This was then extended for use with EncoderDecoder
models. 
As such both input and output sequences are tokenized and collected within __getitem__. The logic for 
extracting tokens are done here so preprocessing leaves the tokens within their respective conversations.

## GenerativeDM
Newest version of the data manager attempting to only output the tokenizer output for each dialog without 
specific labels or target text. 
For preprocessing the dialog for each conversation is tokenized and then each dialog is segmented into 
sequences of some max length where there is some overlap with the next extracted sequence.