# BEspoke taRT  

## Machine Guidance
Tested on a Debian-based system with versions as listed below. 

## General Installation 
### Prerequisties
1. `conda`

### Setup Envrionment
```
conda create -n diss python=3
conda activate diss
```
### Install Dependencies
If a GPU is avaialble: 
```
conda install pytorch pytorch-cuda=12.1 -c pytorch-nightly -c nvidia 
pip install -r requirements .txt
```
If a GPU is not available:
``` 
conda install pytorch torchvision torchaudio -c pytorch -c conda-forge
pip install -r requirements .txt
```

### Install Data managing package 
```
cd data && pip install -e . 
```

### Install BespokeTart as a module
```
pip install -e . 
```

## Training the Model 
### Download Data 
#### EDACC - The Edinburgh Interational Accents of English Corpus 
Can be found [here](https://groups.inf.ed.ac.uk/edacc/) where the entire corpus dataset can be downloaded. 
If you want to do so automatically please run the following from the project root directory.
```
mkdir data/edacc
cd data/edacc/ && ./edacc.wget.sh
```

For more information on the structure of the data see [here](https://github.com/Sean-Leishman/Bespoke-Tart/data/edacc/README.md)

#### Switchboard
Was collected in a licensed manner so data is distributed privately

For more information on the structure of the data see [here](https://github.com/Sean-Leishman/Bespoke-Tart/data/switchboard/README.md)


#### HCRC MapTask 
([HCRC Maptask](https://groups.inf.ed.ac.uk/maptask/transcripts/))
1. Retreive files
```
cd data/maptask && ./maptask.wget.sh
```
Transcripts are located at `data/maptask/transcripts/`edacc

For more information on the structure of the data see [here](https://github.com/Sean-Leishman/Bespoke-Tart/data/maptask/README.md)

### Train Model 
```
cd bespoketart
python train.py
```
For more informative instructions and arguments available run
```
python train.py --help
```

```
usage: train.py [-h] [--cuda CUDA] [--load-model LOAD_MODEL] [--load-path LOAD_PATH] [--save-path SAVE_PATH]
                [--bert-finetuning BERT_FINETUNING] [--bert-pretraining BERT_PRETRAINING] [--epoch-size EPOCH_SIZE] [--batch-size BATCH_SIZE]
                [--learning-rate LEARNING_RATE] [--early-stop EARLY_STOP]

Bespoke Tart model used to predict turn-taking from linguistic features

options:
  -h, --help            show this help message and exit
  --cuda CUDA           true/false if cuda should be enabled
  --load-model LOAD_MODEL
                        true/false if model should be loaded
  --load-path LOAD_PATH
                        load model config and weights from this file and ignore input configurations
  --save-path SAVE_PATH
                        model weights and config options save directory
  --bert-finetuning BERT_FINETUNING
                        true/false if BERT should be finetuned
  --bert-pretraining BERT_PRETRAINING
                        name of pretrained BERT model
  --epoch-size EPOCH_SIZE
  --batch-size BATCH_SIZE
  --learning-rate LEARNING_RATE
  --early-stop EARLY_STOP
                        number of iterations without improvement for early stop

```

## Running ipynb Analysis
```
# Primarily suitable for use in the base envrionment
conda activate base
conda install jupyter_notebook
conda decativate

# If torch etc. is installed in another envrionment
conda activate diss
conda install ipykernel

cd nb_notebooks
jupyter notebook
```
