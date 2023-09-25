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
### Download Data ([HCRC Maptask](https://groups.inf.ed.ac.uk/maptask/transcripts/))
1. Retreive files
```
cd data/maptask
./maptask.wget.sh
```
Transcripts are located at `data/maptask/transcripts/`

### Train Model 
```
cd bespoketart
python train.py
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
