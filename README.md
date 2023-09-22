# BEspoke taRT  

## Steps
### Machine Guidance
Tested on a Debian-based system with versions as listed below. 

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

### Download Data ([HCRC Maptask](https://groups.inf.ed.ac.uk/maptask/transcripts/))
1. Retreive files
```
cd data/maptask
./maptask.wget.sh
```
Transcripts are located at `data/maptask/transcripts/`

### Train Model 
```
python train.py
```
