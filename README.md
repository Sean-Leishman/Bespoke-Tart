# Dissertation 

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
```
conda install pytorch pytorch-cuda=12.1 -c pytorch-nightly -c nvidia numpy transformers scikit-learn
pip install -r requirements .txt
```
### Download Data ([HCRC Maptask](https://groups.inf.ed.ac.uk/maptask/transcripts/))
1. Retreive files
```
cd data/maptask
./maptask.wget.sh
```
Transcripts are located at `data/maptask/transcripts/`

