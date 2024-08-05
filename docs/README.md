# Spike-to-excite: photosensitive seizures in biologically-realistic spiking neural networks
This repository contains the code of the spike-to-excite project and the code for reproducing all the results reported in our paper.

## Installing dependencies
Install all required dependencies and activate the spiketoexcite environment using conda.
```
conda env create -f environment.yml
conda activate spiketoexcite
```

## Downloading pre-trained models and results
The pre-trained model, experimental data and pre-computed results can be found under the releases of this repository. Make sure to place the downloaded `data` directory in the project path. Note: You will still need to download the natural stimulus dataset (see below).

## The pre-trained model
In our work, we extended the spiking V1 model from [(Taylor et al. 2024)](https://www.biorxiv.org/content/10.1101/2024.05.12.593763v1) with weakened inhibitory connections. The pre-trained model files are contained in `data/prediction_0.0017782794100389228_0.3_0.1_0_17_0.2_0.5`.

## The datasets
We used three different datasets in our evaluations:
1. Provocative stimuli, which can be found in `data/vids`.
2. Star wars clips, which can be found at `data/starwars_patches.pt`.
3. Natural stimulus dataset, can be downloaded from: https://figshare.com/articles/dataset/Natural_movies/24265498. You can place this file anywhere on your machine, but need to define this path at the beginning of some notebooks.

## Downloading pre-computed results
The motion tuning, contrast tuning and stimulation results are found in the `data/tuning`, `data/mechanisms` and `data/neurostimulation` directories respectively.

## Reproducing paper results
All the figures in the paper can be reproduced using the notebooks found in the ```notebooks``` directory.
