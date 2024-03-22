# MT-HCCAR: Multi-Task Deep Learning with Hierarchical Classification and Attention-based Regression for Cloud Property Retrieval

The implementation of MT-HCCAR published at [Needs to add link after publication] in PyTorch.

## Table of contents
- [Table of contents](https://github.com/AI-4-atmosphere-remote-sensing/MT-HCCAR/tree/main?tab=readme-ov-file#table-of-contents)
- [Introduction](https://github.com/AI-4-atmosphere-remote-sensing/MT-HCCAR/tree/main?tab=readme-ov-file#introduction)
- [Architecture](https://github.com/AI-4-atmosphere-remote-sensing/MT-HCCAR/tree/main?tab=readme-ov-file#architecture)
- [Downloading Datasets](https://github.com/AI-4-atmosphere-remote-sensing/MT-HCCAR/tree/main?tab=readme-ov-file#downloading-datasets)
- [Usage](https://github.com/AI-4-atmosphere-remote-sensing/MT-HCCAR/tree/main?tab=readme-ov-file#usage)
  - [Installation](https://github.com/AI-4-atmosphere-remote-sensing/MT-HCCAR/tree/main?tab=readme-ov-file#installation)
  - [Training](https://github.com/AI-4-atmosphere-remote-sensing/MT-HCCAR/tree/main?tab=readme-ov-file#training)
  - [Prediction](https://github.com/AI-4-atmosphere-remote-sensing/MT-HCCAR/tree/main?tab=readme-ov-file#prediction)
- [Our Results](https://github.com/AI-4-atmosphere-remote-sensing/MT-HCCAR/tree/main?tab=readme-ov-file#our-results)

## Introduction
The retrieval of cloud properties, denoting the estimation of diverse characteristics of clouds through the analysis of data acquired from remote sensing instruments, plays an essential role in atmospheric and environmental studies. This research involves the identification of clouds, and the prediction of their phase (liquid or ice) and cloud optical thickness (COT), in satellite imagery. 

One practical motivation for our work is NASA’s Plankton, Aerosol, Cloud, ocean Ecosystem (PACE) mission [3], which will launch in 2024 and advance science in its eponymous disciplines. The primary satellite instrument of interest is the Ocean Color Instrument (OCI) [4] to be launched on the forthcoming NASA Plankton, Aerosol, Cloud, ocean Ecosystem (PACE) satellite in early 2024. However, the project also includes results for the Visible Infrared Imaging Radiometer (VIIRS) [6] and Advance Baseline Imager (ABI) [7] instruments that are the US' current operational polar-orbiting and geostationary imagers.

## Architecture

Our research objective involves the training of deep learning models to accomplish two tasks: 1) the classification of cloud mask and phase for each pixel based on its reflectance values, and 2) the subsequent prediction of Cloud Optical Thickness (COT) values for pixels classified as cloudy.
To address these objectives, we propose an end-to-end Multi-Task Learning (MTL) model, denoted as MT-HCCAR, which integrates a Hierarchical Classification network (HC) and a Classification-assisted with Attention-based Regression network (CAR). 
![hccar](https://github.com/AI-4-atmosphere-remote-sensing/MT-HCCAR/assets/90643297/4489a51f-2370-4088-99f9-88c630b31ff0)

MT-HCCAR: an end-to-end multi-task learning model with hierarchical classification (HC) and cross attention assisted regression (CAR). The HC sub-network consists of the cloud masking module $C_{Mask}(\cdot)$ and the cloud phase classification module $C_{Phase}(\cdot)$. The CAR sub-network consists of the auxiliary coarse classification module $C_{Aux}(\cdot)$, the cross attention module $A(\cdot)$, and the regression module $R(\cdot)$. On the right is the structure of $A(\cdot)$.

## Downloading Datasets
Please download the simulated dataset in '.nc' file from [this link](https://www.dropbox.com/scl/fi/9lnzwn4k3apzo2wcse0mn/rt_nn_cloud_training_data_20231016.nc?rlkey=3jqgl2uqq1ed8ndef24h2vzz1&dl=0).

## Usage
### Installation
1. Install PyTorch 1.13 or a newer version.
2. Clone this repository
3. Install required packages
```
pip install -r requirements.txt
```
### Training
Under the directory of 'MTHCCAR', run command:
```
python train.py
```
## Our Results
![scatter1_pressed](https://github.com/AI-4-atmosphere-remote-sensing/MT-HCCAR/assets/90643297/894ededd-568f-423f-a259-59013747f99f)


