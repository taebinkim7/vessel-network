# Blood Vessel Network Analysis

This repository was created to implement a blood vessel network analysis.

## Setup
Create a venv, e.g., `conda create -n vessel-network python=3.7.2`.
Then activate the venv and install the required packages using 
```
conda activate vessel-network
pip install -r requirements.txt
```

## Blood Vessel Segmentation
First, save training and validation data (images and masks) in `data/train_data` and `data/validation_data`, respectively.
To train the UNet model run
```
cd segmentation
python train_unet.py
```
Save test data (images) in `data/test_data`, then to generate predicted masks run
```
python test_unet.py
```
One can adjust hyperparameters in both scripts but should use the same `PATCH_SIZE` and `LOG_NUM`. Pretrained weights are also available in `segmentation/ckpt_1`.

After training the UNet with the current hyperparameters, we get the following results:  

The comparison plot for the train data looks like
![Comparison plot for train data](data/train_data/comparison_plot.png | width = 70)
<img src="data/train_data/comparison_plot.png" width="300">

, and the one for the validation data looks like
![Comparison plot for validation data](data/validation_data/comparison_plot.png)

## Vessel Network Reconstruction & Network Feature Extraction
We reconstruct a vessel network using the predicted binary images in `data/test_data/predictions`. To construct a vessel network, run
```
cd ../feature_extraction
feature_extraction.py
```
The reconstructed network is saved in `feature_extraction/feature` as `imagename_network.png`.           
The above code also extracts the features of the network (e.g., vessel length, branching point, vessel segment, tortuosity, etc.) and they are saved in `feature_extraction/feature` as `imagename_alldata.xlsx` and `imagename_degreedata.xlsx`.

## Vessel Network Analysis

