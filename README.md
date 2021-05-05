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
First, save your training and validation data (images and masks) in `data/train_data` and `data/validation_data`, respectively.
To train the UNet model run
```
cd segmentation
python train_unet.py
```
Save test data (images) in `data/test_data`, then to generate predicted masks run
```
python test_unet.py
```
You can adjust hyperparameters in both scripts. Use the same `PATCH_SIZE` and `LOG_NUM`.

## Vessel Network Reconstruction & Network Feature Extraction
We reconstruct a vessel network using the predicted binary images in `data/test_data/predictions`. To construct a vessel network, run
```
cd ../feature_extraction
feature_extraction.py
```
The reconstructed network is saved in 'feature_extraction/feature' by 'imagename_network.png'.           
The above code also extracts some features and the features are saved in 'feature_extraction/feature' by 'imagename_alldata.xlsx' and 'imagename_degreedata.xlsx'.

## Vessel Network Analysis

