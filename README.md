# Blood Vessel Network Analysis

This repository was created to implement a blood vessel network analysis.

## Setup
Using python 3.7.2, (e.g., `conda create -n vessel-network python=3.7.2`, then `conda activate vessel-network`) install the required packages by running 
```
conda activate vessel-network
pip install -r requirements.txt
```

## Blood Vessel Segmentation
The first step of our analysis is vessel segmentation.

Save training and validation data (images and masks) in `data/train_data` and `data/validation_data`, respectively.
To train the U-Net model run
```
cd segmentation
python train_unet.py
```
Our model can be trained on either GPU or CPU. CUDA version 11.0 or above and cuDNN version 8.0 or above are needed to train the model on GPU.
`conda install -c nvidia cudnn` will let you download the newest version of CUDA and cuDNN.

Save test data (images) in `data/test_data`, then to generate predicted masks run
```
python test_unet.py
```
You can adjust hyperparameters in both scripts but should use the same `PATCH_SIZE` and `LOG_NUM`. Pretrained weights are also available in `segmentation/ckpt_1`.

With the pretrained weights, the comparison plot of image, mask, and prediction patches for the validation data looks like
<img src="data/validation_data/comparison_plot.png" width="600">

## Vessel Network Reconstruction & Network Feature Extraction
In the next step, we reconstruct a vessel network using the predicted binary images in `data/test_data/predictions`. To reconstruct a vessel network, run
```
cd ../feature_extraction
feature_extraction.py
```
The reconstructed network is saved in `feature_extraction/feature` as `imagename_network.png`.           
The above code also extracts the features of the network (e.g., vessel length, branching point, vessel segment, tortuosity, etc.) and they are saved in `feature_extraction/feature` as `imagename_alldata.xlsx` and `imagename_degreedata.xlsx`.

## Vessel Network Analysis

