import tensorflow as tf
from unet import get_unet
from utils import get_dataset


train_dataset

model = get_unet()
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

if(pretrained_weights):
	model.load_weights(pretrained_weights)
