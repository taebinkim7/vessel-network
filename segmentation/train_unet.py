import tensorflow as tf
from unet.unet import get_unet

train_dataset

model = get_unet()
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

if(pretrained_weights):
	model.load_weights(pretrained_weights)
