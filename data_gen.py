#Required imports 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import array_to_img, img_to_array, load_img



#Writing code to generate the images 
datagen = ImageDataGenerator(
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0,
    horizontal_flip=True,
    fill_mode='nearest'
)
output_height = 416
output_width = 416
img = load_img('file path to the image you want to augment', target_size = (output_height,output_width))          #File path of image which is to be loaded
x = img_to_array(img)       #This is a numpy array with shape ()
x = x.reshape((1,) + x.shape)       #This is a numpy array with shape(1, )

#The .flow() command will generate batches of randomly transformed images and it saves the results in a specific directory

i=0
for batch in datagen.flow(x, batch_size=1, save_to_dir='preview', save_prefix='bottle', save_format='jpeg'):
    i += 1
    if i>1000:
        break               #Or else the generator would loop indefinitely