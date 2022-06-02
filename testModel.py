import numpy as np
import tensorflow as tf
import os
from keras import backend as K
import skimage.io as io
import matplotlib.pyplot as plt
import skimage.transform as trans
from Unet import Unet

import cv2
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou


if __name__=="__main__":
    
    model = Unet(model_path=True)
    image = io.imread(os.path.join(os.getcwd(), "testDataset", "test3.jpeg"))
    image = trans.resize(image, (256, 256))
    expImage = np.expand_dims(image, axis=0) 
    test = model.predict(expImage)[0,:,:,0]
    test = (test*255).astype("uint8")
    
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.xlabel("Original")
    image = (255*image).astype("uint8")
    camera_mode = True
    if camera_mode:
            hsvframe = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h,s,v =cv2.split(hsvframe)
            seg = cv2.bitwise_and(v, test)
            seg = cv2.merge([h,s,seg])
            seg = cv2.cvtColor(seg, cv2.COLOR_HSV2RGB)
    plt.subplot(1,2,2)
    plt.imshow(seg, cmap="gray")
    plt.xlabel("Labeling")
    
    plt.show()
    