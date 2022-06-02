from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import os
from keras import backend as K
import skimage.transform as trans
import skimage.io as io
from skimage import color
from keras import activations
from dataset_resizer import maximum_file

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou



def Unet(model_path=False):
    def iou_coef(y_true, y_pred, smooth=1):
        intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
        union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
        iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
        return iou
    
    in_put = tf.keras.layers.Input((256, 256, 3))
    
    conv_1 = tf.keras.layers.Conv2D(32, 3, padding="same", activation= "relu", kernel_initializer="he_normal")(in_put)
    batch_1 = tf.keras.layers.BatchNormalization()(conv_1)
    conv_2 = tf.keras.layers.Conv2D(32, 3, padding="same", activation= "relu", kernel_initializer="he_normal")(batch_1)
    batch_2 = tf.keras.layers.BatchNormalization()(conv_2)
    add_1 = tf.add(batch_1, batch_2)
    activation_1 = activations.relu(add_1)
    residual_1 = tf.add(conv_1, activation_1)
    pool_1 = tf.keras.layers.MaxPooling2D()(residual_1)

    conv_3 = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", kernel_initializer="he_normal")(pool_1)
    batch_3 = tf.keras.layers.BatchNormalization()(conv_3)
    conv_4 = tf.keras.layers.Conv2D(64, 3, padding="same", activation= "relu", kernel_initializer="he_normal")(batch_3)
    batch_4 = tf.keras.layers.BatchNormalization()(conv_4)
    add_2 = tf.add(batch_3, batch_4)
    activation_2 = activations.relu(add_2)
    residual_2 = tf.add(conv_3, activation_2)
    pool_2 = tf.keras.layers.MaxPooling2D()(residual_2)
    
    conv_5 = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu", kernel_initializer="he_normal")(pool_2)
    batch_5 = tf.keras.layers.BatchNormalization()(conv_5)
    conv_6 = tf.keras.layers.Conv2D(128, 3, padding="same", activation= "relu", kernel_initializer="he_normal")(batch_5)
    batch_6 = tf.keras.layers.BatchNormalization()(conv_6)
    add_3 = tf.add(batch_5, batch_6)
    activation_3 = activations.relu(add_3)
    residual_3 = tf.add(conv_5, activation_3)
    pool_3 = tf.keras.layers.MaxPooling2D()(residual_3)

    conv_7 = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu", kernel_initializer="he_normal")(pool_3)
    batch_7 = tf.keras.layers.BatchNormalization()(conv_7)
    conv_8 = tf.keras.layers.Conv2D(256, 3, padding="same", activation= "relu", kernel_initializer="he_normal")(batch_7)
    batch_8 = tf.keras.layers.BatchNormalization()(conv_8)
    add_4 = tf.add(batch_7, batch_8)
    activation_4 = activations.relu(add_4)
    residual_4 = tf.add(conv_7, activation_4)
    pool_4 = tf.keras.layers.MaxPooling2D()(residual_4)
    
    conv_9 = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu", kernel_initializer="he_normal")(pool_4)
    batch_9 = tf.keras.layers.BatchNormalization()(conv_9)
    conv_10 = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu", kernel_initializer="he_normal")(batch_9)
    batch_10 = tf.keras.layers.BatchNormalization()(conv_10)
    add_5 = tf.add(batch_9, batch_10)
    activation_5 = activations.relu(add_5)
    residual_5 = tf.add(conv_9, activation_5)
    drop_out1 = tf.keras.layers.Dropout(0.1)(residual_5)

    up_sample1 = tf.keras.layers.UpSampling2D()(drop_out1)
    concat1 = tf.keras.layers.concatenate([up_sample1, residual_4])
    conv_11 = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu", kernel_initializer="he_normal")(concat1)
    batch_11 = tf.keras.layers.BatchNormalization()(conv_11)
    conv_12 = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu", kernel_initializer="he_normal")(batch_11)
    batch_12 = tf.keras.layers.BatchNormalization()(conv_12)
    add_6 = tf.add(batch_11, batch_12)
    activation_6 = activations.relu(add_6)
    residual_6 = tf.add(conv_11, activation_6)
    drop_out2 = tf.keras.layers.Dropout(0.1)(residual_6)

    up_sample2 = tf.keras.layers.UpSampling2D()(drop_out2)
    concat2 = tf.keras.layers.concatenate([up_sample2, residual_3])
    conv_13 = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu", kernel_initializer="he_normal")(concat2)
    batch_13 = tf.keras.layers.BatchNormalization()(conv_13)
    conv_14 = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu", kernel_initializer="he_normal")(batch_13)
    batch_14 = tf.keras.layers.BatchNormalization()(conv_14)
    add_7 = tf.add(batch_13, batch_14)
    activation_7 = activations.relu(add_7)
    residual_7 = tf.add(conv_13, activation_7)
    drop_out3 = tf.keras.layers.Dropout(0.1)(residual_7)

    up_sample3 = tf.keras.layers.UpSampling2D()(drop_out3)
    concat3 = tf.keras.layers.concatenate([up_sample3, residual_2])
    conv_15 = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", kernel_initializer="he_normal")(concat3)
    batch_15 = tf.keras.layers.BatchNormalization()(conv_15)
    conv_16 = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", kernel_initializer="he_normal")(batch_15)
    batch_16 = tf.keras.layers.BatchNormalization()(conv_16)
    add_8 = tf.add(batch_15, batch_16)
    activation_8 = activations.relu(add_8)
    residual_8 = tf.add(conv_15, activation_8)
    drop_out4 = tf.keras.layers.Dropout(0.1)(residual_8)

    up_sample4 = tf.keras.layers.UpSampling2D()(drop_out4)
    concat4 = tf.keras.layers.concatenate([up_sample4, residual_1])
    conv_17 = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", kernel_initializer="he_normal")(concat4)
    batch_17 = tf.keras.layers.BatchNormalization()(conv_17)
    conv_18 = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", kernel_initializer="he_normal")(batch_17)
    batch_18 = tf.keras.layers.BatchNormalization()(conv_18)
    add_9 = tf.add(batch_17, batch_18)
    activation_9 = activations.relu(add_9)
    residual_9 = tf.add(conv_17, activation_9)
    drop_out5 = tf.keras.layers.Dropout(0.1)(residual_9)
    out_put = tf.keras.layers.Conv2D(1, 1 , padding="same", activation="sigmoid", kernel_initializer="he_normal")(drop_out5)

    model = tf.keras.Model(in_put, out_put)

    model.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.0001), loss = 'binary_crossentropy', metrics = ['accuracy', iou_coef])

    if model_path:
        model.load_weights(os.path.join(os.getcwd(), 'models' ,"Unet_tyoon4.h5"))
        return model
    return model

def y_testGenerator(test_path,num_image = 1, target_size = (256,256), flag_multi_class = False, as_gray = True, dimension3D=False):
    for i in range(num_image):
        try:
            img = io.imread(os.path.join(test_path,f"{i}.jpeg"))
        except:
            continue
        try:
            img = trans.resize(img, target_size)
        except:
            print("Resize Error")
        if dimension3D:
            try:
                img = color.rgb2gray(img)
            except:
                img = img
        yield img

def x_testGenerator(test_path,num_image = 1, target_size = (256,256), flag_multi_class = False, as_gray = True, dimension3D=False):
    for i in range(num_image):
        try:
            img = io.imread(os.path.join(test_path,f"{i}.jpeg"))
        except:
            continue
        try:
            img = trans.resize(img, target_size)
        except:
            print("Resize Error")
        if dimension3D:
            img = color.gray2rgb(img)
        yield img

if __name__=="__main__":
    num = maximum_file(True)
    print("파일 개수: ", num)
    train_x = np.array(list(x_testGenerator(os.path.join(os.getcwd(), "x_resize"), num)), "float32")
    train_y = np.array(list(y_testGenerator(os.path.join(os.getcwd(), "y_resize"), num, dimension3D = True)), "float32")
    print(train_x.shape)
    train_y = np.expand_dims(train_y, axis=-1)
    print(train_y.shape)
    model = Unet(model_path=False)
    history = model.fit(train_x, train_y, epochs= 370, batch_size=16)
    plt.plot(history.epoch, history.history["iou_coef"])
    plt.plot(history.epoch, history.history["loss"])
    plt.plot(history.epoch, history.history["accuracy"])
    plt.xlabel("Epochs")
    plt.savefig(os.path.join(os.getcwd(), "graph", "graph.jpeg"))
    model.save(os.path.join(os.getcwd(), 'models', 'Unet_tyoon5.h5'))
    