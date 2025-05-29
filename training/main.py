from keras.utils import normalize
import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt


#Resizing images, if needed
SIZE_X = 256 
SIZE_Y = 256
n_classes=4 #Number of classes for segmentation

#Capture training image info as a list
train_images = []

for directory_path in glob.glob(r"M:/499/Dataset/Processed/Processed/L/L/t2_SPACE/All/images"):
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, 0)       
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        train_images.append(img)
       
#Convert list to array for machine learning processing        
train_images = np.array(train_images)

#Capture mask/label info as a list
train_masks = [] 
for directory_path in glob.glob(r"M:/499/Dataset/Processed/Processed/L/L/t2_SPACE/All/masks"):
    for mask_path in glob.glob(os.path.join(directory_path, "*.png")):
        mask = cv2.imread(mask_path, 0)       
        mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
        train_masks.append(mask)
        
#Convert list to array for machine learning processing          
train_masks = np.array(train_masks)


###############################################
#Encode labels... but multi dim array so need to flatten, encode and reshape
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
n, h, w = train_masks.shape
train_masks_reshaped = train_masks.reshape(-1,1)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

np.unique(train_masks_encoded_original_shape)

#################################################
train_images = np.expand_dims(train_images, axis=3)
train_images = normalize(train_images, axis=1)

train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)



#Create a subset of data for quick testing
#Picking 10% for testing and remaining for training
from sklearn.model_selection import train_test_split
# X1, X_test, y1, y_test = train_test_split(train_images, train_masks_input, test_size = 0.10, random_state = 0)

# #Further split training data t a smaller subset for quick testing of models
# X_train, X_do_not_use, y_train, y_do_not_use = train_test_split(X1, y1, test_size = 0.2, random_state = 0)
# Splitting 10% for testing
X_train, X_test, y_train, y_test = train_test_split(train_images, train_masks_input, test_size=0.10, random_state=0)
print("Class values in the dataset are ... ", np.unique(y_train))  # 0 is the background/few unlabeled 

from keras.utils import to_categorical
train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))

test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))


################ Class Weight###################
# Compute class weights
from sklearn.utils.class_weight import compute_class_weight

# Define classes and labels correctly for the compute_class_weight function
classes = np.unique(train_masks_encoded_original_shape)  # unique classes in the mask
labels = train_masks_encoded_original_shape.flatten()  # labels from the masks

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)

# Convert class weights to a dictionary to use with training functions
class_weights_dict = {cls: weight for cls, weight in zip(classes, class_weights)}
print("Class weights dictionary:", class_weights_dict)

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]

 



################################################################
def multi_unet_model(n_classes=4, IMG_HEIGHT=SIZE_X, IMG_WIDTH=SIZE_Y, IMG_CHANNELS=1):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    #NOTE: Compile the model in the main program to make it easy to test with various loss functions
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    #model.summary()
    
    return model

####################################### MODEL SAN ###############################################

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose, BatchNormalization
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Activation
from tensorflow.keras.models import Model

def upsample_block(x, filters, kernel_size=(3, 3), padding='same', strides=1):
    x = Conv2DTranspose(filters, kernel_size, padding=padding, strides=strides)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)  # Add Activation here
    return x

def multi_unet_model(n_classes=4, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    # Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Extra layer with 512 channels
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    c6 = BatchNormalization()(c6)
    c6 = Dropout(0.3)(c6)
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    # Expansive path
    u6 = upsample_block(c6, 256, strides=(2, 2))
    u6 = concatenate([u6, c4])
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c7 = BatchNormalization()(c7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u7 = concatenate([u7, c3])
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c8 = BatchNormalization()(c8)
    c8 = Dropout(0.2)(c8)
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u8 = concatenate([u8, c2])
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c9 = BatchNormalization()(c9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c9)
    u9 = concatenate([u9, c1], axis=3)
    c10 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c10 = BatchNormalization()(c10)
    c10 = Dropout(0.1)(c10)
    c10 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c10)

    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c10)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model

#######################################################################################################



############################## MODEL SAN 3 ############################################################

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose, BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.callbacks import ReduceLROnPlateau

def upsample_block(x, filters, kernel_size=(3, 3), padding='same', strides=1):
    x = Conv2DTranspose(filters, kernel_size, padding=padding, strides=strides)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)  # LeakyReLU activation
    return x

def multi_unet_model2(n_classes=4, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    # Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(s)  # Glorot uniform initializer
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(c1)  # Glorot uniform initializer
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(p1)  # Glorot uniform initializer
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(c2)  # Glorot uniform initializer
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(p2)  # Glorot uniform initializer
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(c3)  # Glorot uniform initializer
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(p3)  # Glorot uniform initializer
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(c4)  # Glorot uniform initializer
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(p4)  # Glorot uniform initializer
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(c5)  # Glorot uniform initializer

    # Extra layer with 512 channels
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(c5)  # Glorot uniform initializer
    c6 = BatchNormalization()(c6)
    c6 = Dropout(0.3)(c6)
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(c6)  # Glorot uniform initializer

    # Expansive path
    u6 = upsample_block(c6, 256, strides=(2, 2))
    u6 = concatenate([u6, c4])
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(u6)  # Glorot uniform initializer
    c7 = BatchNormalization()(c7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(c7)  # Glorot uniform initializer

    u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u7 = concatenate([u7, c3])
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(u7)  # Glorot uniform initializer
    c8 = BatchNormalization()(c8)
    c8 = Dropout(0.2)(c8)
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(c8)  # Glorot uniform initializer

    u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u8 = concatenate([u8, c2])
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(u8)  # Glorot uniform initializer
    c9 = BatchNormalization()(c9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(c9)  # Glorot uniform initializer

    u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c9)
    u9 = concatenate([u9, c1], axis=3)
    c10 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(u9)  # Glorot uniform initializer
    c10 = BatchNormalization()(c10)
    c10 = Dropout(0.1)(c10)
    c10 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(c10)  # Glorot uniform initializer

    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c10)

    model = Model(inputs=[inputs], outputs=[outputs])



    return model 



#######################################################################################################



def get_model():
    return multi_unet_model2(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)




import keras.backend as K
import tensorflow as tf
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose
from keras.metrics import binary_accuracy
from keras.metrics import binary_crossentropy
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda




#####################################COMBINED LOSS##################################
def focal_loss(y_true, y_pred, gamma=2.0):
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    cross_entropy = -y_true * tf.math.log(y_pred)
    loss = tf.pow(1 - y_pred, gamma) * cross_entropy
    return tf.reduce_mean(loss, axis=-1)

def soft_dice_coefficient(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred)
    dice_coefficient = (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)
    return dice_coefficient

def soft_dice_loss(y_true, y_pred):
    return 1 - soft_dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred, gamma=2.0, alpha=0.5):
    focal = focal_loss(y_true, y_pred, gamma)
    dice = soft_dice_loss(y_true, y_pred)
    return alpha * focal + (1 - alpha) * dice


import tensorflow as tf
from keras.metrics import MeanIoU

# Create MeanIoU object outside the function
n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)

@tf.function
def mean_iou(y_true, y_pred):
    # Update the state of IOU_keras inside the function
    IOU_keras.update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1))
    return IOU_keras.result()

#####################################COMBINED LOSS 2##################################

import tensorflow as tf

# def focal_loss(y_true, y_pred, gamma=4.0, class_weights=None):
#     epsilon = tf.keras.backend.epsilon()
#     y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
#     cross_entropy = -y_true * tf.math.log(y_pred)
    
#     if class_weights is not None:
#         # Retrieve class weights based on the true labels
#         weights = tf.reduce_sum(class_weights * y_true, axis=-1)
#         cross_entropy *= weights

#     loss = tf.pow(1 - y_pred, gamma) * cross_entropy
#     return tf.reduce_mean(loss, axis=-1)

def focal_loss(y_true, y_pred, gamma=4.0, class_weights=None):
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    cross_entropy = -y_true * tf.math.log(y_pred)
    
    # if class_weights is not None:
    #     # Convert class weights dictionary to a list or array
    #     weights = [class_weights[i] for i in range(len(class_weights))]
    #     weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        
    #     # Expand dimensions of weights tensor to match shape of cross_entropy tensor
    #     weights = tf.expand_dims(weights, axis=0)  # Add a singleton dimension at the beginning

    #     cross_entropy *= weights

    loss = tf.pow(1 - y_pred, gamma) * cross_entropy
    return tf.reduce_mean(loss, axis=-1)




def soft_dice_coefficient(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    sum_true = tf.reduce_sum(y_true, axis=(1, 2, 3))
    sum_pred = tf.reduce_sum(y_pred, axis=(1, 2, 3))
    dice_coefficient = (2. * intersection + smooth) / (sum_true + sum_pred + smooth)
    return tf.reduce_mean(dice_coefficient)

import tensorflow as tf

def dice_coefficient(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2))  # Sum along spatial dimensions (height and width)
    sum_true = tf.reduce_sum(y_true, axis=(1, 2))
    sum_pred = tf.reduce_sum(y_pred, axis=(1, 2))
    dice_coefficient = (2. * intersection + smooth) / (sum_true + sum_pred + smooth)
    return tf.reduce_mean(dice_coefficient)






def soft_dice_loss(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    sum_true = tf.reduce_sum(y_true, axis=(1, 2, 3))
    sum_pred = tf.reduce_sum(y_pred, axis=(1, 2, 3))
    dice_coefficient = (2. * intersection + smooth) / (sum_true + sum_pred + smooth)
    dice_loss = 1 - dice_coefficient
    return tf.reduce_mean(dice_loss)


# Class weights dictionary
class_weights = {0: 0.29509034665527933, 1: 11.184575419074891, 2: 2.5614765838968205, 3: 7.610426880436339}


def combined_loss(y_true, y_pred, gamma=4.0, alpha=0.7, class_weights=class_weights):
    focal = focal_loss(y_true, y_pred, gamma, class_weights)
    dice = soft_dice_loss(y_true, y_pred)
    return alpha * focal + (1 - alpha) * dice
#########      With class Weight         ################################
# def combined_loss(y_true, y_pred, gamma=None, alpha=None, class_weights=None):
#     # Set default values for gamma, alpha, and class_weights
#     if gamma is None:
#         gamma = 2.0
#     if alpha is None:
#         alpha = 0.5
#     if class_weights is None:
#         class_weights = {0: 0.29509034665527933, 1: 11.184575419074891, 2: 2.5614765838968205, 3: 7.610426880436339}

#     focal = focal_loss(y_true, y_pred, gamma, class_weights)
#     dice = soft_dice_loss(y_true, y_pred)
#     return alpha * focal + (1 - alpha) * dice

# Assuming you already defined your class_weights dictionary, soft_dice_loss, and focal_loss functions



# import tensorflow as tf
# from keras.metrics import MeanIoU

# # Create MeanIoU object outside the function
# n_classes = 4
# IOU_keras = MeanIoU(num_classes=n_classes)
# #IOU_keras = MeanIoU(num_classes=n_classes, name='mean_iou')

# @tf.function
# def mean_iou(y_true, y_pred):
#     # Update the state of IOU_keras inside the function
#     IOU_keras.update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1))
#     return IOU_keras.result()




import tensorflow as tf


class CustomMeanIoU(tf.keras.metrics.MeanIoU):
   
    def __init__(self, num_classes=None, name=None, dtype=None):
        super(CustomMeanIoU, self).__init__(
            num_classes=num_classes, name=name, dtype=dtype
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.math.argmax(y_true, axis=-1)
        y_pred = tf.math.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

custom_mIoU_metric = CustomMeanIoU(num_classes=4, name='mean_iou')     
    

    
    
    
   

model = get_model()
# Compile the model with Focal Loss and both Accuracy and Dice Coefficient as metrics
#model.compile(optimizer=Adam(), loss=combined_loss, metrics=['accuracy', soft_dice_coefficient])
model.compile(optimizer=Adam(learning_rate=0.001), loss=combined_loss, metrics=['accuracy', soft_dice_coefficient, custom_mIoU_metric])

# Print model summary
model.summary()

# Define Early Stopping to monitor 'val_dice_coefficient' and restore the best weights
early_stopping = EarlyStopping(monitor='val_mean_iou', patience=20, mode='max', restore_best_weights=True)

# Define Model Checkpoint to save the best model based on 'val_dice_coefficient'
model_checkpoint = ModelCheckpoint('best_modeel1_t1_600Epochs.keras', monitor='val_mean_iou', mode='max', save_best_only=True)

# Train the model with Early Stopping and Model Checkpoint
history = model.fit(X_train, y_train_cat, 
                    batch_size=1, 
                    verbose=1, 
                    epochs=100, 
                    validation_data=(X_test, y_test_cat), 
                    callbacks=[early_stopping, model_checkpoint], 
                    shuffle=False)


# Save the trained model in HDF5 format
model.save('saved_model1_test_600Epochs.hdf5')

#########Model Loading

from keras.models import load_model

# Define the custom objects
custom_objects = {
    'combined_loss': combined_loss,
    'soft_dice_coefficient': soft_dice_coefficient,
    'mean_iou': mean_iou
}

#Load Model 
file_path = 'M:/499/Dataset/Processed/Lumber Spine all/Pre-process/Dataset-20240326T154540Z-001/Dataset/Nahid/saved_model2.hdf5'
model = load_model(file_path, custom_objects=custom_objects)
# Recreate the architecture
model = get_model()  # make sure this reconstructs your model architecture exactly
model.load_weights('M:/499/Dataset/Processed/Processed/L/L/t2_SPACE/Kaggle/MODEL SAN2/Batch 8/700 Epochs/saved_model1_t2_SPACE_Kaggle_700eo_gamma_2_alpha_0.5_MODEL_SAN2.keras')  # just load weights
model.compile(optimizer=Adam(learning_rate=0.001), loss=combined_loss, metrics=['accuracy', soft_dice_coefficient, custom_mIoU_metric])


# Evaluate the model on the test data
evaluation_results = model.evaluate(X_test, y_test_cat)

# Unpack the evaluation results
loss = evaluation_results[0]
accuracy = evaluation_results[1]
dice_coefficient = evaluation_results[2]

# Print the evaluation metrics
print("Loss:", loss)
print("Accuracy:", accuracy)
print("Dice Coefficient:", dice_coefficient)


# # Load the best model weights
# best_model = load_model('best_model.keras', custom_objects={'focal_loss': focal_loss, 'dice_coefficient': dice_coefficient})

# # Evaluate the best model on the test data
# _, _, best_dice_coeff = best_model.evaluate(X_test, y_test_cat)

# # Print the best dice coefficient
# print("Best Dice Coefficient:", best_dice_coeff)



import random
import matplotlib.pyplot as plt

# Define the threshold
threshold = 0.2

# Select a random test image
test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth = y_test[test_img_number]
test_img_norm = test_img[:, :, 0]  # Assuming the image is grayscale

# Make prediction
prediction = model.predict(np.expand_dims(test_img, axis=0))
predicted_img = np.argmax(prediction, axis=-1)[0]  # Assuming channel dimension is the last axis

# Apply threshold
# predicted_img_thresholded = (prediction[..., 1] > threshold).astype(np.uint8)  # Assuming you have 2 classes and you want to threshold class 1

# Plotting
plt.figure(figsize=(16, 12))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img_norm, cmap='gray')  # Display the grayscale image
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:, :, 0], cmap='viridis')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img, cmap='viridis') #viridis, gray, inferno, plasma, coolwarm, cividis, magma
#plt.subplot(234)
# plt.title('Thresholded Prediction')
# plt.imshow(predicted_img_thresholded, cmap='gray')
plt.show()

#IOU
y_pred=model.predict(X_test)
y_pred_argmax=np.argmax(y_pred, axis=3)
#Using built in keras function
from keras.metrics import MeanIoU
n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_test[:,:,:,0], y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

# Calculate IoU for each class
iou_classes = []
for i in range(n_classes):
    intersection = np.sum((y_test[:,:,:,0] == i) & (y_pred_argmax == i))
    union = np.sum((y_test[:,:,:,0] == i) | (y_pred_argmax == i))
    if union == 0:
        iou = 1.0  # If there are no ground truth or predicted pixels for this class, set IoU to 1 (by definition)
    else:
        iou = intersection / union
    iou_classes.append(iou)

class_iou = dict(zip(range(n_classes), iou_classes))
print("IoU for each class:", class_iou)




import numpy as np

# Extract y_true and y_pred from your model training
# Assuming your model is named 'model' and you have a validation dataset (X_test, y_test_cat)
# Predict on validation data
y_pred = model.predict(X_test)

# Assuming y_test_cat is the ground truth segmentation masks
y_true = y_test_cat
import tensorflow as tf

# Cast y_pred to the same data type as y_true (float64)
y_pred = tf.cast(y_pred, dtype=tf.float64)

# Function to calculate mean Dice coefficient for each class
def mean_dice_per_class(y_true, y_pred, num_classes, smooth=1):
    mean_dice_per_class = []
    for class_idx in range(num_classes):
        class_true = y_true[..., class_idx]
        class_pred = y_pred[..., class_idx]
        dice = dice_coefficient(class_true, class_pred, smooth)
        mean_dice_per_class.append(dice)
    return mean_dice_per_class

# Example usage:
# Calculate mean Dice coefficient for each class
mean_dice_per_class_values = mean_dice_per_class(y_true, y_pred, n_classes)

# Print mean Dice coefficient for each class
for class_idx, dice_value in enumerate(mean_dice_per_class_values):
    print(f"Mean Dice coefficient for class {class_idx}: {dice_value.numpy()}")







import numpy as np
import tensorflow as tf

# Assuming your model is named 'model' and you have a validation dataset (X_test, y_test_cat)
# Predict on validation data
y_pred = model.predict(X_test)

# Assuming y_test_cat is the ground truth segmentation masks
y_true = y_test_cat

# Ensure y_pred is cast to the same data type as y_true (float64)
y_pred = tf.cast(y_pred, dtype=tf.float64)
y_true = tf.cast(y_true, dtype=tf.float64)

# Define the dice_coefficient function
def dice_coefficient(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2))  # Sum along spatial dimensions (height and width)
    sum_true = tf.reduce_sum(y_true, axis=(1, 2))
    sum_pred = tf.reduce_sum(y_pred, axis=(1, 2))
    dice = (2. * intersection + smooth) / (sum_true + sum_pred + smooth)
    return tf.reduce_mean(dice, axis=0)  # Average over batch

# Function to calculate mean Dice coefficient for each class
def mean_dice_per_class(y_true, y_pred, num_classes, smooth=1):
    mean_dice_per_class_values = []
    for class_idx in range(num_classes):
        class_true = y_true[..., class_idx]
        class_pred = y_pred[..., class_idx]
        dice = dice_coefficient(class_true, class_pred, smooth)
        mean_dice_per_class_values.append(dice)
    return mean_dice_per_class_values

# Example usage:
# Calculate mean Dice coefficient for each class
num_classes = y_true.shape[-1]  # Assuming the last dimension is the class dimension
mean_dice_per_class_values = mean_dice_per_class(y_true, y_pred, num_classes)

# Print mean Dice coefficient for each class
for class_idx, dice_value in enumerate(mean_dice_per_class_values):
    print(f"Mean Dice coefficient for class {class_idx}: {dice_value.numpy()}")











import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion
from joblib import Parallel, delayed

def surface_distances(y_true_bin, y_pred_bin):
    """
    Calculate the surface distances using Euclidean distance transform.
    """
    true_surface = y_true_bin & ~binary_erosion(y_true_bin)
    pred_surface = y_pred_bin & ~binary_erosion(y_pred_bin)
    
    if np.sum(true_surface) == 0 or np.sum(pred_surface) == 0:
        return float('nan')  # Return NaN if no surface found

    true_dist = distance_transform_edt(~true_surface)
    pred_dist = distance_transform_edt(~pred_surface)

    # Calculate distances from true surface to predicted surface and vice versa
    true_to_pred_dist = pred_dist[true_surface]
    pred_to_true_dist = true_dist[pred_surface]

    # Calculate the mean surface distance
    asd = (np.mean(true_to_pred_dist) + np.mean(pred_to_true_dist)) / 2.0

    return asd

def asd_per_class(y_true, y_pred, num_classes):
    asd_per_class_values = Parallel(n_jobs=-1)(delayed(surface_distances)(
        y_true[..., class_idx] > 0.5, y_pred[..., class_idx] > 0.5
    ) for class_idx in range(num_classes))
    return asd_per_class_values

# Example usage:
# Assuming y_pred and y_true are already defined and have the shape (batch_size, height, width, num_classes)
# Cast y_pred to the same data type as y_true (float64)
y_pred = tf.cast(y_pred, dtype=tf.float64).numpy()
y_true = tf.cast(y_true, dtype=tf.float64).numpy()

# Calculate ASD for each class
asd_per_class_values = asd_per_class(y_true, y_pred, n_classes)

# Print ASD for each class
for class_idx, asd_value in enumerate(asd_per_class_values):
    print(f"ASD for class {class_idx}: {asd_value}")







import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion
from joblib import Parallel, delayed

def surface_distances(y_true_bin, y_pred_bin, threshold):
    """
    Calculate the surface distances using Euclidean distance transform and compute NSD.
    """
    true_surface = y_true_bin & ~binary_erosion(y_true_bin)
    pred_surface = y_pred_bin & ~binary_erosion(y_pred_bin)
    
    if np.sum(true_surface) == 0 or np.sum(pred_surface) == 0:
        return float('nan')  # Return NaN if no surface found

    true_dist = distance_transform_edt(~true_surface)
    pred_dist = distance_transform_edt(~pred_surface)

    # Calculate distances from true surface to predicted surface and vice versa
    true_to_pred_dist = pred_dist[true_surface]
    pred_to_true_dist = true_dist[pred_surface]

    # Count points within the threshold distance
    true_within_threshold = np.sum(true_to_pred_dist <= threshold)
    pred_within_threshold = np.sum(pred_to_true_dist <= threshold)

    # Normalize by the total number of surface points
    nsd_true = true_within_threshold / np.sum(true_surface)
    nsd_pred = pred_within_threshold / np.sum(pred_surface)

    # Calculate the mean normalized surface distance
    nsd = (nsd_true + nsd_pred) / 2.0

    return nsd

def nsd_per_class(y_true, y_pred, num_classes, threshold):
    nsd_per_class_values = Parallel(n_jobs=-1)(delayed(surface_distances)(
        y_true[..., class_idx] > 0.5, y_pred[..., class_idx] > 0.5, threshold
    ) for class_idx in range(num_classes))
    return nsd_per_class_values

# Example usage:
# Assuming y_pred and y_true are already defined and have the shape (batch_size, height, width, num_classes)
# Cast y_pred to the same data type as y_true (float64)
y_pred = tf.cast(y_pred, dtype=tf.float64).numpy()
y_true = tf.cast(y_true, dtype=tf.float64).numpy()

# Define the threshold distance
threshold_distance = 1.0  # Adjust this value as needed

# Calculate NSD for each class
nsd_per_class_values = nsd_per_class(y_true, y_pred, n_classes, threshold_distance)

# Print NSD for each class
for class_idx, nsd_value in enumerate(nsd_per_class_values):
    print(f"NSD for class {class_idx}: {nsd_value}")









import matplotlib.pyplot as plt

# Set custom style
# plt.style.use('seaborn-darkgrid')

# Plot accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['soft_dice_coefficient'], label='soft_dice_coefficient', linewidth=2)
plt.plot(history.history['val_soft_dice_coefficient'], label='Validation Soft Dice Coefficient', linewidth=2)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('soft_dice_coefficient', fontsize=14)
plt.title('soft_dice_coefficient vs. Epoch', fontsize=16)
plt.legend(fontsize=10)
plt.grid(True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)  # Decrease font size for y-axis ticks
plt.locator_params(axis='y', nbins=10)  # Set the number of ticks on the y-axis
plt.tight_layout()
plt.show()



# Plot accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('Accuracy vs. Epoch', fontsize=16)
plt.legend(fontsize=10)
plt.grid(True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)  # Decrease font size for y-axis ticks
plt.locator_params(axis='y', nbins=10)  # Set the number of ticks on the y-axis
plt.tight_layout()
plt.show()

# Plot training history
plt.figure(figsize=(8, 6))
plt.plot(history.history['mean_iou'], label='Mean IoU', linewidth=2)
plt.plot(history.history['val_mean_iou'], label='Validation Mean IoU', linewidth=2)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Mean IoU', fontsize=14)
plt.title('Mean IoU vs. Epoch', fontsize=16)
plt.legend(fontsize=10)
plt.grid(True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()




# Save metrics to a log file
with open("evaluation_log_model1_t2_SPACE_Epochs.txt", "w") as file:
    file.write("Mean IoU: {}\n".format(mean_iou))
    file.write("IoU for each class: {}\n".format(class_iou))
    file.write("Loss: {}\n".format(loss))
    file.write("Accuracy: {}\n".format(accuracy))
    file.write("Dice Coefficient: {}\n".format(dice_coefficient))
    

print("Evaluation metrics saved to evaluation_log1.txt")



from sklearn.metrics import confusion_matrix

# Predict on the test dataset
y_pred = model.predict(X_test)

# Convert predicted masks and ground truth masks to categorical labels
y_pred_labels = np.argmax(y_pred, axis=-1)
y_true_labels = np.argmax(y_test_cat, axis=-1)

# Flatten both predicted and ground truth labels
y_pred_labels_flat = y_pred_labels.flatten()
y_true_labels_flat = y_true_labels.flatten()

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true_labels_flat, y_pred_labels_flat)

print("Confusion Matrix:")
print(conf_matrix)


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Predict on the test dataset
y_pred = model.predict(X_test)

# Convert predicted masks and ground truth masks to categorical labels
y_pred_labels = np.argmax(y_pred, axis=-1)
y_true_labels = np.argmax(y_test_cat, axis=-1)

# Flatten both predicted and ground truth labels
y_pred_labels_flat = y_pred_labels.flatten()
y_true_labels_flat = y_true_labels.flatten()

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true_labels_flat, y_pred_labels_flat)

# Define class labels
class_labels = ["Class 0", "Class 1", "Class 2", "Class 3"]

# Plot confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Reds', fmt='g', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

def calculate_metrics(conf_matrix):
    TP = np.diag(conf_matrix)
    FP = np.sum(conf_matrix, axis=0) - TP
    FN = np.sum(conf_matrix, axis=1) - TP

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score

precision, recall, f1_score = calculate_metrics(conf_matrix)

# Print the results
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)


import numpy as np

def calculate_metrics(conf_matrix):
    # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
    TP = np.diag(conf_matrix)
    FP = np.sum(conf_matrix, axis=0) - TP
    FN = np.sum(conf_matrix, axis=1) - TP

    # Calculate Precision
    precision = TP / (TP + FP)

    # Calculate Recall (Sensitivity or True Positive Rate)
    recall = TP / (TP + FN)

    # Calculate F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Calculate average precision, recall, and F1 score
    avg_precision = np.mean(precision)
    avg_recall = np.mean(recall)
    avg_f1_score = np.mean(f1_score)

    return avg_precision, avg_recall, avg_f1_score


# Call the function to calculate metrics
avg_precision, avg_recall, avg_f1_score = calculate_metrics(conf_matrix)

print("Precision:", avg_precision)
print("Recall:", avg_recall)
print("F1 Score:", avg_f1_score)







import numpy as np
from scipy.spatial.distance import directed_hausdorff
from skimage import measure

# Function to extract boundary points from a 2D segmentation map
def get_boundary_points(segmentation):
    contours = measure.find_contours(segmentation, 0.5)
    if len(contours) > 0:
        boundary_points = np.concatenate(contours)
    else:
        boundary_points = np.array([])
    return boundary_points

# Function to calculate Hausdorff distance for each class across a batch of 2D slices
def hausdorff_distance(y_true, y_pred, num_classes):
    hausdorff_distances = {i: [] for i in range(num_classes)}
    
    # Iterate over each 2D slice
    for slice_idx in range(y_true.shape[0]):
        true_slice = y_true[slice_idx, :, :]
        pred_slice = y_pred[slice_idx, :, :]
        
        for i in range(num_classes):
            true_class = (true_slice == i).astype(np.uint8)
            pred_class = (pred_slice == i).astype(np.uint8)
            
            true_boundary_points = get_boundary_points(true_class)
            pred_boundary_points = get_boundary_points(pred_class)
            
            if true_boundary_points.size == 0 or pred_boundary_points.size == 0:
                hausdorff_distance = 0.0  # If there are no boundary points, set Hausdorff distance to 0
            else:
                forward_hausdorff = directed_hausdorff(true_boundary_points, pred_boundary_points)[0]
                backward_hausdorff = directed_hausdorff(pred_boundary_points, true_boundary_points)[0]
                hausdorff_distance = max(forward_hausdorff, backward_hausdorff)
            
            hausdorff_distances[i].append(hausdorff_distance)
    
    # Compute the mean Hausdorff distance for each class
    mean_hausdorff_distances = {i: np.mean(hausdorff_distances[i]) for i in range(num_classes)}
    return mean_hausdorff_distances

# Example usage
y_pred = model.predict(X_test)
y_pred_argmax = np.argmax(y_pred, axis=3)

# Assuming y_test has the shape (num_samples, height, width, 1)
y_test_2d = y_test[:,:,:,0]

n_classes = 4
hausdorff_distances = hausdorff_distance(y_test_2d, y_pred_argmax, n_classes)

print("Hausdorff distance for each class:", hausdorff_distances)










import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define the threshold
threshold = 0.2

# Select a random test image
test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth = y_test[test_img_number]
test_img_norm = test_img[:, :, 0]  # Assuming the image is grayscale

# Make prediction
prediction = model.predict(np.expand_dims(test_img, axis=0))
predicted_img = np.argmax(prediction, axis=-1)[0]  # Assuming channel dimension is the last axis

# Define class names
class_names = ['Background', 'Disc', 'Vertebra', 'Spine Canal']

# Plotting
plt.figure(figsize=(16, 12))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img_norm, cmap='gray')  # Display the grayscale image
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:, :, 0], cmap='viridis', extent=(0, ground_truth.shape[1], ground_truth.shape[0], 0))  # Set extent to match the size of the image
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img, cmap='viridis')

# Add bounding boxes with class names for non-background classes
for class_index in range(1, len(class_names)):
    # Find contours for each class
    contours, _ = cv2.findContours(np.uint8(ground_truth[:, :, 0] == class_index), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:  # Check if there are contours for the class
        contour = max(contours, key=cv2.contourArea)  # Get the largest contour
        # Get bounding box coordinates
        x, y, w, h = cv2.boundingRect(contour)
        # Draw bounding box
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='white', facecolor='none')  # Change edgecolor here
        plt.gca().add_patch(rect)
        # Add class name
        class_name = class_names[class_index]
        # Position the text to the right of the bounding box
        text_x = x + w + 5  # Adjust the offset as needed
        text_y = y + h // 2  # Position at the center of the bounding box
        plt.text(text_x, text_y, class_name, color='white', fontsize=12, verticalalignment='center')

plt.show()








#brightness####################
import random
import numpy as np
import matplotlib.pyplot as plt

# Define the threshold
threshold = 0.2

# Select a random test image
test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
test_img_norm = test_img[:, :, 0]  # Assuming the image is grayscale

# Increase brightness by scaling pixel values
brightness_factor = 6  # Adjust as needed
brightened_img = test_img_norm * 255  # Assuming pixel values are in the range [0, 1]
brightened_img = np.clip(brightened_img * brightness_factor, 0, 255) / 255  # Rescale pixel values back to [0, 1] range

# Make prediction
prediction = model.predict(np.expand_dims(test_img, axis=0))
predicted_probabilities = prediction[0]  # Assuming channel dimension is the last axis

# Apply threshold
predicted_mask = np.argmax(predicted_probabilities, axis=-1)
predicted_mask[predicted_probabilities.max(axis=-1) < threshold] = 0  # Assuming background class index is 0

# Define transparency levels for each class
alpha_values = np.ones_like(predicted_mask, dtype=np.float32)

# Set transparency for background class
alpha_values[predicted_mask == 0] = 0.1

# Set transparency for other classes
alpha_values[(predicted_mask != 0) & (predicted_mask != 1)] = 0.9  # Adjust alpha for other classes as desired

# Plotting
plt.figure(figsize=(8, 6))

# Overlay the original image and the predicted ground truth mask with transparency
plt.imshow(brightened_img, cmap='gray') #viridis, gray, c, plasma, coolwarm, cividis, magma
plt.imshow(predicted_mask, cmap='viridis', alpha=alpha_values, interpolation='nearest')  # Overlay the predicted ground truth mask with transparency
plt.title('Original Image with Predicted Ground Truth Mask')
plt.show() 







import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load testing image and label from your PC
test_img_path = "C:/Users/User/Desktop/test2/images/test_img_45.png"  # Change this to the path of your test image
test_label_path = "C:/Users/User/Desktop/test2/masks/test_img_45.png"  # Change this to the path of your test label

# Load test image and label in grayscale
test_img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
ground_truth = cv2.imread(test_label_path, cv2.IMREAD_GRAYSCALE)

# Resize the test image to match the input shape expected by the model
test_img = cv2.resize(test_img, (256, 256))

# Normalize test image (assuming the same normalization as training)
test_img_normalized = test_img / 255.0  # Assuming training images were normalized to [0, 1]

# Make prediction
prediction = model.predict(np.expand_dims(test_img_normalized, axis=0))
predicted_img = np.argmax(prediction, axis=-1)[0]  # Assuming channel dimension is the last axis

# Plotting
plt.figure(figsize=(16, 12))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img, cmap='gray')  # Display the grayscale image
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth, cmap='viridis')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img, cmap='viridis')  # Change the cmap as needed
plt.show()
