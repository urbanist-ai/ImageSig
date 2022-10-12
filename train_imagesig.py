# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 15:01:49 2021

@author: Ibrahim
"""
import pandas as pd

import torch
import signatory
import numpy as np
import cv2




import numpy as np
import os
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

import tensorflow_addons as tfa

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import Input,regularizers
from tensorflow.keras.layers import Dense, BatchNormalization,Reshape,Lambda, Embedding, Concatenate, Flatten,Dropout,Bidirectional, Conv1D,TimeDistributed, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Model
from IPython.display import SVG
import time

from focal_loss import SparseCategoricalFocalLoss

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
import os
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras import layers
from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np

import tensorflow.keras.backend as K

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LayerNormalization,Dense, GlobalAveragePooling2D, Dropout,Concatenate
from IPython.display import clear_output

import random


from image_signature import image_signature



##### SIG paramaters
SIG_DEPTH = 4
IMAGE_SIZE = (64,64) ##  W  X  H
FLATTEN = False
TRAIN_DIR = "data/training_set"
TEST_DIR = "data/test_set"

train_imagesig = image_signature (
                            image_dir = TRAIN_DIR,
                            depth = SIG_DEPTH,
                            image_size = IMAGE_SIZE,
                            flatten=FLATTEN,
                            #back_end="iisignature",
                            log_sig = False,
                            two_direction = False,  
                            augment_flip_horizontal = True,
                            augment_flip_vertical =True,
                            augment_add_noise = False,
                            augment_brightness= True,
                            )

test_imagesig = image_signature (
                            image_dir = TEST_DIR,
                            depth = SIG_DEPTH,
                            image_size = IMAGE_SIZE,
                            flatten=FLATTEN,
                            #back_end="iisignature",
                            log_sig = False,
                            two_direction=False
                            )

    


def label_weights(y_train, dict_output=True):
    
    """
    RETURN: NUMBER OF CLASSES, AND CLASS WEIGHT 
    dict_output = in case the output of class weight needed as dict, else: list 
    """
    num_classes = len(np.unique(y_train))
    class_weight = []
    class_weight_dict = {}
    class_values = np.bincount(y_train)
    total = 0
    for i in range(0, num_classes):
        total += class_values[i] 
    for i in range(0, num_classes):
        print('Class {}:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(i,
        total, class_values[i], 100 * class_values[i] / total))
        weight= (1 / class_values[i]) * (total / num_classes)
        
        class_weight.append(weight)
        class_weight_dict.update({i:weight})
    if dict_output == True:
        return(num_classes, class_weight_dict)
    else:  
        return(num_classes, class_weight)








class PlotProgress(keras.callbacks.Callback):
    
    def __init__(self, entity='loss'):
        self.entity = entity
        
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('{}'.format(self.entity)))
        self.val_losses.append(logs.get('val_{}'.format(self.entity)))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="{}".format(self.entity))
        plt.plot(self.x, self.val_losses, label="val_{}".format(self.entity))
        plt.legend()
        plt.show();
        
        
plot_progress = PlotProgress(entity='accuracy')



"""
Only Dense layers Model, requires Flatten == TRUE
"""

def dense_model(num_classes):
    
    input_1 = Input(batch_shape=(None, x_train.shape[1],x_train.shape[2]), name='sig_input')
    f = Flatten()(input_1)
    dense_2 = Dense(units = 50, activation="relu")(f)
    dropout_2 = Dropout(0.4)(dense_2)
  
    dense_4 = Dense(units = num_classes, activation ="softmax")(dropout_2)
    model = Model([input_1], dense_4)
    return(model)



def cnn_model(num_classes):
    
    input_1 = Input(batch_shape=(None, x_train.shape[1],x_train.shape[2]), name='sig_input')
    cnn_1 = tf.keras.layers.Conv1D(32, kernel_size= 3, activation="relu")(input_1)
    pool_1= tf.keras.layers.MaxPooling1D(3)(cnn_1)

    cnn_2 = tf.keras.layers.Conv1D(64, kernel_size= 3, activation="relu")(pool_1)
    pool_2= tf.keras.layers.MaxPooling1D(3)(cnn_2)
    f = Flatten()(pool_2)
    dense_2 = Dense(units = 50, activation="relu")(f)
    dropout_2 = Dropout(0.4)(dense_2)
  
    dense_4 = Dense(units = num_classes, activation ="softmax")(dropout_2)
    model = Model([input_1], dense_4)
    return(model)






#########################################################








"""
ImageSig: A path signature based vision transformer 

Requires Flatten == False

"""


positional_emb = True

projection_dim = 64 #x_train.shape[2] ### signature length 

num_heads = 8
transformer_units = [
    projection_dim,
    projection_dim,
]
transformer_layers = 50
stochastic_depth_rate = 0.2

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 128
num_epochs = 50



def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x





def transformer_model(num_classes,
    num_heads=num_heads,
    projection_dim=projection_dim,
    transformer_units=transformer_units,
):

    inputs = Input(batch_shape=(None, x_train.shape[1], x_train.shape[2]), name='sig_input')


    cnn_1 = tf.keras.layers.Conv1D(32, kernel_size= 3, activation="relu")(inputs)
    pool_1= tf.keras.layers.MaxPooling1D(3)(cnn_1)

    cnn_2 = tf.keras.layers.Conv1D(64, kernel_size= 3, activation="relu")(pool_1)
    pool_2= tf.keras.layers.MaxPooling1D(3)(cnn_2)


    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-5)(pool_2)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)

        # Skip connection 1.s
        #attention_output = StochasticDepth(dpr[i])(attention_output)
        x2 = layers.Add()([attention_output, pool_2])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-5)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        ###Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Apply sequence pooling.
    representation = layers.LayerNormalization(epsilon=1e-5)(encoded_patches)
    representation = Dropout(0.4)(representation)

    attention_weights = tf.nn.softmax(
        layers.Dense(1)(representation), axis=1)
    x = tf.matmul(
        attention_weights, representation, transpose_a=True)
    x = tf.squeeze(x, -2)

    x_1 = Dense(50, activation='relu')(x)
    x_1 = Dropout(0.4)(x_1)
    y_1 = Dense(num_classes, name="y",activation="softmax")(x_1)
    
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=[y_1])

    return model





def main (model_name):
    
    if model_name == "fc":
        num_classes, class_weight = label_weights(y_train,dict_output=False)

        model = dense_model(num_classes)
        model.summary()
    
        # Calculae FLOPS
        from keras_flops import get_flops
        flops = get_flops(model, batch_size=1)
        print(f"FLOPS: {flops / 1000000} ")
        
        # checkpoint_filepath = os.path.join("tmp","checkpoint")
        
        # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        #     filepath=checkpoint_filepath,
        #     save_weights_only=False,
        #     monitor='val_accuracy',
        #     mode='max',
        #     save_best_only=True)
        
        loss=SparseCategoricalFocalLoss(gamma=2,
                                        class_weight=class_weight)  # Used here like a tf.keras loss
        
        optimizer = tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001)
        
        model.compile(optimizer = 'adam', 
                      loss = loss,  
                      metrics= ['accuracy']
                      )
        
        t1 = time.time()
        history = model.fit(x_train, #x_train, 
                            y_train, 
                            batch_size = 3000, #1024, 
                            epochs = 300, #150,
                            validation_split =0.1 ,
                            verbose = 2 ,shuffle=False,
                            #class_weight=class_weight,
                            callbacks=[plot_progress] ###,model_checkpoint_callback]
                            )
        t2 = time.time()
        
        print("training time in min: ", ((t2-t1)/60))
        
        model.evaluate(x_test,y_test)  
        
        #model.summary()
        
        #model.save(f"base_models/fc_resolution_64.h5")
        
        #model.save("facemask_cnn_checkpoint_64_depth_4_batch_4000_2_direction")
        
        #model.save(f"sig_models/imagesig_dense_log2.h5")
        
        #model.save(f"fog_models/fog_dense_input_{IMAGE_SIZE[0]}_depth_{SIG_DEPTH}_batch_3000_signatory.h5")
        
        #model.save(f"crack_detection_dense_input_{IMAGE_SIZE[0]}_depth_{SIG_DEPTH}_batch_3000_signatory.h5")

    if model_name == "cnn":
        num_classes, class_weight = label_weights(y_train,dict_output=False)
        model = cnn_model(num_classes)
        
        model.summary()
    
        
        # Calculae FLOPS
        from keras_flops import get_flops
        flops = get_flops(model, batch_size=1)
        print(f"FLOPS: {flops / 1000000} ")
        
        
        
        checkpoint_filepath = os.path.join("tmp","checkpoint")
        
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
        
        loss=SparseCategoricalFocalLoss(gamma=2,
                                        class_weight=class_weight)  # Used here like a tf.keras loss
        
        optimizer = tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001)
        
        model.compile(optimizer = 'adam', 
                      loss = loss,  
                      metrics= ['accuracy']
                      )
        
        t1 = time.time()
        history = model.fit(x_train, #x_train, 
                            y_train, 
                            batch_size = 3000, #1024, 
                            epochs = 300, #150,
                            validation_split =0.1 ,
                            verbose = 2 ,shuffle=False,
                            #class_weight=class_weight,
                            callbacks=[plot_progress] ###,model_checkpoint_callback]
                            )
        t2 = time.time()
        
        print("training time in min: ", ((t2-t1)/60))
        
        model.evaluate(x_test,y_test)      
    if model_name =="transformer":
        
        num_classes, class_weight = label_weights(y_train,dict_output=True)



        y_train_1 = keras.utils.to_categorical(y_train, num_classes)
        y_test_1 = keras.utils.to_categorical(y_test, num_classes)


        model = transformer_model(num_classes)
        
        model.summary()
        
        
        
        optimizer = tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001)
        
        model.compile(
            optimizer=optimizer,
            loss=keras.losses.CategoricalCrossentropy(
                from_logits=True, label_smoothing=0.1
            ),
            metrics=[
                keras.metrics.CategoricalAccuracy(name="accuracy"),
        
            ],
        )
        
        
        checkpoint_filepath = os.path.join("sig_transformer","checkpoint")
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            checkpoint_filepath,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        )
        
        
        t1 = time.time()
        history = model.fit(x_train, 
                            y_train_1, batch_size = 3000, 
                            epochs = 300, 
                            validation_split=0.1,
                            #validation_data=(x_test, y_test), 
                            verbose = 2 ,shuffle=False,
                            class_weight=class_weight,
                            callbacks=[checkpoint_callback,plot_progress
                                       ])
        
        t2 = time.time()
        
        
        
        #model.load_weights("transformer_Checkpoint")
        model.evaluate(x_test,y_test_1)
        
        print("training time in second: ", (t2-t1)/60)
        #model.save("transformer_fog_64_4_3000")
        
        
        # model_2 = model.load_weights("transformer_fire_signatory")
        
        # saved_model_dir = "transformer_fire_signatory"
        # # Convert the model
        # converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
        # tflite_model = converter.convert()
        
        # # Save the model.
        # with open('transformer_fire_signatory.tflite', 'wb') as f:
        #   f.write(tflite_model)
        

if __name__ == "__main__":
    
    ### choose ImageSig model to train: options = ["fc", "cnn", "transformer"]
    model_name = "fc"
    
    #### processing the data to image signature
    im_train,x_train,y_train = train_imagesig.read_image_dir()
    im_test, x_test,y_test = test_imagesig.read_image_dir()
    ### visalize a given sample, change 1 to any number within the lenght of a sequence
    train_imagesig.visualize_image(1,save_fig=False)

    main (model_name)