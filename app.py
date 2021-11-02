# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 21:49:19 2021

@author: Ibrahim
"""

import streamlit as st


import tensorflow as tf
import time 
import numpy as np
import cv2,io
import time
import numpy as np
import torch
import signatory





def classifier_quant(path, classes_names,image_frame):

  interpreter = tf.lite.Interpreter(model_path = path)
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  start_time = time.time()
  interpreter.set_tensor(input_details[0]['index'], image_frame)
   # run the inference
  interpreter.invoke()
  output_data = interpreter.get_tensor(output_details[0]['index'])
  elapsed_ms = (time.time() - start_time) * 1000
        #print predict classes
  classes = np.argmax(output_data, axis = 1)
  #print("elapsed time: ", elapsed_ms, " , predict class number: ", classes, " ,is class name: ", classes_names[classes[0]], sep='')
  return(classes_names[classes[0]])



def imagesig_predict_quant (image_path ,depth=4, 
               image_size = (64,64),two_direction = False):
    
    #image = cv2.imread(image_path)
    image = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,image_size, interpolation = cv2.INTER_AREA)
    image_sample = image
    image = image.astype("float32")/ 255.
    im_torch = torch.from_numpy(image)
    sig = signatory.signature(im_torch, depth)
    im_signature = sig.numpy()
    im_signature_array = np.reshape(im_signature,(1,image_size[0],-1))

    if two_direction == True:
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, 90, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        im_col_torch = torch.from_numpy(result)
        
        sig2 = signatory.signature(im_col_torch, depth)
        sig2 = sig2.numpy()
        x = np.concatenate((im_signature,sig2), axis=0)
        im_signature_array = np.reshape(x,(1,x.shape[0],-1))



    return im_signature_array





#image_path = "fire4.jpg"

#path = 'cnn_checkpoint_64_2_direction_quant.tflite'
classes_names = ["Fire", "No fire"]


image_size = (500,500)


st.title("Image Signature")
st.header("Fire prediction")

st.write("Demo copyrights: Mohamed R Ibrahim")

option = ['cnn_checkpoint_64_2_direction_quant.tflite']
path= st.sidebar.selectbox("Select model weight:", option)
uploaded_file = st.file_uploader("Select an image",accept_multiple_files=False,type=["jpg","jpeg","png"])

if uploaded_file is not None:
    g= io.BytesIO(uploaded_file.read())  ## BytesIO Object
    temporary_location = "saved_data/testout_{}.jpg".format(time.time())

    with open(temporary_location, 'wb') as out:  ## Open temporary file as bytes
        out.write(g.read())  ## Read bytes into file
    
        image = cv2.imread(temporary_location)
        
        
        image_visualization = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image_visualization = cv2.resize(image_visualization,image_size, interpolation = cv2.INTER_AREA)

        sig = imagesig_predict_quant(image,two_direction=True)
        
        with st.spinner('Wait for it...'):
            t1= time.time()
            pred = classifier_quant(path, classes_names,sig)
            t2= time.time()
            print("Inference time in second: ", (1/(t2-t1)))
            print("Predicted class: ",pred)
        
            
            
            st.image(image_visualization)
            # st.write("Predicted class: ",pred)
            fps= round((1/(t2-t1)),2)
            # st.write("Inference time in second: ", fps)
    
            diff_fps = round((fps-30),2)
            col1, col2 = st.columns(2)
            col1.metric("PREDICTION", pred, delta=None, delta_color='normal')
            col2.metric("FPS", fps, delta=diff_fps, delta_color='normal')

