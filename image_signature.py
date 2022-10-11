# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 02:09:28 2021

@author: Ibrahim
"""

import torch, cv2, os, random
import numpy as np
import matplotlib.pyplot as plt
import signatory

import time
import tensorflow as tf 

from focal_loss import SparseCategoricalFocalLoss

import iisignature

'''
IMAGE SIGNATURE CLASS FOR IMAGE DIRS: TRAINING AND TESTING
'''



class image_signature ():
    
    def __init__(self,image_dir,depth, image_size, 
                 flatten,log_sig,two_direction,
                 back_end="signatory",
                 augment_flip_horizontal=False,
                 augment_flip_vertical=False,
                 augment_color=False
                 ,augment_rotate_45= False, 
                 augment_add_noise = False,
                 augment_brightness=False):

        self.image_dir = image_dir
        self.depth = depth
        self.image_size = image_size
        self.flatten  = flatten
        self.log_sig = log_sig
        self.augment_flip_horizontal = augment_flip_horizontal
        self.augment_color = augment_color
        self.augment_rotate_45 = augment_rotate_45
        self.augment_add_noise = augment_add_noise
        self.augment_brightness = augment_brightness
        self.augment_flip_vertical = augment_flip_vertical

        self.two_direction= two_direction
        self.back_end=back_end
        self.x = []
        self.label = []
        self.signature_array = []

    # def get_image(self,image_path):
    #     image = cv2.imread(image_path)
    #     image = cv2.resize(image,self.image_size, interpolation = cv2.INTER_AREA)
    #     image = image.astype("float32") / 255.
        
    #     return image
    

    def visualize_image (self,index=1,save_fig=False):
        #index = random.randint(0, len(self.x))
        im1 = self.x[index]
        fig, ax = plt.subplots(3, 1, figsize=(5, 11))
        ax[0].set_title('Input image')
        ax[0].axis("off")
        #ax[0].imshow(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))
        ax[0].imshow(im1)
    
    
        ax[1].set_title('Image streams')
        ax[1].plot(self.signature[index])
        
        ax[2].set_title('Image signature')
        ax[2].imshow(self.signature[index])
        if save_fig == True:
            file_path = "signature_visualisation"
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            plt.savefig(f'{file_path}/image_sginature_{index}.jpg',bbox_inches='tight', pad_inches=0.1)
            plt.show()
                
        plt.show()

    def imagesig (self,im,type="RGB"):
        
        signatures = []
        
        for stream in im:
            #print(stream.shape)
            stream = stream.astype("float32") / 255.
            if type=="GRAY":
                stream = np.reshape(stream,(stream.shape[0],1))
            sig =  iisignature.sig(stream, self.depth)# compute the signature
            #sig = esig.path2sig(stream, depth)
            #sig = sig[1:]
            signatures.append(sig)
        signatures = np.array(signatures)
        return(signatures)
    
    
    #    return(signatures)
    def array_to_sig (self):
        s = []
        for index, i in enumerate(self.x):
            print("Computed signature....",round(index/len(self.x),2))
            sig2 = self.imagesig(i,self.depth)
            s.append(sig2)
            
        ##s= tf.stack(s,0)  
        return (np.array(s))
        
    def get_signature(self):
        signature = []
        
        for index, img in enumerate(self.x): 
            img = img.astype("float32") / 255.
            
            img_col = self.rotate_image(img,90) #### to get signature from col as well
            im_col_torch = torch.from_numpy(img_col)

            im_torch = torch.from_numpy(img)
            if self.log_sig == True: 
                sig = signatory.logsignature(im_torch, self.depth)
                sig = sig.numpy()
                signature.append(sig)
            else:
                if self.two_direction == False:
                    sig = signatory.signature(im_torch, self.depth)
                    sig = sig.numpy()
                    signature.append(sig)
                else:
                    sig = signatory.signature(im_torch, self.depth)
                    sig2 = signatory.signature(im_col_torch, self.depth)
                    # sig_combined = signatory.signature_combine(sig, sig2,
                    #                            3, self.depth)
                       
                    # sig_combined = sig_combined.numpy()
                    sig = sig.numpy()
                    sig2 = sig2.numpy()
                    sig_combined = np.concatenate((sig,sig2), axis=0)
                    signature.append(sig_combined)

        return np.array(signature)


    def add_noise(self,image):
        gauss = np.random.normal(0,0.6,image.size)
        gauss = gauss.reshape(image.shape[0],image.shape[1],image.shape[2]).astype('uint8')
        # Add the Gaussian noise to the image
        img_gauss = cv2.add(image,gauss)
        return img_gauss

    
    def rotate_image(self,image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result
    
    def increase_brightness(self,image, value=30):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
    
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    
        final_hsv = cv2.merge((h, s, v))
        outcome2 = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return outcome2


    def read_image_dir (self):
        image_path = []
        for class_index, classes in enumerate(os.listdir(self.image_dir)):
            for image_index, j in enumerate(os.listdir(os.path.join(self.image_dir,classes))):
                
                path = self.image_dir +"/"+ classes+"/"+j 
                #im = self.get_image(j)
                im = cv2.imread(path)
                if im is not None: 
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    im = cv2.resize(im,self.image_size, interpolation = cv2.INTER_AREA)
                    #im = im.astype("float32") / 255.
                    self.x.append(im)
                    image_path.append(path)
                    self.label.append(class_index)
                    
                    if self.augment_flip_horizontal == True:
                        im_2 = cv2.flip(im, 1)
                        self.x.append(im_2)
                        self.label.append(class_index)
                        
                    if self.augment_color == True:
                        im_3 = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                        self.x.append(im_3)
                        self.label.append(class_index)                  

                    if self.augment_rotate_45 == True:
                        im_4 = self.rotate_image(im,45)
                        self.x.append(im_4)
                        self.label.append(class_index)     
                        
                    if self.augment_add_noise == True:
                        im_5 = self.add_noise(im)
                        self.x.append(im_5)
                        self.label.append(class_index)   
                        
                                                    
                    if self.augment_brightness == True:
                        im_6 = self.increase_brightness(im,40)
                        self.x.append(im_6)
                        self.label.append(class_index)                          

                        im_7 = self.increase_brightness(im,80)
                        self.x.append(im_7)
                        self.label.append(class_index)

                        
                    if self.augment_flip_vertical == True:
                        im_9 = cv2.flip(im, 0)
                        self.x.append(im_9)
                        self.label.append(class_index)
                        
                print(f"...PROCESSED IMAGES: {classes}_{image_index}")

        self.x =np.array(self.x)
        if (self.back_end =="iisignature"):
            sig = self.array_to_sig()
        else:
            sig = self.get_signature()
     
            

        self.signature = sig
        
        if self.flatten == True:
            sig = np.reshape(sig,(sig.shape[0],-1))
            self.signature = sig
        label = np.array(self.label)
         
        return(self.x,self.signature,label)
    




class image_signature_from_array():
    
    def __init__(self,x_train, y_train, x_test, y_test, depth,image_size,
                 augment_flip_horizontal,
                 augment_color,augment_rotate_45, 
                 augment_add_noise,augment_brightness):
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.depth = depth
        self.image_size = image_size

        self.augment_flip_horizontal = augment_flip_horizontal
        self.augment_color = augment_color
        self.augment_rotate_45 = augment_rotate_45
        self.augment_add_noise = augment_add_noise
        self.augment_brightness = augment_brightness

    
    def get_signature(self,array, depth):
        
        signature = []
        for index, image in enumerate(array): 
            image = cv2.resize(image,self.image_size, interpolation = cv2.INTER_AREA)
            ##### when image is grey
            image = np.reshape(image, (image.shape[0],image.shape[1],3))
            ### else
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype("float32") / 255.

            im_torch = torch.from_numpy(image)
            sig = signatory.signature(im_torch, depth)
            sig = sig.numpy()
            signature.append(sig)
        return np.array(signature)


    def add_noise(self,image,prob):
        '''
        Add salt and pepper noise to image
        prob: Probability of the noise
        '''
        output = np.zeros(image.shape,np.uint8)
        thres = 1 - prob 
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = image[i][j]
        return output
    
    def rotate_image(self,image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result
    
    def increase_brightness(self,image, value=30):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
    
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    
        final_hsv = cv2.merge((h, s, v))
        outcome2 = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return outcome2
    
    
    def read_array (self):
        
        x_train_aug = []
        y_train_aug = []
        for index, im in enumerate(self.x_train): 
            
            if self.augment_flip_horizontal == True:
                    im = cv2.flip(im, 1)
                    x_train_aug.append(im)
                    class_index = self.y_train[index]
                    y_train_aug.append(class_index)
            

            if self.augment_color == True:
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    x_train_aug.append(im)
                    class_index = self.y_train[index]
                    y_train_aug.append(class_index)
                        

            if self.augment_rotate_45 == True:
                    im = self.rotate_image(im,45)
                    x_train_aug.append(im)
                    class_index = self.y_train[index]
                    y_train_aug.append(class_index)
            
                
            if self.augment_add_noise == True:
                    im = self.add_noise(im,0.03)
                    x_train_aug.append(im)
                    class_index = self.y_train[index]
                    y_train_aug.append(class_index)
            
                
            if self.augment_brightness == True:
                    im = self.increase_brightness(im,40)
                    x_train_aug.append(im)
                    class_index = self.y_train[index]
                    y_train_aug.append(class_index)
                    
                             
            print (f"Computed signature... {index}")
        x_train_sig = self.get_signature(x_train_aug,self.depth)
        x_test_sig = self.get_signature(self.x_test,self.depth)
        return x_train_sig, np.array(y_train_aug), x_test_sig, self.y_test
    



"""
Predict a given image image, 
"""



def imagesig_predict (image_path, model, 
                      class_names= ['fire','no fire'],
                      depth=4, 
                      image_size = (64,64),
                      two_direction = False):
    
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

    
    t1 =time.time()
    #image_path = "fire3.jpg"
    prediction = model.predict(im_signature_array)
    prediction = np.argmax(prediction, axis = 1).tolist()[0]
    prediction = class_names[prediction]
    
    t2 = time.time()
    
    print("Inference time in second: ", (1/(t2-t1)))
    print(f"predicted class: {prediction}" )
    
    plt.axis('off')
    plt.title(f"predict class: {prediction}")
    plt.imshow(image_sample)
    return prediction



"""
TfLite model
"""

def quantize_model(checkpoint):
    # Convert to TFLite. This form of quantization is called
    # post-training dynamic-range quantization in TFLite.
    ###Requires Check_point_path as str
    
    converter = tf.lite.TFLiteConverter.from_saved_model(str(checkpoint))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS,  # Enable TensorFlow ops.
    ]
    tflite_model = converter.convert()
    
    name = (str(checkpoint).split(".",2)[0])
    open(f"{name}_quant.tflite", "wb").write(tflite_model)
    
# model = tf.keras.models.load_model("sig_models/imagesig_dense_input_32_depth_1.h5")
# model.save("sig_models/imagesig_dense_input_32_depth_1")
# quantize_model("sig_models/imagesig_dense_input_32_depth_1")
