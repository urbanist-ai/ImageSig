# ImageSig

This is the official implementation of ImageSig (CVPR 2022- Embedded vision)
https://openaccess.thecvf.com/content/CVPR2022W/EVW/papers/Ibrahim_ImageSig_A_Signature_Transform_for_Ultra-Lightweight_Image_Recognition_CVPRW_2022_paper.pdf

Install requirements:
```python
pip install -r requirements.txt
```
Download dataset:

The demo needs Celeba dataset or Crack detection dataset as presented in the paper.

Key hyperparameters for using ImageSig with image array:
```python
train_imagesig = image_signature_from_array (
    
                            x_train = x_train,
                            y_train = y_train,
                            x_test = x_test,
                            y_test = y_test,
                            depth = 4,
                            image_size = (64,64),
                            augment_flip_horizontal = True,
                            augment_color = False,
                            augment_rotate_45 = False,
                            augment_add_noise = False,
                            augment_brightness= True
                            )
                            
x_train, y_train, x_test, y_test = train_imagesig.read_array()
```



Key hyperparameters for using ImageSig with image directory:
```python
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

```

To train the model on the downloaded dataset or your dataset:

Make sure that all categories are included in the structure as follows:
TRAIN_DIR = "data/training_set"
TEST_DIR = "data/test_set"

```python
python train_imagesig.py
```

##Acknowledgement

If you use this code, please cite the following article:
