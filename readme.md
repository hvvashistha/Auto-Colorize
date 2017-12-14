
<center><h1>Automated Colorization of Images</h1></center>

<center>Harsh Vashistha, Deepanjan Bhattacharyya, Bharat Prakash</center>
<center>{harsh5, deep8, bh1} @umbc.edu</center>
<hr/>
<br/><br/>
<center>
Project for Data Science course ([CMSC 491/691](https://www.csee.umbc.edu/~kalpakis/courses/491-fa17/))
</center>
<center>
[Project Slides](resources/slides.pdf)
</center>
<center>
Data Set - [ImageNet fall 11 URLs](http://image-net.org/imagenet_data/urls/imagenet_fall11_urls.tgz)
</center>

## Project Introduction

Converting a black & white image to color is a tedious task

There are multiple issues which needs to be tackled to create a good model. Some of these limitations and issues are: 
* Too many objects and varying textures.
* Each object  or part of image can take on different colors. 
* The problem is under-constrained and can have multiple solutions.

| GrayScale Input | A possible color for image | Another possible color for the image |
|:---:|:---:|:---:|
| <img src="resources/intro_bnw.png" style="width:300px; height:200px; vertical-align:bottom; display:inline-block;"/> | <img src="resources/intro_c1.png" style="width:300px; height:200px; vertical-align:bottom; display:inline-block;"/> | <img src="resources/intro_c2.png" style="width:300px; height:200px; vertical-align:bottom; display:inline-block;"/> |




```python
# Setting up environment variable for code which needs to run only from jupyter notebook

__IPython__ = True
try:
    get_ipython        
except NameError:
    __IPython__ = False

if __IPython__:
    %matplotlib inline
```

Minsky/BlueSpeed Server has **tensorflow v1.3** as latest available version while Keras uses tensorflow v1.4 for
```multi_gpu_model``` call.
The only missing function which cause conflicts is ```tf.Session.sess.list_devices()``` which is in v1.3 is ```device_lib.list_local_devices()```
Creating a dummy below


```python
from tensorflow.python.client import device_lib
from keras import backend as K

def get_local_devices():
    local_devices = device_lib.list_local_devices()
    for device in local_devices:
        device.name = device.name.replace('/', 'device:')
    return local_devices

sess = K.get_session()
sess.list_devices = get_local_devices
```


```python
import os, sys, threading

import numpy as np
import tensorflow as tf

import keras
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.layers.core import RepeatVector, Permute
from keras.models import Model
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose, Input, Reshape, merge, concatenate
from keras.initializers import TruncatedNormal
from keras.optimizers import RMSprop

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.io import imsave

import matplotlib.pyplot as plt
```

We are training our model on a multi-GPU server and hence the data generator needs to be thread safe.
```threadsafe_generator``` below provides the functionality in the form of a decorator.

```batch_apply``` Applies function func to batch provided as numpy.ndarray

Architecture of Minsky:
* 2 Power-8 nodes with 10 Cores each
* Each Core supports upto 8 threads
* 4 Nvidia Tesla P100 GPUs, interconnnected with NVlinks and CPU:GPU NVlinks
* 1 TB of Flash RAM


```python
# Helper functions and classes

def batch_apply(ndarray, func, *args, **kwargs):
    """Calls func with samples, func should take ndarray as first positional argument"""

    batch = []
    for sample in ndarray:
        batch.append(func(sample, *args, **kwargs))
    return np.array(batch)


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g
```

### Model Pipeline Architecture <sup><a href="#References">[1]</a></sup> [citation](https://github.com/baldassarreFe/deep-koalarization/blob/master/paper.pdf)

<img src="resources/network.png" />

The model architecture is much like __Auto-Encoders__. The first part is an __Encoder__. The last part is __Decoder__. 

It's the central part, the fusion layer, which is a little different. Fusion layer takes output from the __Encoder__ and the embeddings generated by __Inception-ResNet-V2 model__ and concatenates both the outputs before continuing. This Inception-ResNet-V2 model is pre-trained on [ImageNet](http://www.image-net.org/) dataset.

The embeddings from Inception-ResNet-v2 are replicated as shown in diagram above to match output size of Encoder. 
> Replicating the embeddings also attaches the same information to all the pixel outputs of Encoder and hence is present spatially on the whole image. <sup><a href="#References">[1]</a></sup>

<hr/>
We will be using [Keras](http://keras.io) as higher level framework to build our model and [tensorflow](http://www.tensorflow.org) as it's backend.


Start by downloading the Inception-ResNet-V2 model from keras along with the model weights


```python
inception = InceptionResNetV2(weights='imagenet', include_top=True)
inception.graph = tf.get_default_graph()
```


```python
def create_inception_embedding(grayscaled_rgb):
    '''Takes (299, 299, 3) RGB and returns the embeddings(predicions) generated on the RGB image'''
    with inception.graph.as_default():
        embed = inception.predict(grayscaled_rgb)
    return embed
```

### Auto Encoder Graph

Create the auto-encoder in CPU memory. Doing this stores the weights of the model in CPU memory. These weights are shared among all the replicas for multiple GPU architecture, created later in the notebook.


```python
with tf.device('/cpu:0'):
    #Inputs
    embed_input = Input(shape=(1000,))
    encoder_input = Input(shape=(256, 256, 1,))
    
    #Encoder
    encoder_output = Conv2D(64, (3,3), activation='relu', padding='same', strides=2,
                            bias_initializer=TruncatedNormal(mean=0.0, stddev=0.05))(encoder_input)
    encoder_output = Conv2D(128, (3,3), activation='relu', padding='same',
                            bias_initializer=TruncatedNormal(mean=0.0, stddev=0.05))(encoder_output)
    encoder_output = Conv2D(128, (3,3), activation='relu', padding='same', strides=2,
                            bias_initializer=TruncatedNormal(mean=0.0, stddev=0.05))(encoder_output)
    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same',
                            bias_initializer=TruncatedNormal(mean=0.0, stddev=0.05))(encoder_output)
    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same', strides=2,
                            bias_initializer=TruncatedNormal(mean=0.0, stddev=0.05))(encoder_output)
    encoder_output = Conv2D(512, (3,3), activation='relu', padding='same',
                            bias_initializer=TruncatedNormal(mean=0.0, stddev=0.05))(encoder_output)
    encoder_output = Conv2D(512, (3,3), activation='relu', padding='same',
                            bias_initializer=TruncatedNormal(mean=0.0, stddev=0.05))(encoder_output)
    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same',
                            bias_initializer=TruncatedNormal(mean=0.0, stddev=0.05))(encoder_output)
    
    #Fusion
    fusion_output = RepeatVector(32 * 32)(embed_input) 
    fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
    fusion_output = concatenate([encoder_output, fusion_output], axis=3) 
    fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same',
                            bias_initializer=TruncatedNormal(mean=0.0, stddev=0.05))(fusion_output)
    
    #Decoder
    decoder_output = Conv2D(128, (3,3), activation='relu', padding='same',
                            bias_initializer=TruncatedNormal(mean=0.0, stddev=0.05))(fusion_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(64, (3,3), activation='relu', padding='same',
                            bias_initializer=TruncatedNormal(mean=0.0, stddev=0.05))(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(32, (3,3), activation='relu', padding='same',
                            bias_initializer=TruncatedNormal(mean=0.0, stddev=0.05))(decoder_output)
    decoder_output = Conv2D(16, (3,3), activation='relu', padding='same',
                            bias_initializer=TruncatedNormal(mean=0.0, stddev=0.05))(decoder_output)
    decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same',
                            bias_initializer=TruncatedNormal(mean=0.0, stddev=0.05))(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    
    model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)
```

__ImageDataGenerator__, a keras utility, provides real-time data augmentation by performing simple tranformations on the image. This also helps in generalizing the network towards shapes and orientations. It also increses the training data size.


```python
datagen = ImageDataGenerator(shear_range=0.2, zoom_range=0.2, rotation_range=20, horizontal_flip=True)
```

Images are converted from __```RGB```__ profile to __```CIE L*a*b*```__ color profile.<br/>
__```RGB```__ has three channels corresponding to the three primary colors (Red, Green and Blue).<br/>
While,<br/>
__```L*a*b*```__ has three channels Corresponding to L for lightness, a and b for the color spectra green–red and blue–yellow.

The advantage of using ```L*a*b*``` is that Lightness channel which is the black & white image and the color channels are separate. This gives us the convenience to combine the black & white input with the ```a*b*``` color channels at the output.

__Eg:__ <br />
<img src="http://shutha.org/sites/default/files//uploads/3_Courses/3_Digital_Imaging/02%20LAB1b.jpg" style="height:500px; display:inline-block; float:left;" />&nbsp;&nbsp;&nbsp;__becomes__&nbsp;&nbsp;&nbsp;
<img src="http://shutha.org/sites/default/files//uploads/3_Courses/3_Digital_Imaging/02%20LAB2b.jpg" style="height:500px; display:inline-block; float:right;" />

#### Color Space ranges:

``RGB``
```python 
{'R': [0, 255], 'G': [0, 255], 'B': [0, 255]} # type(int)
or
{'R': [0.0, 1.0], 'G': [0.0, 1.0], 'B': [0.0, 1.0]} # type(float)
```

``L*a*b*``
```python
{'L*': [0.0, 100.0], 'a*': [-128.0, 128.0], 'b*': [-128.0, 128.0]} # type(float)
```


```python
# Convert images to LAB format and resizes to 256 x 256 for Encoder input.
# Also, generates Inception-resnet embeddings and returns the processed batch

def process_images(rgb, input_size=(256, 256, 3), embed_size=(299, 299, 3)):
    """Takes RGB images in float representation and returns processed batch"""
    
    # Resize for embed and Convert to grayscale
    gray = gray2rgb(rgb2gray(rgb))
    gray = batch_apply(gray, resize, embed_size, mode='constant')
    # Zero-Center [-1, 1]
    gray = gray * 2 - 1
    # Generate embeddings
    embed = create_inception_embedding(gray)
    
    # Resize to input size of model
    re_batch = batch_apply(rgb, resize, input_size, mode='constant')
    # RGB => L*a*b*
    re_batch = batch_apply(re_batch, rgb2lab)
    
    # Extract L* into X, zero-center and normalize
    X_batch = re_batch[:,:,:,0]
    X_batch = X_batch/50 - 1
    X_batch = X_batch.reshape(X_batch.shape+(1,))
    
    # Extract a*b* into Y and normalize. Already zero-centered.
    Y_batch = re_batch[:,:,:,1:]
    Y_batch = Y_batch/128
    
    return [X_batch, embed], Y_batch
```


```python
# Generates augmented dataset and feed it to the model during training
@threadsafe_generator
def image_a_b_gen(images, batch_size):
    while True:
        for batch in datagen.flow(images, batch_size=batch_size):
            yield process_images(batch)
```

### Dataset

Size: __12519__ images from [ImageNet](http://www.image-net.org) published in [fall_11 urls](http://image-net.org/imagenet_data/urls/imagenet_fall11_urls.tgz)

Data Split
* Train - 0.7225 (9044 images)
* Validation - 0.1275 (1597 images)
* Test - 0.15 (1878 images)


```python
# Consist of 12k images from imagenet
DATASET = '../data/imagenet/'

# Get images file names
training_files, testing_files = train_test_split(shuffle(os.listdir(DATASET)), test_size=0.15)

def getImages(DATASET, filelist, transform_size=(299, 299, 3)):
    """Reads JPEG filelist from DATASET and returns float represtation of RGB [0.0, 1.0]"""
    img_list = []
    for filename in filelist:
        # Loads JPEG image and converts it to numpy float array.
        image_in = img_to_array(load_img(DATASET + filename))
        
        # [0.0, 255.0] => [0.0, 1.0]
        image_in = image_in/255
        
        if transform_size is not None:
            image_in = resize(image_in, transform_size, mode='reflect')

        img_list.append(image_in)
    img_list = np.array(img_list)
    
    return img_list
```

Convert our model to multi gpu model. This function replicates the model graph we created in CPU memory onto the number of GPUs provided.

We will be using __Mean Squared Error__ loss function to train the network.

Initial Learning rate set to 0.001


```python
model = multi_gpu_model(model, gpus=4)
model.compile(optimizer=RMSprop(lr=1e-3), loss='mse')
```

### Training

While training we keep track of improvements on our model using __Validation Loss__ metrics.

During training, this metric score triggers following routines:
* Save the model on improvement in score
* Reduce Learning Rate by a factor of 0.1 if no improvements for cosequtive few epochs
* Stop training if we are not seeing any improvement at all for considerable amount of time
<hr/>
<center>
    <div><b>Sample Input Images</b></div>
    <img src="resources/bnw_1.png" style="width:150px; height:150px; vertical-align:bottom; display:inline-block;"/>
    <img src="resources/bnw_2.png" style="width:150px; height:150px; vertical-align:bottom; display:inline-block;"/>
    <img src="resources/bnw_3.png" style="width:150px; height:150px; vertical-align:bottom; display:inline-block;"/>
    <img src="resources/bnw_4.png" style="width:150px; height:150px; vertical-align:bottom; display:inline-block;"/>
    <img src="resources/bnw_5.png" style="width:150px; height:150px; vertical-align:bottom; display:inline-block;"/>
</center>


```python
def train(model, training_files, batch_size=100, epochs=500, steps_per_epoch=50):
    """Trains the model"""
    training_set = getImages(DATASET, training_files)
    train_size = int(len(training_set)*0.85)
    train_images = training_set[:train_size]
    val_images = training_set[train_size:]
    val_steps = (len(val_images)//batch_size)
    print("Training samples:", train_size, "Validation samples:", len(val_images))

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, verbose=1, min_delta=1e-5),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, cooldown=0, verbose=1, min_lr=1e-8),
        ModelCheckpoint(monitor='val_loss', filepath='model_output/colorize.hdf5', verbose=1,
                         save_best_only=True, save_weights_only=True, mode='auto'),
        TensorBoard(log_dir='./logs', histogram_freq=10, batch_size=20, write_graph=True, write_grads=True,
                    write_images=False, embeddings_freq=0)
    ]

    model.fit_generator(image_a_b_gen(train_images, batch_size), epochs=epochs, steps_per_epoch=steps_per_epoch,
                        verbose=1, callbacks=callbacks, validation_data=process_images(val_images))
```

### Testing

We are not calculating test scores as of yet! This is becuase it doesn't give us much insight into how the actual image gets colored.

All the test images are passed through the model and are saved onto the disk for manual inspection.

The output from the model is first scaled back from normalized, zero-centered range to ``L*a*b*`` range.
This reproduced ``L*a*b*`` image is than converted back into RGB and saved onto disk.


```python
def test(model, testing_files, save_actual=False, save_gray=False):
    test_images = getImages(DATASET, testing_files)
    model.load_weights(filepath='model_output/colorize.hdf5')

    print('Preprocessing Images')
    X_test, Y_test = process_images(test_images)
    
    print('Predicting')
    # Test model
    output = model.predict(X_test)
    
    # Rescale a*b* back. [-1.0, 1.0] => [-128.0, 128.0]
    output = output * 128
    Y_test = Y_test * 128

    # Output colorizations
    for i in range(len(output)):
        name = testing_files[i].split(".")[0]
        print('Saving '+str(i)+"th image " + name + "_*.png")
        
        lightness = X_test[0][i][:,:,0]
        
        #Rescale L* back. [-1.0, 1.0] => [0.0, 100.0]
        lightness = (lightness + 1) * 50
        
        predicted = np.zeros((256, 256, 3))
        predicted[:,:,0] = lightness
        predicted[:,:,1:] = output[i]
        plt.imsave("result/predicted/" + name + ".jpeg", lab2rgb(predicted))
        
        if save_gray:
            bnw = np.zeros((256, 256, 3))
            bnw[:,:,0] = lightness
            plt.imsave("result/bnw/" + name + ".jpeg", lab2rgb(bnw))
        
        if save_actual:
            actual = np.zeros((256, 256, 3))
            actual[:,:,0] = lightness
            actual[:,:,1:] = Y_test[i]
            plt.imsave("result/actual/" + name + ".jpeg", lab2rgb(actual))
```


```python
# Executed when not running in ipython notebook environment
if not __IPython__ and __name__ == "__main__":
    if sys.argv[1] == "train":
        train(model, training_files, batch_size)
    elif sys.argv[1] == "test":
        test(model, testing_files)
```


```python
if __IPython__:
    train(model, training_files, epochs=100)
```

    /home/harsh/anaconda3/envs/datascience/lib/python3.6/site-packages/PIL/TiffImagePlugin.py:709: UserWarning: Corrupt EXIF data.  Expecting to read 12 bytes but only got 4. 
      warnings.warn(str(msg))
    /home/harsh/anaconda3/envs/datascience/lib/python3.6/site-packages/PIL/TiffImagePlugin.py:709: UserWarning: Corrupt EXIF data.  Expecting to read 4 bytes but only got 0. 
      warnings.warn(str(msg))


    Training samples: 9044 Validation samples: 1597
    Epoch 1/100
    49/50 [============================>.] - ETA: 10s - loss: 0.0937Epoch 00001: val_loss improved from inf to 0.01715, saving model to model_output/colorize.hdf5
    50/50 [==============================] - 551s 11s/step - loss: 0.0922 - val_loss: 0.0171
    Epoch 2/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0166 Epoch 00002: val_loss improved from 0.01715 to 0.01683, saving model to model_output/colorize.hdf5
    50/50 [==============================] - 469s 9s/step - loss: 0.0165 - val_loss: 0.0168
    Epoch 3/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0163 Epoch 00003: val_loss improved from 0.01683 to 0.01601, saving model to model_output/colorize.hdf5
    50/50 [==============================] - 502s 10s/step - loss: 0.0162 - val_loss: 0.0160
    Epoch 4/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0160 Epoch 00004: val_loss did not improve
    50/50 [==============================] - 491s 10s/step - loss: 0.0160 - val_loss: 0.0167
    Epoch 5/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0150 Epoch 00005: val_loss did not improve
    50/50 [==============================] - 500s 10s/step - loss: 0.0151 - val_loss: 0.0162
    Epoch 6/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0152 Epoch 00006: val_loss improved from 0.01601 to 0.01526, saving model to model_output/colorize.hdf5
    50/50 [==============================] - 496s 10s/step - loss: 0.0152 - val_loss: 0.0153
    Epoch 7/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0147 Epoch 00007: val_loss improved from 0.01526 to 0.01502, saving model to model_output/colorize.hdf5
    50/50 [==============================] - 500s 10s/step - loss: 0.0148 - val_loss: 0.0150
    Epoch 8/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0146 Epoch 00008: val_loss did not improve
    50/50 [==============================] - 493s 10s/step - loss: 0.0146 - val_loss: 0.0151
    Epoch 9/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0147 Epoch 00009: val_loss did not improve
    50/50 [==============================] - 499s 10s/step - loss: 0.0147 - val_loss: 0.0150
    Epoch 10/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0143 Epoch 00010: val_loss did not improve
    50/50 [==============================] - 493s 10s/step - loss: 0.0143 - val_loss: 0.0170
    Epoch 11/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0145 Epoch 00011: val_loss improved from 0.01502 to 0.01469, saving model to model_output/colorize.hdf5
    50/50 [==============================] - 532s 11s/step - loss: 0.0144 - val_loss: 0.0147
    Epoch 12/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0142 Epoch 00012: val_loss did not improve
    50/50 [==============================] - 474s 9s/step - loss: 0.0142 - val_loss: 0.0147
    Epoch 13/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0142 Epoch 00013: val_loss improved from 0.01469 to 0.01457, saving model to model_output/colorize.hdf5
    50/50 [==============================] - 492s 10s/step - loss: 0.0141 - val_loss: 0.0146
    Epoch 14/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0139 Epoch 00014: val_loss improved from 0.01457 to 0.01440, saving model to model_output/colorize.hdf5
    50/50 [==============================] - 500s 10s/step - loss: 0.0139 - val_loss: 0.0144
    Epoch 15/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0138 Epoch 00015: val_loss did not improve
    50/50 [==============================] - 492s 10s/step - loss: 0.0139 - val_loss: 0.0152
    Epoch 16/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0138 Epoch 00016: val_loss improved from 0.01440 to 0.01427, saving model to model_output/colorize.hdf5
    50/50 [==============================] - 498s 10s/step - loss: 0.0138 - val_loss: 0.0143
    Epoch 17/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0139 Epoch 00017: val_loss did not improve
    50/50 [==============================] - 492s 10s/step - loss: 0.0138 - val_loss: 0.0146
    Epoch 18/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0138 Epoch 00018: val_loss did not improve
    50/50 [==============================] - 499s 10s/step - loss: 0.0138 - val_loss: 0.0143
    Epoch 19/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0133 Epoch 00019: val_loss improved from 0.01427 to 0.01411, saving model to model_output/colorize.hdf5
    50/50 [==============================] - 493s 10s/step - loss: 0.0133 - val_loss: 0.0141
    Epoch 20/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0137 Epoch 00020: val_loss did not improve
    50/50 [==============================] - 498s 10s/step - loss: 0.0137 - val_loss: 0.0142
    Epoch 21/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0136 Epoch 00021: val_loss did not improve
    50/50 [==============================] - 532s 11s/step - loss: 0.0136 - val_loss: 0.0147
    Epoch 22/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0133 Epoch 00022: val_loss did not improve
    50/50 [==============================] - 470s 9s/step - loss: 0.0133 - val_loss: 0.0143
    Epoch 23/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0132 Epoch 00023: val_loss improved from 0.01411 to 0.01404, saving model to model_output/colorize.hdf5
    50/50 [==============================] - 498s 10s/step - loss: 0.0132 - val_loss: 0.0140
    Epoch 24/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0131 Epoch 00024: val_loss did not improve
    50/50 [==============================] - 493s 10s/step - loss: 0.0131 - val_loss: 0.0141
    Epoch 25/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0129 
    Epoch 00025: reducing learning rate to 0.00010000000474974513.
    Epoch 00025: val_loss did not improve
    50/50 [==============================] - 500s 10s/step - loss: 0.0129 - val_loss: 0.0142
    Epoch 26/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0130 Epoch 00026: val_loss improved from 0.01404 to 0.01394, saving model to model_output/colorize.hdf5
    50/50 [==============================] - 499s 10s/step - loss: 0.0130 - val_loss: 0.0139
    Epoch 27/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0125 Epoch 00027: val_loss improved from 0.01394 to 0.01383, saving model to model_output/colorize.hdf5
    50/50 [==============================] - 497s 10s/step - loss: 0.0124 - val_loss: 0.0138
    Epoch 28/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0125 Epoch 00028: val_loss improved from 0.01383 to 0.01378, saving model to model_output/colorize.hdf5
    50/50 [==============================] - 495s 10s/step - loss: 0.0126 - val_loss: 0.0138
    Epoch 29/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0124 Epoch 00029: val_loss improved from 0.01378 to 0.01375, saving model to model_output/colorize.hdf5
    50/50 [==============================] - 498s 10s/step - loss: 0.0124 - val_loss: 0.0138
    Epoch 30/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0118 Epoch 00030: val_loss improved from 0.01375 to 0.01374, saving model to model_output/colorize.hdf5
    50/50 [==============================] - 493s 10s/step - loss: 0.0118 - val_loss: 0.0137
    Epoch 31/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0126 Epoch 00031: val_loss did not improve
    50/50 [==============================] - 535s 11s/step - loss: 0.0126 - val_loss: 0.0138
    Epoch 32/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0124 Epoch 00032: val_loss did not improve
    50/50 [==============================] - 477s 10s/step - loss: 0.0124 - val_loss: 0.0138
    Epoch 33/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0121 
    Epoch 00033: reducing learning rate to 1.0000000474974514e-05.
    Epoch 00033: val_loss improved from 0.01374 to 0.01373, saving model to model_output/colorize.hdf5
    50/50 [==============================] - 492s 10s/step - loss: 0.0121 - val_loss: 0.0137
    Epoch 34/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0121 Epoch 00034: val_loss did not improve
    50/50 [==============================] - 501s 10s/step - loss: 0.0121 - val_loss: 0.0137
    Epoch 35/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0122 Epoch 00035: val_loss did not improve
    50/50 [==============================] - 493s 10s/step - loss: 0.0121 - val_loss: 0.0137
    Epoch 36/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0119 Epoch 00036: val_loss did not improve
    50/50 [==============================] - 498s 10s/step - loss: 0.0119 - val_loss: 0.0137
    Epoch 37/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0122 Epoch 00037: val_loss did not improve
    50/50 [==============================] - 492s 10s/step - loss: 0.0122 - val_loss: 0.0137
    Epoch 38/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0120 
    Epoch 00038: reducing learning rate to 1.0000000656873453e-06.
    Epoch 00038: val_loss did not improve
    50/50 [==============================] - 498s 10s/step - loss: 0.0120 - val_loss: 0.0137
    Epoch 39/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0123 Epoch 00039: val_loss did not improve
    50/50 [==============================] - 493s 10s/step - loss: 0.0122 - val_loss: 0.0137
    Epoch 40/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0121 Epoch 00040: val_loss did not improve
    50/50 [==============================] - 496s 10s/step - loss: 0.0121 - val_loss: 0.0137
    Epoch 41/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0120 Epoch 00041: val_loss did not improve
    50/50 [==============================] - 533s 11s/step - loss: 0.0120 - val_loss: 0.0137
    Epoch 42/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0122 Epoch 00042: val_loss did not improve
    50/50 [==============================] - 471s 9s/step - loss: 0.0122 - val_loss: 0.0137
    Epoch 43/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0123 
    Epoch 00043: reducing learning rate to 1.0000001111620805e-07.
    Epoch 00043: val_loss did not improve
    50/50 [==============================] - 497s 10s/step - loss: 0.0123 - val_loss: 0.0137
    Epoch 44/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0120 Epoch 00044: val_loss did not improve
    50/50 [==============================] - 494s 10s/step - loss: 0.0119 - val_loss: 0.0137
    Epoch 45/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0123 Epoch 00045: val_loss did not improve
    50/50 [==============================] - 501s 10s/step - loss: 0.0123 - val_loss: 0.0137
    Epoch 46/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0121 Epoch 00046: val_loss did not improve
    50/50 [==============================] - 495s 10s/step - loss: 0.0122 - val_loss: 0.0137
    Epoch 47/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0120 Epoch 00047: val_loss did not improve
    50/50 [==============================] - 498s 10s/step - loss: 0.0121 - val_loss: 0.0137
    Epoch 48/100
    49/50 [============================>.] - ETA: 9s - loss: 0.0125 
    Epoch 00048: reducing learning rate to 1.000000082740371e-08.
    Epoch 00048: val_loss did not improve
    50/50 [==============================] - 494s 10s/step - loss: 0.0125 - val_loss: 0.0137
    Epoch 00048: early stopping



```python
if __IPython__:
    test(model, testing_files, True, True)
```

    /home/harsh/anaconda3/envs/datascience/lib/python3.6/site-packages/PIL/TiffImagePlugin.py:709: UserWarning: Corrupt EXIF data.  Expecting to read 4 bytes but only got 0. 
      warnings.warn(str(msg))


    Preprocessing Images
    Predicting
    Saving 0th image c31e46b81126940ff23ccd993cd64325_*.png
    Saving 1th image 1024f822baa6cd5b9008a3778819f9fd_*.png
    Saving 2th image e8e63bc18816f11ec9ca21da653d406c_*.png
    Saving 3th image d88755f9e33954cc34eb81711f17fdcf_*.png
    Saving 4th image 7967556802bde7eb2946950ce5294285_*.png
    Saving 5th image 1d4b5ff464846133b74bbcdfff2c0e5f_*.png
    Saving 6th image 7b90a863f321902b0ce8abf9f08b7f4e_*.png
    Saving 7th image 68fc08518b886770249fbf64c57a421e_*.png
    Saving 8th image e06c022833b32e50ec9ce200247e6bff_*.png
    Saving 9th image 98301e8d62f3390b67de4cb8d79d07a6_*.png
    Saving 10th image f6a57a4b7db082acd038b69a156ba5d4_*.png
    Saving 11th image 4579ae362b1967ee8350d58bf2ef3665_*.png
    Saving 12th image d7c5cb13634d1f55bdfe0c628709f412_*.png
    Saving 13th image 6bf92d50b8375b3acbcc890c5d6f22d1_*.png
    Saving 14th image 895430c6d361af568ca8f045f1e1ef68_*.png
    Saving 15th image 5ce406f210c1869d6f5c2cf39f3cb1db_*.png
    Saving 16th image 653a8985bc1813e4edf23f5097a9ae27_*.png
    Saving 17th image 518980e2da4c0a69451bd793081dd27e_*.png
    Saving 18th image 6f93052c8f660a2765d4cd1a3af5813d_*.png
    Saving 19th image dc5d9e285e72fcb84961d1495fb5cd85_*.png
    Saving 20th image 197b24634b59d30917e438b89ad9e1a8_*.png
    Saving 21th image 19df20673d9714f5ac9f5ea5b9d2c076_*.png
    Saving 22th image c2165011de8e89b09c0899d462faa623_*.png
    Saving 23th image 00dd886aea0762196556119f3b6d3e7d_*.png
    Saving 24th image e899366c66547cbd4c1370e214930524_*.png
    Saving 25th image ea5fccd92dff25d57d7d7c69e59ffb63_*.png
    Saving 26th image d797ffc26ee7b0a765680e8c85cf0927_*.png
    Saving 27th image 23695794895e1814d70cb9d0e1a77f80_*.png
    Saving 28th image 129092566ce8feb06b490b6ed8bb5ec6_*.png
    Saving 29th image 120197b624f9e1c5c2dbebccfb548b06_*.png
    Saving 30th image 01ed512d8d9754589de8d843c040a9e6_*.png
    Saving 31th image 03365a00818b1db51c640550c4b1bd8d_*.png
    Saving 32th image 33bcca00a00d246e6d61658245ea34ec_*.png
    Saving 33th image 431dd813251abcd04db519bcd242f7d5_*.png
    Saving 34th image 7e10a15ff8f77ef41e50772dbb154b32_*.png
    Saving 35th image dde447a48c1f00f211ffb92f32a8b84e_*.png
    Saving 36th image f83d11eeb61cac28fa81cadb03a36b2d_*.png
    Saving 37th image d2609c64e5edfb582db00f7002130a4b_*.png
    Saving 38th image 4f6ee41dc5c37b1cef75939076a8e149_*.png
    Saving 39th image 9a2b164ac6f2255d0d8a3a444c87b883_*.png
    Saving 40th image 93ea75fd7ca41f5828fa25dda0c5c615_*.png
    Saving 41th image c8c119ab1577508ae4fa3f6ea2522f27_*.png
    Saving 42th image a63cce1fccc3533965b24827996d4881_*.png
    Saving 43th image cb05b5c857a488710061764625418cd5_*.png
    Saving 44th image 3a3015abf44c335008879c6ce5228437_*.png
    Saving 45th image ee481bd2c0e81eea7ae83f0ffcd6be65_*.png
    Saving 46th image 1c331b121c0e0178871e56278a6cb728_*.png
    Saving 47th image af16164f12a58881b625f3b0fa006627_*.png
    Saving 48th image ce4b372585411eced5369556be6599b8_*.png
    Saving 49th image 04f989e8dc2dd8fcc7e5a98c651ba6f6_*.png
    Saving 50th image 3b3d56e833e332dbb00fb8e825253b2c_*.png
    Saving 51th image 12dec6697454a8bee5379d2cfc3d5231_*.png
    Saving 52th image 6955b4bf76f476d1d5d79b0cd5bc9657_*.png
    Saving 53th image 0133ef5e0eccd0a1197a0b0278509bea_*.png
    Saving 54th image 17d5bcddb8febbf518699eae186d603b_*.png
    Saving 55th image 8f22645275fdd7b6c3265640e9111d94_*.png
    Saving 56th image fc44846827d22bb8b56b52250ae5e925_*.png
    Saving 57th image d847d2e42a5f71f73b97db4cdccd56d8_*.png
    Saving 58th image b49a8ca5f64764f5cdb5abcb3d488caf_*.png
    Saving 59th image 4a21b5f3407432033e824b65de9fa78a_*.png
    Saving 60th image e82ad9cdc57fc93e0caf6d031e684fc4_*.png
    Saving 61th image e251dd7111b009f2dd4d2b7160750806_*.png
    Saving 62th image a95d8428805b48cac3fa15bd92216efa_*.png
    Saving 63th image 5aa593555e2dfd63f2777171a70a2678_*.png
    Saving 64th image a59f3280628d2a26b0624bd3c673bada_*.png
    Saving 65th image 9a38809b4d735235d08d5f02a810e756_*.png
    Saving 66th image 6082fb852e7f6c9c7a70b523c907522c_*.png
    Saving 67th image b4f534bc5cbf69384c3001525b0d42a6_*.png
    Saving 68th image cd28ab7a640335c3a3af72df06ca015d_*.png
    Saving 69th image 3e76c0b50233d5bb0917b2a3cc3fb157_*.png
    Saving 70th image 77de59c29ff122938c27b6bf7a9e28c4_*.png
    Saving 71th image bccbd785c645fd55910decc2052e4cb2_*.png
    Saving 72th image 6bd834794159780c9d6a056785db935d_*.png
    Saving 73th image 23102eb6cf7e7bdb63ed5290c510c18d_*.png
    Saving 74th image b7c178c21b5a55cd759ea0e684c15073_*.png
    Saving 75th image 90ee97ac277d6b7ea54b72e0da7c3306_*.png
    Saving 76th image befb8aca0ae285c2c44153a09a97d16c_*.png
    Saving 77th image af5bc1be9c497e4a3e732822c4b2aa54_*.png
    Saving 78th image 16602a3e7206dfc216945126cbfa5f8d_*.png
    Saving 79th image 516ec798b56e82ef998602373b326066_*.png


    /home/harsh/anaconda3/envs/datascience/lib/python3.6/site-packages/skimage/color/colorconv.py:985: UserWarning: Color data out of range: Z < 0 in 1 pixels
      warn('Color data out of range: Z < 0 in %s pixels' % invalid[0].size)


    Saving 80th image 589b76029527ca00e406be627b7f7840_*.png
    Saving 81th image b38b464263477f59f7003170c4140a7a_*.png
    Saving 82th image 5c7f94d80dc7569471310563160fc08d_*.png
    Saving 83th image b91339bfa546d79b39a176145dc964f4_*.png
    Saving 84th image 19573bd8d027a3e98e4b1352c2a8f42d_*.png
    Saving 85th image 2c317fb561e9ec54c90228d44c07587a_*.png
    Saving 86th image 619cc07d443fbe2ca710c57deb89c98e_*.png
    Saving 87th image 68c43c7c7989d141ec2ccba01db7d06d_*.png
    Saving 88th image 83e221e6e0d884bd2aba21f60a270404_*.png
    Saving 89th image ed114144b4aa27d20fde5ed710dd1151_*.png
    Saving 90th image 325735e6d48f3601b1130d3092fdb8f6_*.png
    Saving 91th image 14737b2baa34cd53a1e5896c60d15d82_*.png
    Saving 92th image a6b359ecadba1cd848901f7d03934dac_*.png
    Saving 93th image 1f77a6686923d835c2337c8e72511786_*.png
    Saving 94th image 2dc6e0e5f93a7fc4c2fa2f9d8d9f0bb7_*.png
    Saving 95th image 3a5099d82ad629625bb35faab69c5524_*.png
    Saving 96th image 524b4be34b61ccc7b2f944a57b8ea832_*.png
    Saving 97th image 708c9e2f2f620375be819e3154c12c97_*.png
    Saving 98th image cd63b2c3aa695a96479308e096f9e4e3_*.png
    Saving 99th image 0d9fc753e198f6c13fb6c156fb04547e_*.png
    Saving 100th image 2d8bb48f02dfc73b592754e7bdf9c3f5_*.png
    Saving 101th image afaad8869e97dc172c4bf2f879ff46aa_*.png
    Saving 102th image b09844d414d8a30926a1d125cc231ca7_*.png
    Saving 103th image b962383802e644fff6305d045b13ccd4_*.png
    Saving 104th image e23f03553cab017c7948c9c6b1276ffd_*.png
    Saving 105th image 1e9ce7c519326af4cbcab310fe16c07f_*.png
    Saving 106th image 6c63cccf27b2c0d5e766db250be94965_*.png
    Saving 107th image dc2fe005e2ce54cf06d07a6de1e8b98d_*.png
    Saving 108th image d059f7268ecf77ff7a00c52bf96af635_*.png
    Saving 109th image 756a9d98bd2cb44e08d4a54b1f2d20bf_*.png
    Saving 110th image c97d4f7b0bc2997a7903a50ef6a427a7_*.png
    Saving 111th image cf232615d41da865f43367f1d81db190_*.png
    Saving 112th image d69a9564050cc9c2d6e551940a25d467_*.png
    Saving 113th image bed2cc74ed2932f0476b952691fb6b59_*.png
    Saving 114th image 6f0a95130d8c4027ab49273ed681f53d_*.png
    Saving 115th image 8fcfea94dae2fc9b299780e580d2b251_*.png
    Saving 116th image 2814e8c78d96271ed1ad09b9aaa06d05_*.png
    Saving 117th image c7c98dac9868f39c039f4e55e81222f1_*.png
    Saving 118th image 90223c1e5a0e0439839ccbfca465bc2f_*.png
    Saving 119th image e216afa287920d8b7fc55d1c550f8c76_*.png
    Saving 120th image 1ed3104108ade455389073fa9684607b_*.png
    Saving 121th image a501215aa55424f50277d799ba113317_*.png
    Saving 122th image b2046b8416b6d78d675bf63901500bbf_*.png
    Saving 123th image 51304f18213b55e72c8e901e07d71002_*.png
    Saving 124th image 289d6914f64c51c22c096f7c3fad5c72_*.png
    Saving 125th image 1db7ebe2d961e64dd36496717480ac4a_*.png
    Saving 126th image e0fc38c273e1e31675ffb31070c2587a_*.png
    Saving 127th image 96077c1606951775966963e3e2169d4d_*.png
    Saving 128th image 7b15ccc71783ddb04ee80355f1e1a2d1_*.png
    Saving 129th image d85c33380c1d451004a8548ecfde78b2_*.png
    Saving 130th image 925b898fee4e302d438a04d335cf7ab1_*.png
    Saving 131th image be52170389870c89c7104dc702e163dd_*.png
    Saving 132th image 0878f5f30ca78c5940097b17e5269bf0_*.png
    Saving 133th image 745249cc3a8426579a3ce40b119b6e28_*.png
    Saving 134th image 8e231df9eaf2b300182e96072d9ce19f_*.png
    Saving 135th image da1f8c362fe989bdfda7fb2c8bc3f589_*.png
    Saving 136th image be12cb4ba6bb0e4d163385efe183c7d1_*.png
    Saving 137th image 6cb77edc53b0c4b76d41beb3a3ce2281_*.png
    Saving 138th image 85a9f2389b7fdc0393e23eb7bb29d283_*.png
    Saving 139th image 908471557bdf80384e9f64c794ba5eba_*.png
    Saving 140th image d1e48e9ea4f77996381211a8912fa71b_*.png
    Saving 141th image 9858952b8ffd29fb50a7f5fc08cf8a6e_*.png
    Saving 142th image f81565f315cbc089076fa8744f9112da_*.png
    Saving 143th image 68f976bebf146b3a98b16fcded93f340_*.png
    Saving 144th image 6bb3b598b434a6afbc0ac3ea11bece59_*.png
    Saving 145th image 0b3a50e7b93a18ca702c2f998499b481_*.png
    Saving 146th image a3a04cd0bb7b428d21d2255981a1e798_*.png
    Saving 147th image c04115a8643c9cfb47208bf72192329c_*.png
    Saving 148th image 35d791c0e3047dbd84d5b5ff60b12ab7_*.png
    Saving 149th image dc01d6b661d1b13b309085e0e919caab_*.png
    Saving 150th image a90bd7571a7a6c4abbd35727608adf53_*.png
    Saving 151th image 7e80458b195c8458230cc42e7bb28c8f_*.png
    Saving 152th image 8195f7036a37af544b995ad48454ea2e_*.png
    Saving 153th image 29678b05417ab834ee34625975c86a59_*.png
    Saving 154th image 851152898de5cc38220c31b619dfca7f_*.png
    Saving 155th image 61f3d7db888425ae9dbc2651abf176d0_*.png
    Saving 156th image 9c05493e6ddb0f53addc0ab0a962fb74_*.png
    Saving 157th image 8f05e6127dbfad77589a33ccf71a5b3a_*.png
    Saving 158th image 0e6e09516b24eb8972fdd85b20d2fcf9_*.png
    Saving 159th image 5e206219e2955bd151186544c0eae234_*.png
    Saving 160th image 105234bcd48bf00e76a52884e9d2126b_*.png
    Saving 161th image 2ecc7d36bf1fb4497ef2ff8cf259fff7_*.png
    Saving 162th image 5f40b8a0e3f0c65139e53e29eccadd54_*.png
    Saving 163th image 38180bbb6730f4a8c8e9f166256ba194_*.png
    Saving 164th image 2400630f96085926c98f9621c63e59cb_*.png
    Saving 165th image 78f8607bb617b25c470d0d7dd2893d88_*.png
    Saving 166th image e8b887b0b3e4f63c97c3e224bc8de233_*.png
    Saving 167th image 9b255ec52c7befe76b74a38738bb7358_*.png
    Saving 168th image 6c95416e8c085b785c258d8c03a87d5d_*.png
    Saving 169th image 592dbbf7af82c6e7a5eb5baf3e6f15b5_*.png
    Saving 170th image e9c24884841b913fb96761a790b39320_*.png
    Saving 171th image 64aca04598090d678b2202b7d41e037b_*.png
    Saving 172th image c5697d87ed2b8d9a0fc788a3464ac2e7_*.png
    Saving 173th image c8adce2308284162cd7f0635fa8d62cb_*.png
    Saving 174th image 5cefa083542440eb14007d9a652185a7_*.png
    Saving 175th image 41f1b51f0f7200f2483305d049435e73_*.png
    Saving 176th image 595b1e0375ad76bc438d7c62418ca668_*.png
    Saving 177th image 7f69b2e0b592f8ae2d1a15b83245a945_*.png
    Saving 178th image ed445b9dcb8e96ea9f47dc663deff2a8_*.png
    Saving 179th image 667c57fc06f3dfd18460b799cf0c45bd_*.png
    Saving 180th image b6ed4d851c86627343776a53a5bb6374_*.png
    Saving 181th image b9bfdeb274dea333a8936ca2debd49ef_*.png
    Saving 182th image 614df4e42a113d87ffb4e0b1f37283af_*.png
    Saving 183th image 367d4ec28fab3933fdd390bdeb960036_*.png
    Saving 184th image 856affb824a657ae762cefaaf2f329dd_*.png
    Saving 185th image 246638df60e5be3d0b6241165a08fa2b_*.png
    Saving 186th image 65005232fb5743c820705b8c2c2047a6_*.png
    Saving 187th image b54e66960b9422c2c77133830afdc579_*.png
    Saving 188th image 33ce4854ad811737188d28b31c0f3bec_*.png
    Saving 189th image 0b4281ec3bb11c94e7e4622bc9025ff9_*.png
    Saving 190th image 2327d6ca759f507fcdb189fc77212c40_*.png
    Saving 191th image 49007387628e705b11a3ab13be219d8b_*.png
    Saving 192th image d08a51fd30d463905cd14f4bc6c38f3f_*.png
    Saving 193th image 71cae9bc70e97a0176b277651cfd8f98_*.png
    Saving 194th image 468bbf1542978c42b04296c51bcf05ab_*.png
    Saving 195th image 9759e974096124e63ea9ae58565bb45d_*.png
    Saving 196th image cd1296fa417eadce083240f79e9608de_*.png
    Saving 197th image 6bcf5ebc23f59c3079f497fef158da7c_*.png
    Saving 198th image cf3fe1ca652b30019dc8d270699c735a_*.png
    Saving 199th image ec331164da424271d5a3e87fea725b7d_*.png
    Saving 200th image 496d0f307e4bf6e19cbb85cc60fa2bde_*.png
    Saving 201th image da30a5788a003ae296026eebb65f168c_*.png
    Saving 202th image e2429ae10906d222f7374193ed285bf2_*.png
    Saving 203th image d3ca9c9c8a13271f7acaf16bcc602d34_*.png
    Saving 204th image 0339c4efbd3616ff548e4418e2dc108d_*.png
    Saving 205th image 7871976aa413df5b5cc678bff284cba8_*.png
    Saving 206th image a6b061cc0b35e6df6605f164ce303030_*.png
    Saving 207th image 550aed6ef5eb035ad06b90e011e4db34_*.png
    Saving 208th image 9273c87dd59ee03bfda9b9d139855b78_*.png
    Saving 209th image d8eeef5e5467e37f6f1506ce0ec88a54_*.png
    Saving 210th image c97e42908d6e0c760e75a68466601cc9_*.png
    Saving 211th image ccbf252d83afa5508897284495d88f88_*.png
    Saving 212th image bf46d68bd9ee20108f789ba3b3985f08_*.png
    Saving 213th image f9d54f074f52bb55138078ab661985fb_*.png
    Saving 214th image 5d3f25175fec98764cf2db814174473f_*.png
    Saving 215th image ceb9965439e17559da4d5e6de2ca11e8_*.png
    Saving 216th image c5ff2e69fb00450340bd0951c038788f_*.png
    Saving 217th image c4e78a69f27e684a391a1e484c6747b9_*.png
    Saving 218th image 874150e8c1b502e95a88ad07d2945ef8_*.png
    Saving 219th image 1683523dd95527fea9c9c2fbf5e059b1_*.png
    Saving 220th image c2f98e0754e644e0bfc136e86972aea0_*.png
    Saving 221th image be46d93e84602352f60c681e5a895318_*.png
    Saving 222th image 005763e15873128dc4b20bf6ff371b82_*.png
    Saving 223th image 4e098f14f268522c2fc6e492c6d8e019_*.png
    Saving 224th image 42805369f30656d72d1adb8e0734b1c1_*.png
    Saving 225th image d7138ece3f52e7df84c713d31be59e15_*.png
    Saving 226th image 1d47ca5ca6a082e5a936a485c1c736e0_*.png
    Saving 227th image 7ea4aa995ea7a30005efecaf8c6488a2_*.png
    Saving 228th image 62f5c99fdbdcb00d73de1f82bcdf74e4_*.png
    Saving 229th image bbe144cb8e490e26a470be4a287dc25e_*.png
    Saving 230th image 7fc360095e950c4d261641324e8ff3f2_*.png
    Saving 231th image 17020303ae3398fd9bd4e3c54d862fa6_*.png
    Saving 232th image 71d190df735f37684ea326f6b6fe42b3_*.png
    Saving 233th image d2d15c0cce0eb7046108da933562a84f_*.png
    Saving 234th image 384febbc6005c0c35964ae41350813c7_*.png
    Saving 235th image 263b0da3a953e6c7de642e757ce9f783_*.png
    Saving 236th image a2c338a2dfab74f5cd45cb1a8097b0a3_*.png
    Saving 237th image 5d75bfea017dd43f0aaddc82fe30727a_*.png
    Saving 238th image 30b5337803a54d5f1e833ea49108ca30_*.png
    Saving 239th image 521a0cfa16fec19121defc677256cef6_*.png
    Saving 240th image 9cf148a6d49208587b191740a688a58e_*.png
    Saving 241th image ac943892c31ac184180ef6e20dedd526_*.png
    Saving 242th image aa9cc205a125bf08a629caf916fd0d94_*.png
    Saving 243th image 2eb409b826d7073183334abe373decb7_*.png
    Saving 244th image 1fce8884a5b25931f28e77d0557e2844_*.png
    Saving 245th image e999ce8305df0796027594549151656a_*.png
    Saving 246th image 71053e520d51052061f380c41e1cd26a_*.png
    Saving 247th image 7a0cd3c77ea10404bff7ccc51274a5ff_*.png
    Saving 248th image 6e42e060cdbc601e8e06a3461cb33d94_*.png
    Saving 249th image c4e94db705be78708014747417b25be7_*.png
    Saving 250th image 8a5c29265f4c3f2fe795daa74897bc82_*.png
    Saving 251th image acc1a3077d71ed315eb437f0219412e5_*.png
    Saving 252th image 7197e66fb6a3bd1b53b87d69a09f2712_*.png
    Saving 253th image d7755b5909399e271bc7bda05c9eb6e9_*.png
    Saving 254th image 64d5482f0b98699f5cc8b2c25162aa81_*.png
    Saving 255th image 5a2a5ee54628863ac6f1fa09a8d93a8d_*.png
    Saving 256th image bdb35f47d05f2e8c66a0e44d305bdb06_*.png
    Saving 257th image d1c18fd29bf11960caa862967c9928e0_*.png
    Saving 258th image cf1fc3349edcdb90984c6918c56f5bb4_*.png
    Saving 259th image 5b943bf67b605f62b877fd6147206941_*.png
    Saving 260th image 21177df2996e65404547c752e527c470_*.png
    Saving 261th image 65f0a4f42d0bf5fbb00db5dd3d5a7f57_*.png
    Saving 262th image ff7e98b74f81a1f94557b9ca414abaf4_*.png
    Saving 263th image 28ceb76ced9cd7c09b2b8afd21dcfb7e_*.png
    Saving 264th image 07af73b1abdbfb7454e444d16fcd13de_*.png
    Saving 265th image 4d25e2cb664137d6ee607c77acb3f99a_*.png
    Saving 266th image b8a6757652a3b91e83b38835f2dc03ea_*.png
    Saving 267th image 2282bf1bfeb9137b255c5e28894a0aa6_*.png
    Saving 268th image 2a11b4a02108626d3e6c0a8f1be89b0c_*.png
    Saving 269th image b9bb19cf74f91f2fa0714561129fd5e6_*.png
    Saving 270th image 6addce3851f49d24b2b4931f31ce2219_*.png
    Saving 271th image eb369fe7646459493f582087e48b69df_*.png
    Saving 272th image 2e3752622449ee325d6e1dd496b90490_*.png
    Saving 273th image 89cb9558fd8d3edd443df618b2e599d5_*.png
    Saving 274th image 026a74348fae1d89c694b9a856c1e02f_*.png
    Saving 275th image d651dc39b9042e651f3b2eb91fe35de7_*.png
    Saving 276th image 433d978913179ca1ac65e7e21363060b_*.png
    Saving 277th image 852ee889a2d5393b22e3b2d26738c0f1_*.png
    Saving 278th image d6a9366e3456f8a3ccb759e3ee70a887_*.png
    Saving 279th image 836eb4267c2d28554433597097624b79_*.png
    Saving 280th image 14fd40cc9dd6fae3f786d6e715c146b5_*.png
    Saving 281th image 0f541b5ad25e24c3f7ad3c483b86a772_*.png
    Saving 282th image d77f08beb0794313f0e702d128b00094_*.png
    Saving 283th image 45a9796f1dda7e03f35e2877597e439c_*.png
    Saving 284th image 30d5b37a7d2203a67888c5861fbb6616_*.png
    Saving 285th image 6801a6bf3f5b30f3af3c5585281f1ad6_*.png
    Saving 286th image 22ddcea860661730f924f4eece032640_*.png
    Saving 287th image 974a961a3b6c78531c36d259c698bef4_*.png
    Saving 288th image 20d211f4e54109477d7c154ea3f8ddf0_*.png
    Saving 289th image 32fb9352bb8ec69009aa7674221c9f69_*.png
    Saving 290th image 931bf69a9b0b689da075d502496eef7e_*.png
    Saving 291th image 27fcf66391aa819a89c3444b9377e754_*.png
    Saving 292th image f7eb8a5d4601a6bd3008bb375d11661a_*.png
    Saving 293th image 966a93c32be744910af2eafbd7cf0164_*.png
    Saving 294th image df8e76507fdf40481cb775917051f18d_*.png
    Saving 295th image be7cc3e065ab5ce337a12842701b4c5c_*.png
    Saving 296th image f6a6249fe1ff2c268ca3556cb453ddb2_*.png
    Saving 297th image c25b8ddf2f19bfa41d3a45052800e4d2_*.png
    Saving 298th image 15c80c5c6499be21011f04a91ff60589_*.png
    Saving 299th image 928a5de9f1eb98a6e7ba832c9598851b_*.png
    Saving 300th image 4628674a071ef5f4e4f9619fbe832d59_*.png
    Saving 301th image 6d80e06342504f47371aa62943a8ab8d_*.png
    Saving 302th image abde27a4a77df1493b8bc147b17cfb74_*.png
    Saving 303th image 5cb10849d4f17942986b8d1dc704fac2_*.png
    Saving 304th image 6aa367454401e23501947e2ec57d7ad4_*.png
    Saving 305th image 9b7440d7f9c6a9a371bea1a0a704022c_*.png
    Saving 306th image b3f4498dbb6ceff2caadde0eb0be5cb3_*.png
    Saving 307th image 59f7004fd4e9b3a9e409ae2e616b1e27_*.png
    Saving 308th image 6b03a993af21ca5d1d58a763380aac0a_*.png
    Saving 309th image f96fae3506a64ae3c7894c6514dae38b_*.png
    Saving 310th image e96f703275b392548ba11225d893063c_*.png
    Saving 311th image 3fadcee982e1f89c45f922d7d42beb2f_*.png
    Saving 312th image 46bc5fadcf97fa9bf4b45a5d50a4f9dc_*.png
    Saving 313th image 988ae1ef8619687186c928bafdd524bc_*.png
    Saving 314th image 57d1e2262f0a2d47c698d5ca15b9dd63_*.png
    Saving 315th image e57393e0f0dd5d7fa2bb3c20d22f9c41_*.png
    Saving 316th image de19ec1d09ab61359ae99cb6303d765e_*.png
    Saving 317th image d38721895f1c533afcc09ffa205facc3_*.png
    Saving 318th image b8e8b8f9896005faafdf1d249bec6617_*.png
    Saving 319th image aa0f69948227aa50ef79e7eef709dcc5_*.png
    Saving 320th image ec579ffcf08f92d2d7982fdd122e7199_*.png
    Saving 321th image 58e34269c133aef4611936a8e5c440a1_*.png
    Saving 322th image e48ce50ecbc24d05317540c127da3fbd_*.png
    Saving 323th image 4d4b80f3e1abc9c1fc691b75fc5d221e_*.png
    Saving 324th image fe04aaa636bf10f66198508c83c6960a_*.png
    Saving 325th image 296c5f7d70c1af6b9c84ba468e04f27f_*.png
    Saving 326th image 0e94c6f2c240acfa1bf2f3a84979a9eb_*.png
    Saving 327th image fdf1fe556eee94f716dbf4369071702e_*.png
    Saving 328th image cc5512c3da0e23f4db41a363f9e7292b_*.png
    Saving 329th image 972908be25ca9688d32339e8259cfc84_*.png
    Saving 330th image ca4f7952668f64734ccd6750f7469572_*.png
    Saving 331th image 8ee594756f0c4e08584a3df41f376ef6_*.png
    Saving 332th image 9766bb6aa6be96b43a234a8088760b8b_*.png
    Saving 333th image 5766afcc7c5856bdb551364e17db79f3_*.png
    Saving 334th image a3a30a0563ad35e785d0eff466a2c992_*.png
    Saving 335th image 066b29ec717259b30d9599f8219d6b81_*.png
    Saving 336th image 09a3074f2cbe13b3147341b67ef30a60_*.png
    Saving 337th image 8a33fa0efd6f87a14abb2734ae1252c7_*.png
    Saving 338th image 984a4794d30de5e9e3b53b653b713aec_*.png
    Saving 339th image 333d5eb1f23c367fe477a53e015e59f7_*.png
    Saving 340th image 8e65e7c59fc6e2a54a8cfbce13232d75_*.png
    Saving 341th image ecc9738c51ec63a36d853248ac1dc1c2_*.png
    Saving 342th image 3f2d5986e2063290b273e25a024c1e8a_*.png
    Saving 343th image d418d242dee5074102bfc7dd776ec524_*.png
    Saving 344th image 286c89fec841d3116d1cf86f578f28a6_*.png
    Saving 345th image ec106f5d8adc560540b042904d37bb4f_*.png
    Saving 346th image b0d2b462badd1c455a158b32f55a57b5_*.png
    Saving 347th image 43b9ae498bbb03a06d1dd5609565d7ca_*.png
    Saving 348th image cc3c1f9176a4944f285166f0f0e6fec7_*.png
    Saving 349th image 95deda3310af388ce767e1d9aca44238_*.png
    Saving 350th image 3958c907783cb43548e1feddc6efa1ce_*.png
    Saving 351th image 3a6f6dc330d5b353c4418b1e449dfae0_*.png
    Saving 352th image 245e9d480792dccbd301ba367428eca3_*.png
    Saving 353th image 1406e7e71a5a914acd65ad15a08385e3_*.png
    Saving 354th image 979d5167ba19c933639f6f70ea79cd29_*.png
    Saving 355th image 91c3f50a42cc8cb4de7d427df6f8b7b3_*.png
    Saving 356th image de4063b21abf680b6a90f817aebdaa6f_*.png
    Saving 357th image 053124f9d72f9c27d632a9dc7aad8926_*.png
    Saving 358th image e1988433a5ae5dfd1a6b73aafcfb089c_*.png
    Saving 359th image d4dab9e2b1644689dbde0d5107005f5c_*.png
    Saving 360th image 373d19dc832d50a7957b49378675ea63_*.png
    Saving 361th image 9361165d25911ad9b081b4fe75cb7bfa_*.png
    Saving 362th image 357f0ebb37298dcc878f952fc948909d_*.png
    Saving 363th image 0870c93ce2f45fd74acd73b73491b6f8_*.png
    Saving 364th image 7131af250bbd274550a837f1d52331db_*.png
    Saving 365th image d7bdc4fd108bb3c89f06759d42e88d05_*.png
    Saving 366th image e005aa3785230bd141e35d7f842e9302_*.png
    Saving 367th image de191b23554e459ba286541feda0d351_*.png
    Saving 368th image 5ba3f523c9ee4ae4a52c1025bf02b296_*.png
    Saving 369th image 03c496cb5f4cfefad89fae6ce2c58a09_*.png
    Saving 370th image aa93fceecb0ad24b35e20a46c70aed1d_*.png
    Saving 371th image 395045af2844a986e4453564a07b5a03_*.png
    Saving 372th image 11c395588fe7f03162a311fa76e16bbf_*.png
    Saving 373th image deee0e43af55b013ab80acc42177de77_*.png
    Saving 374th image aaea5c1e62e394f17d7386c93fc69d72_*.png
    Saving 375th image 71ededed3ae24705806c7fa288ec56ba_*.png
    Saving 376th image 642f230cfaec3a370acffca095b7187b_*.png
    Saving 377th image a91024cc870008b3e6ba36e57d9d2554_*.png
    Saving 378th image 81e8fced93d017f8fff06503ec2d503c_*.png
    Saving 379th image 89a68d967b2b0a95265745f1a743beb1_*.png
    Saving 380th image 62860daa1cbec6e1ddc1c0b2ee570144_*.png
    Saving 381th image bc5599ad60e8a791514f1ebcd09de606_*.png
    Saving 382th image 7e87fc217ba4c06305e39870e52b27d3_*.png
    Saving 383th image 146bd12d341db70b62649c1f3c80504d_*.png
    Saving 384th image 023388dcb9a2a73fed7dbbe35c241582_*.png
    Saving 385th image ef43440ab267d86f8bac2e7360875822_*.png
    Saving 386th image 94fe83a76c6d5d6630702f42aae43deb_*.png
    Saving 387th image 4f93194c22efa7d77a5d58476ae60784_*.png


    /home/harsh/anaconda3/envs/datascience/lib/python3.6/site-packages/skimage/color/colorconv.py:985: UserWarning: Color data out of range: Z < 0 in 3 pixels
      warn('Color data out of range: Z < 0 in %s pixels' % invalid[0].size)


    Saving 388th image e5db80983e38a5698c22fe25ccb0c4c6_*.png
    Saving 389th image 2803e709de6a971579dd6ef161d788e9_*.png
    Saving 390th image fcd61e108dfb56eda11ba5421c9edc61_*.png
    Saving 391th image 72f652f91e8aa126d86f2288aa287009_*.png
    Saving 392th image b85b4c04603a7c375dabfa56946510c0_*.png
    Saving 393th image f20c2a2e17090c3fdcc0c48f54ab495e_*.png
    Saving 394th image f90894f6b7981cf2111ab46bba5fa452_*.png
    Saving 395th image be557544113e99f39ba7257c75581996_*.png
    Saving 396th image 69ca71cf1bb646db0f990900d3660e83_*.png
    Saving 397th image c9eae66468e9a0c7cbfcc6fa953d1abc_*.png
    Saving 398th image 6318f85a038c009e57a4a94fe0787812_*.png
    Saving 399th image 5eb2bb6ad6016efc235a821e8f3f4ea9_*.png
    Saving 400th image f9fd5be46e3e54becba379ed96a13763_*.png
    Saving 401th image c15229cac61f33886848f0d88ee7a330_*.png
    Saving 402th image 0dedeb889db9a750a692fbe498dab5fa_*.png
    Saving 403th image 73ae211c39a1facc936a37f1a0961770_*.png
    Saving 404th image 380564b2b68d378d60d146f91e9bb1ed_*.png
    Saving 405th image 215b54ec498a3d06f1a9f5cbc3a623bf_*.png
    Saving 406th image 5b07859d66990d6e6446135da585628f_*.png
    Saving 407th image c60e455f3ba3e2db7f99c5983f45460d_*.png
    Saving 408th image 8009736128412e85c4b9d3a706b0f258_*.png
    Saving 409th image e1c368a36a45e38e0bb2d219d9fd9730_*.png
    Saving 410th image 846f7130b4265c869202dea824686103_*.png
    Saving 411th image c41a2c10a15d2ad4eb0d7e27f2f5e5fa_*.png
    Saving 412th image 4e3a4a37de47377ab381c35263f597af_*.png
    Saving 413th image 9827141229544c4da9437589e9bd61d5_*.png
    Saving 414th image 26beec1ddfdec13c7bf121e6a7813d9c_*.png
    Saving 415th image 022d8e2211ae15fd0222e7b747cfe276_*.png
    Saving 416th image 2302fef3edd2a2da03eb8ca9a2d36191_*.png
    Saving 417th image 7e1d2c27a8ae1b91335ef8316a65f3e5_*.png
    Saving 418th image 9a25c778ed4af7dbd1b0c447ec22ed02_*.png
    Saving 419th image e6375ccbad331b3c7d8a2223523f3017_*.png
    Saving 420th image 1a043d9acbed6157d9a56dd30e54dd70_*.png
    Saving 421th image ae99c75c5b512caa128c8a98a9a35ca6_*.png
    Saving 422th image a022fa3bcfa67ddbd30e4953be79a8be_*.png
    Saving 423th image 5eb931e24371048195e3d041cf8b04a6_*.png
    Saving 424th image 4fd287e8f4bd6ce0e6ac7285cd92a84e_*.png
    Saving 425th image 8da9906151050c480a43f8d50f9392c8_*.png
    Saving 426th image 82128ecbab4e7e65752ebf6aac57402c_*.png
    Saving 427th image 19a4353f6abe10bd92f3eb2128eff343_*.png
    Saving 428th image 9f34593705e8348f413f558f5a470b12_*.png
    Saving 429th image a1a014c8fcf2774f60f2c45a6913514e_*.png
    Saving 430th image 3165460f9d2c078d083f2f41b0027150_*.png
    Saving 431th image 5dade10225d87cec86d44c4d9d86ad2e_*.png
    Saving 432th image 7b38ed928e15cd3085bbd873f4b4e007_*.png
    Saving 433th image f59af8cdbfb290989054a91fb99473be_*.png
    Saving 434th image 9ca83137d8c0dc7d61fdbe6ea70bdda2_*.png
    Saving 435th image 3ef5d0623cf9a6079da4cb0353f771ae_*.png
    Saving 436th image 81bf9148927a5aee235887bd57ab44ce_*.png
    Saving 437th image b800701e86a3b1ca1e4685be3a296b55_*.png
    Saving 438th image f4876ece6da3896edc422ffb06de2996_*.png
    Saving 439th image 7596de66da5186a3bbb481cb0110fc1f_*.png
    Saving 440th image 18248d6fe5d3c88d2ed3a1c4801446d4_*.png
    Saving 441th image 3e55e88340dea0435794be2d609ff088_*.png
    Saving 442th image 0dafd780f1791352cf0545c1a6d6f41a_*.png
    Saving 443th image 60da8a9f12b0b4ed0ed625f15bff93a9_*.png
    Saving 444th image d39f9a48f540546b938ae6a3db6e0f8f_*.png
    Saving 445th image 005fd68c3d6ff608606663522979a6fe_*.png
    Saving 446th image e5a645b164e69bb49b6615bf40d151fe_*.png
    Saving 447th image a1309552ed93022998e45598fb89c6af_*.png
    Saving 448th image 1cb3b904616c72c5a980d40a0d4a2ece_*.png
    Saving 449th image dce070cd5de9ae6a7305762246e62129_*.png
    Saving 450th image e32e100ea3944ab0901eed77ad0c3082_*.png
    Saving 451th image 804a7c305930d10872bce6758e5cb277_*.png
    Saving 452th image 2d54a5262ef8d01cd1c30f7a99564f95_*.png
    Saving 453th image aae1ba4f4ec9631763c7d7338c53e282_*.png
    Saving 454th image d69ceeef0da583ad55ea0a5b6a67df2e_*.png
    Saving 455th image c568715fbd140e5eb27ab220e0ba9105_*.png
    Saving 456th image 24b3de56d53ca4f840d967cc0094df9f_*.png
    Saving 457th image c3143e40a69d3b8896d1b307514f44fc_*.png
    Saving 458th image 1817c56a47334179b24d978edee64633_*.png
    Saving 459th image 908667cf1a01e8664ca331cf8207a0c8_*.png
    Saving 460th image 2d87b3a564d465658943ef57cb17215a_*.png
    Saving 461th image 396681f9696b247e23b33725051cf898_*.png
    Saving 462th image 8afd146f5e394e49959b6b586a3cc616_*.png
    Saving 463th image ac3fec5a7e348814e8f992f4d6873ac2_*.png
    Saving 464th image 9ed8dbf7b9323de347476a75db02aab6_*.png
    Saving 465th image f9b626d50d49d52ca0d5e1456714c6a0_*.png
    Saving 466th image f97248a174c106ebe742b97ae4d9b98f_*.png
    Saving 467th image d34c3b92f42e629d0c6a96da07d7f92b_*.png
    Saving 468th image a99845c7e179566d375a1e410cd4045b_*.png
    Saving 469th image 9434b203525b61312a72a976bb430521_*.png
    Saving 470th image d2d0323e89a83847abba6ac0eee9386a_*.png
    Saving 471th image 82b7c862876228386df9b6541ecd859c_*.png
    Saving 472th image 3cd3a4d3029e67bbfa0a26fb3d9f38d0_*.png
    Saving 473th image 1c3599bde7863604f0c451657461cfdc_*.png
    Saving 474th image 1be7fe1d7686fa20f8cc209356c648be_*.png
    Saving 475th image 43ee31da4ad91d2fba4babf21dc0cf65_*.png
    Saving 476th image 6b39f5a1775b386fd8591e95314cede0_*.png


    /home/harsh/anaconda3/envs/datascience/lib/python3.6/site-packages/skimage/color/colorconv.py:985: UserWarning: Color data out of range: Z < 0 in 7 pixels
      warn('Color data out of range: Z < 0 in %s pixels' % invalid[0].size)


    Saving 477th image baff51ee2145831032e982de50ae94a9_*.png
    Saving 478th image 4120a07a189dbcfac91d94090ff45e1f_*.png
    Saving 479th image f071af153834e5caaa8e7ce5e1bfcfe0_*.png
    Saving 480th image 41d15f2d6fae96fbc31dfe9dc708e98a_*.png
    Saving 481th image fef5736d61dc11c887369b251af059d1_*.png
    Saving 482th image 8894a8f4f294110c4503df7d34961c25_*.png
    Saving 483th image 7ad4022cc3652a37e747a79c088c5a47_*.png
    Saving 484th image d3ddb4bd0c5b43840d5825d192be8d32_*.png
    Saving 485th image 7052a94175780bfac4b65b849603ef47_*.png
    Saving 486th image 87ccebba449d343a05ab08e58f820b53_*.png
    Saving 487th image d9df8b63e5a637d8d6ec22140433b9ca_*.png
    Saving 488th image 5d2b2b2fe5c5b38c577208a09b8fd967_*.png
    Saving 489th image 62068473cd8fe1d68a649b6985202414_*.png
    Saving 490th image 1a010bdaf3df43b1918395dc6e34eec1_*.png
    Saving 491th image 9095612076b05f58ee3731bdef9573c4_*.png
    Saving 492th image 4e4381f593cb8799bd1c45d9f98bc231_*.png
    Saving 493th image 6e9b1dd03243a74ce72cd9bdaa74d575_*.png
    Saving 494th image 4d801c61d44b2a3ccafa9d6e1afb596c_*.png
    Saving 495th image be678903bc3b11297738aa7293ebe85a_*.png
    Saving 496th image 6263954624ae11972fde161bdb0fba16_*.png
    Saving 497th image d4059e2f5d08d1182cdffb712b0b6ebb_*.png
    Saving 498th image 26fabac6c8546fc4a2f29e56fd625e46_*.png
    Saving 499th image a330154da6e38a7726e4d30f8c47efd9_*.png
    Saving 500th image 59b3431b223ff5fd7ed25fdc4817d640_*.png
    Saving 501th image 85752cfeb7ba7d966ca0523574742377_*.png
    Saving 502th image c4745b060dcde5e89b21f34344ba4911_*.png
    Saving 503th image b8e9fe2c4ce525f253c7105fc8a6b67d_*.png
    Saving 504th image df6bb16bfb5f3caab0a1f2337b6c0e2e_*.png
    Saving 505th image 39115356bac4feb6c1a05d8b79caea05_*.png
    Saving 506th image b7ffc7465f851860ba09be54a00ca6a4_*.png
    Saving 507th image 35bf516d670ae991c9a3976292b6a75e_*.png
    Saving 508th image 4025c79419667d6b2cc376e8b5f79e85_*.png
    Saving 509th image b3fd64ea38cf225121e1126de59e45ec_*.png
    Saving 510th image 352390509b2f12d928001e4191bc8355_*.png
    Saving 511th image 2c29584992db7c2167c450c36362d4dc_*.png
    Saving 512th image 27462576ac6f043eb1bbe863a4e2de90_*.png
    Saving 513th image f084abe4ba693e4ac601a13085e34f11_*.png
    Saving 514th image d3b4ed121b46f9fefa6be53a4f57f51e_*.png
    Saving 515th image a7f90e6ee6dec390f7bd0fc399e15a58_*.png
    Saving 516th image 87610119473657b8b8eaefd5f0d4f556_*.png
    Saving 517th image 2a3b6fa6083d55f38e2ea88867e44c4d_*.png
    Saving 518th image a044ecc471d9896d64e80eae9f57e1ec_*.png
    Saving 519th image 643bdd9fc9fb08e90fc6b5e3bf7a1578_*.png
    Saving 520th image a47f8b8aab2aa005d5c0bd622c7cd697_*.png
    Saving 521th image 851ea40eb33d95a072885f162cd386ce_*.png
    Saving 522th image b82957182354c454b8fe0f5018ed8804_*.png
    Saving 523th image e3a25027ef019d4886344f80e7c51c57_*.png
    Saving 524th image 1e06708c39216fbbee960b7d81c6cbe1_*.png
    Saving 525th image e2dc0b832ecef77265e58ab2a2c80798_*.png
    Saving 526th image 2a2ac765e295b61f2c446a5720f94d59_*.png
    Saving 527th image 63c226dbf8b1ed37e55e2c46cbfbef0e_*.png
    Saving 528th image 250c6c6787a96c27b4324b9df123216a_*.png
    Saving 529th image 233993a9862eec3f91a1eea0b7a3bd58_*.png
    Saving 530th image 38b607e927c8665a17e39457d13ffda3_*.png
    Saving 531th image 4da1652c45f2a1421cfb5e4a9fb2bf29_*.png
    Saving 532th image 7c539b874e6666e4522d8b95fecf12e5_*.png
    Saving 533th image 30c01946b09e75502c8900ea2d2a3b69_*.png
    Saving 534th image f5cda736914ecb390b144b314eb9da26_*.png
    Saving 535th image 8f6359524c4502d403ad3c0e575c75b9_*.png
    Saving 536th image 5038d1f75c1b208ece589f5cc874ccf9_*.png
    Saving 537th image 376f64ca3250be912934b808d746dd25_*.png
    Saving 538th image 7606a1435f53a05659139a1239318e19_*.png
    Saving 539th image 790e92130cb3ac5fc51c831b9f476f61_*.png
    Saving 540th image 4a65a6ccfc6372224b3d73f57b484929_*.png
    Saving 541th image 4148c9345cd9a39e9f1e494c18dcb15e_*.png
    Saving 542th image 83cc13c12cf41cfb99da382d180daa46_*.png
    Saving 543th image fda4bacdd097789cb5913952ad3c6422_*.png
    Saving 544th image a36693ea98877ddf5394c4703b3c43a0_*.png
    Saving 545th image 5de29bd13976ad969fe54700c4004c1b_*.png
    Saving 546th image c5236d89009c167950c41ff0fa8b81f4_*.png
    Saving 547th image e72a17aaa434f110abac4932040ccf4d_*.png
    Saving 548th image 9024eddc447e55b585bc4086d3b2bf0b_*.png
    Saving 549th image 01b0a00adecdcf5b891539381aec35e4_*.png
    Saving 550th image a1729f6cb3c4965470129887039c02b3_*.png
    Saving 551th image 1043b57f03c45b6aa9e00d0fb3dcf724_*.png
    Saving 552th image ebfd53279639f582c9a1878b32fde7ce_*.png
    Saving 553th image ef5318d03e55cf15abea66ca0df7951b_*.png
    Saving 554th image f28a763e5d9a9fc741511765a40b900c_*.png
    Saving 555th image f956450f65cf350d7a690fbd77f03177_*.png
    Saving 556th image 71857bbceafe63c556b9c3f81b2f9986_*.png
    Saving 557th image fddcbe7d78f02af28794f4089f334bae_*.png
    Saving 558th image 5301cc39f9123c811229458f0b342363_*.png
    Saving 559th image f8394d8fe646f6a3bdc1a244af90a659_*.png
    Saving 560th image b10cb50a82b13d5574e7b00cbba80377_*.png
    Saving 561th image 5f20bdcd7e00d3afdd791de17107829c_*.png
    Saving 562th image 7bc6b27c395ca40c522a17ae5edce847_*.png
    Saving 563th image b59d0820889860e75293af27455a56e4_*.png
    Saving 564th image a0e7ee88fd5e19723dc34cb468a7d4ae_*.png
    Saving 565th image 5ae55fdb59c2a922234ed8916096e5af_*.png


    /home/harsh/anaconda3/envs/datascience/lib/python3.6/site-packages/skimage/color/colorconv.py:985: UserWarning: Color data out of range: Z < 0 in 2 pixels
      warn('Color data out of range: Z < 0 in %s pixels' % invalid[0].size)


    Saving 566th image 8d6182f273ee65a3a0b9f74ed909c90b_*.png
    Saving 567th image d380e6619c44399dd3316e2cc73c811c_*.png
    Saving 568th image c2b0603ae5b1d87f1b21c5309884b129_*.png
    Saving 569th image 26c00815032b2ad0a85101f2fb84da3e_*.png
    Saving 570th image d696ac06eced29f6a4848403317fea04_*.png
    Saving 571th image 43b523d7281e5cc91144f940ca253a9d_*.png
    Saving 572th image f6a9787fa0ca2ad97d48676efe96be37_*.png
    Saving 573th image e50d6a4e601a1f7db39c620bb2486cb4_*.png
    Saving 574th image 0b83ce5513f2bab5215eac63fff8bcde_*.png
    Saving 575th image f4cdf6718a94a2f3706bcf931c941aed_*.png
    Saving 576th image 7afca896a1f045c958fb55f7e642673c_*.png
    Saving 577th image 6bde271bd09c790922dc7a863bcbc1d0_*.png
    Saving 578th image 865ac868c617ff58badf005a64d0b4c6_*.png
    Saving 579th image 9dcc086b06df7f50c665a8f7169b745a_*.png
    Saving 580th image 99e27d5004133c0687fa69f073152d0f_*.png
    Saving 581th image 7b4372c520039c212c7df3798fe747b2_*.png
    Saving 582th image 374625f7e58e2d327e87da96fe4befef_*.png
    Saving 583th image 2325e0a59cb0fb502fdba8e38dac4e5e_*.png
    Saving 584th image 49a61037178e97decc09dcf84a1ad766_*.png
    Saving 585th image f78e0ba755df0897123d1d790dae4247_*.png
    Saving 586th image 2c8da3069ff33620c7e11aa243481b7a_*.png
    Saving 587th image b7196cc44d6657f4950486a4515b0ecf_*.png
    Saving 588th image 5831ffd056d5fb8149fc63501ecd2b6b_*.png
    Saving 589th image 90557da1c32b40dee564d34368bb1eee_*.png
    Saving 590th image 6bec9fbf34200a54584ccd3bf753f191_*.png
    Saving 591th image 4b2aabc4700ec56846655200867e9f9f_*.png
    Saving 592th image 9706b8d3a310c82d5a0396f21c216dfb_*.png
    Saving 593th image 23da9ea6ad49525da77801eb2767cea1_*.png
    Saving 594th image 021b24a5c749ceec598132055d435903_*.png
    Saving 595th image 6867ff748c5b13e82fb0bfd2c798e675_*.png
    Saving 596th image 3cc8e5ca5ddc9f60133492d1b22bb999_*.png
    Saving 597th image 2f0a29631c906387adbf97b5306658e0_*.png
    Saving 598th image 810dedf57512909a484b9c0ba34dd485_*.png
    Saving 599th image 98d16693c5b01543cd5e95e5df3cfa73_*.png
    Saving 600th image 4235eb0dbffa514fa633cb1d760df51d_*.png
    Saving 601th image 86305f2c7de4bd6a998a0a9da53d4894_*.png
    Saving 602th image b56a4cc6f3266568524e953b5e3c361b_*.png
    Saving 603th image 2858aeac06b7335e1df1c213392bb1a9_*.png
    Saving 604th image 3479394e0619c30fc8777114c2306e01_*.png
    Saving 605th image e8cb700697a45d561ee1e71264074d7e_*.png
    Saving 606th image a7eac9d58918c385183db26360775d1f_*.png
    Saving 607th image 363938797af9ba623e32ba697c967ef1_*.png
    Saving 608th image ed62187051b3f92c6234a7b6dbe60091_*.png
    Saving 609th image 030ed7aae04939555bf54d7924fd615d_*.png
    Saving 610th image 0a43b00c68359a7c4d11ce39f057f1d0_*.png
    Saving 611th image 9eacde0b8de3ecc879e3befefaa5e088_*.png
    Saving 612th image ee6f95dfada9d0fa66462e0c54e6322f_*.png
    Saving 613th image 2e9edf4905f5398854997bb6df0e9ad2_*.png
    Saving 614th image 78e4ed140ae2e4f2ab4d5c7a8f7ec181_*.png
    Saving 615th image d7a275e0f63dcd5e083783e93547c21d_*.png
    Saving 616th image 1691d1907df456d8b3242c688f2a612e_*.png
    Saving 617th image 188db6d95501ff6fb03390bfee081853_*.png
    Saving 618th image a4dcfbfa6bf2fa064109f4e8a4e95160_*.png
    Saving 619th image 2b0a441f5c4b53ac891d8a2281b55a0b_*.png
    Saving 620th image eedac1eaeaa15d315e2771bc3c371377_*.png
    Saving 621th image 697bdccb51df35a12084917a54e7f9cf_*.png
    Saving 622th image 78f94f980b8a8d16e5ad1df730377782_*.png
    Saving 623th image 13c94dba758980d96b5ce7e96d3b9674_*.png
    Saving 624th image 5f26360caaeda8af507ca4b2864457dc_*.png
    Saving 625th image 07f593d5c863e4c6fd6936ac4edcc266_*.png
    Saving 626th image d4b35d555e72513b434a7d2d061fec0f_*.png
    Saving 627th image 285b2546564b5f5f96d55599343d7afc_*.png
    Saving 628th image a75f9aa5d5555ea29b26a54eabfecd3e_*.png
    Saving 629th image df9d8757bbbc16eab36c266db289b839_*.png
    Saving 630th image 250787d6e92897d6e3f33615dca9ea2c_*.png
    Saving 631th image 4948e312caf886b85acb82eaf7d357fc_*.png
    Saving 632th image 306e39014cac3e43c0446025ce63dae3_*.png
    Saving 633th image 247a9b9ab80430fa63cae9dc2cff8bac_*.png
    Saving 634th image f1cf9e009f203b4d9fd9c49f32a2a43a_*.png
    Saving 635th image 0e4180ebe670d7b15a6859dec6b99ad9_*.png
    Saving 636th image 100118efe6ff0dc8c1e04d3fbac3a3bb_*.png
    Saving 637th image 3a8a0d82076ae623b889b9e8ec2ca148_*.png
    Saving 638th image 901a6e4be883b8b8aa4ad40b8bf91180_*.png
    Saving 639th image 6201afbd4975d3227c6c5cc1e305e46d_*.png
    Saving 640th image 57fe00c9cc4a9f98c516b7593ac28fa3_*.png
    Saving 641th image ad81ce506df540d4a474d13c8d618602_*.png
    Saving 642th image 694550c022fa5fb4c40d0ab10c6fd3c0_*.png
    Saving 643th image 4b5558225322f28fb0cf62c1d519e3d1_*.png
    Saving 644th image 49b7fae76809eb144b3ce3ad29f92be6_*.png
    Saving 645th image af667ba439c4766175474062eb8b8d99_*.png
    Saving 646th image 9b4d5493c4de9d5307a5fd8a529ae2dc_*.png
    Saving 647th image b27b6e10cccfd0619d23ae080e2507f3_*.png
    Saving 648th image 31bfd040e5319483b3bae4352418a9d0_*.png
    Saving 649th image 35bb3f20cbf0396a4dbd6889a4f1bbfd_*.png
    Saving 650th image d3fb6315d4e56a0d3ed1276edd973d2d_*.png
    Saving 651th image 80ec2520bc2cb3dadc306d6d08f18778_*.png
    Saving 652th image 9027322e0ce70fafb4be0221ce8c7c20_*.png
    Saving 653th image 408f39941543982dda5b7ff2716576b9_*.png
    Saving 654th image 55aa6c27bf4c9f3fe50b33abb22f665a_*.png
    Saving 655th image dd3b72fdc25c564a88c9c9a54dce9e01_*.png
    Saving 656th image 1208ac9867ea3d561c4531a9469886d0_*.png
    Saving 657th image d457f9b81a1e29898e1afd54b3d807b9_*.png
    Saving 658th image d33a51d385ce8c6f73df1232715a1d38_*.png
    Saving 659th image 8e0de058d79ac0577b0bf01d28e1355b_*.png
    Saving 660th image 304c7cc50877ead1760011070af11117_*.png
    Saving 661th image c98f3cf81d89d6f80827827ad3624d02_*.png
    Saving 662th image e5b87f0da43a70d57a63afa297bb7b5f_*.png
    Saving 663th image 90077129ecc521cdadca09358e681967_*.png
    Saving 664th image 56be53ef20c83925fee7800fdea5fb8e_*.png
    Saving 665th image 0e0275392a7c124b8c6b0ee0d7f11436_*.png
    Saving 666th image d9a714f105aa6b1b733437979a09435b_*.png
    Saving 667th image 0e1b7cd946a3ec2968453fc425d74a93_*.png
    Saving 668th image 3cd3446f09518561eba4b94fbed2b5b5_*.png
    Saving 669th image 0ad997338c7eb1b84e13db1b890a75fa_*.png
    Saving 670th image 276c9cb284395a9d6c0e77fd9c5af3c1_*.png
    Saving 671th image 88849bc88dd7e85862943526e34a44ce_*.png
    Saving 672th image 40a4dfedb9aaf49d751c5aa20f989db0_*.png
    Saving 673th image 3854c78a23d06352b39253c33c15f811_*.png
    Saving 674th image 9a43aa783d4ca633add65446e1cabf10_*.png
    Saving 675th image 0b1ae4e0fc5180b2c145570d7a430e3a_*.png
    Saving 676th image ccb6f2ec41bcfbc092f0e26ee81ed1d3_*.png
    Saving 677th image 2f6e666129523a38726ba8bfc70d1724_*.png


    /home/harsh/anaconda3/envs/datascience/lib/python3.6/site-packages/skimage/color/colorconv.py:985: UserWarning: Color data out of range: Z < 0 in 18 pixels
      warn('Color data out of range: Z < 0 in %s pixels' % invalid[0].size)


    Saving 678th image b3f87bf1d292a216fd860acc3dc0082a_*.png
    Saving 679th image ce372d035fcf96733451483b5920881c_*.png
    Saving 680th image c67ff467c3cc13ddc647328f6cc274aa_*.png
    Saving 681th image e820635d206ac5eb7451f0ff279f06b7_*.png
    Saving 682th image bc8191df2d91eebc199804830e8c08b6_*.png
    Saving 683th image 6ff6a9a1f18c7b28eb983a6a73b1dfe4_*.png
    Saving 684th image 953b8cd016fb747aac6b2ff325016249_*.png
    Saving 685th image 00237e81f7b181f0305738407cc88d51_*.png
    Saving 686th image 2d6abbf94caa990ad6bd1272de949feb_*.png
    Saving 687th image 821ddea880c4e2f5763077f2dd3aacf9_*.png
    Saving 688th image 25a6cbcf65be2e499985a1f50558fc7d_*.png
    Saving 689th image 237b1e33b02d9e81ef105e1f5dbd5aee_*.png
    Saving 690th image dd85c181a992cd79d56e902d8f8a341c_*.png
    Saving 691th image 28b646b3fab9892a73f470ebb9e7ec1e_*.png
    Saving 692th image 3edc93e79e780caf63d6a846bd9e7135_*.png
    Saving 693th image c150a24006532df1012fcb13ff00ba11_*.png
    Saving 694th image 8438562398e4c70b0c5152578b54116a_*.png
    Saving 695th image ec886e51b0f4585a43505f237a3d5eed_*.png
    Saving 696th image d9f927f4a4771d0acbedd511eda7a75c_*.png
    Saving 697th image e7356083500821ebc841ea4b1ce16e64_*.png
    Saving 698th image 1dfa4461e8437d9b9c3c21c669fc421b_*.png
    Saving 699th image c61bf69e5ca4671a7a65bfbc341c3000_*.png
    Saving 700th image f83894597b4b5e369032bd8f368b8474_*.png
    Saving 701th image d8ad538ed57cb9ed535f261e535727a5_*.png
    Saving 702th image fbebfcf07bba98f9ec57dcc227779ab8_*.png
    Saving 703th image cd0d66be4ba2df5c7cadc1d080dc5d74_*.png
    Saving 704th image 48b5f2bf3bf452ce09511a156be57e34_*.png
    Saving 705th image aca67d189f3a9dc7ddff6bb1e8878c0d_*.png
    Saving 706th image f530865cdf7d5d829a7ba38e93e36132_*.png
    Saving 707th image 0098c5cfd8e94243d260dabb549723e3_*.png
    Saving 708th image 47215f1893b663fa55136ef6c22429f5_*.png
    Saving 709th image 8b1dbb9986b876039a2860b098ab88b1_*.png
    Saving 710th image b8a02aca226ca41aed399f8d46816d02_*.png
    Saving 711th image 70caa61e93f706b7c5a3f24352092865_*.png
    Saving 712th image cd773b7bc2d023a67e7737383974d751_*.png
    Saving 713th image 46f0ff14978d5f8ea128f1953cfc3798_*.png
    Saving 714th image bf3cfb1001b904d4f04ad23567c4874b_*.png
    Saving 715th image 13b8c8f68ddce3f4d13a01a32ba6734e_*.png
    Saving 716th image 73cc9568a5c5a0bdeebc486abdb51297_*.png
    Saving 717th image d06066b73d19caa3d0a22c77bf0a8635_*.png
    Saving 718th image 9d24bc05b118011113b8e263e00d35aa_*.png
    Saving 719th image f34f45ea98bb4ab8667917e1da0775d0_*.png
    Saving 720th image 88ad46b084a6b3fc7362cc77011c5ce0_*.png
    Saving 721th image de0330049eb55f19d49beff45bea13f3_*.png
    Saving 722th image a8755264b5b288d94cc1cd26bbf9d3f0_*.png
    Saving 723th image f373f8c8212ccd44c2813eb10b6fd5e7_*.png
    Saving 724th image ae6d6f1085b6afbe3d7bb9c52f1f3df4_*.png
    Saving 725th image 83028ce21d9f8e8a11e1cfc8740f3239_*.png
    Saving 726th image 05f07aab01429dfc27129ae17884cc9c_*.png
    Saving 727th image b228e6f1900d972810b62c4bdec9bd21_*.png
    Saving 728th image 8c1f01cf4edf1cbc51977864163fe313_*.png
    Saving 729th image e66f6fb97b0afe655ab58774f2c6b430_*.png
    Saving 730th image 26b886d3345498c82e1220c0959009b1_*.png
    Saving 731th image 58b173a3f89c266b9a4d75152d8cd9cc_*.png
    Saving 732th image b7dd305120f6398857b2632818c02748_*.png
    Saving 733th image 4c80c161a49be4b1af9c3e5151f215b6_*.png
    Saving 734th image 6e4b182a07098ac53d53eaf8c039c0cc_*.png
    Saving 735th image d8ed1801341dd52e38b64b16bc17874c_*.png
    Saving 736th image 0efce2ca70242d4dcb6fa98217bd749b_*.png
    Saving 737th image cd276343545396ec19c0bd847b860995_*.png
    Saving 738th image ce1eff88e5b0b4f68b1aec88de04ccf2_*.png
    Saving 739th image f3e2d17e10aa55dbbb3a8767be5f57a2_*.png
    Saving 740th image 4cce3147e0540585530c1835c46c5f02_*.png
    Saving 741th image a48709536d0e7e7d301840cb849d3a6b_*.png
    Saving 742th image 37c8f87f2b7e207d68b90382d0ae137f_*.png
    Saving 743th image ef3cd084de68a3f4f7c61c3dab7c4a3e_*.png
    Saving 744th image 59e911c1929c133d72a11ee0b1269dab_*.png
    Saving 745th image 3b353541bd3efd3bc9a501a98a1b3197_*.png
    Saving 746th image 8839e3563413f2c0f890eb72db48f821_*.png
    Saving 747th image f84f54e9e55bda2571b4677054b9663c_*.png
    Saving 748th image b9eb928cb8e54cf7ac84ee31580cd293_*.png
    Saving 749th image dc7c0a32af43861abcf1ae4ec8d34e9d_*.png
    Saving 750th image f4334896cbcd37f2e7f755231d79b636_*.png
    Saving 751th image d2aa138321cc01daf22014df3557e2e9_*.png
    Saving 752th image 378183764275d5616d35313dfd7dc30e_*.png
    Saving 753th image ca387f57fef3c0402905cd3424d7a3fc_*.png
    Saving 754th image d504ede7c4c231913b021af337f50389_*.png
    Saving 755th image c07e3b1c959ba98b98b12d105a5076bf_*.png
    Saving 756th image 9f10ac583d2f04e649e10f4d95c88e56_*.png
    Saving 757th image 1c15f71560c1c464ce11b3c8bf5c88c4_*.png
    Saving 758th image 524250e1abdaa82151a04dac5b629709_*.png
    Saving 759th image 6c2896210b93af2a44354b3062ee0b02_*.png
    Saving 760th image 0136cad23636739a794bd7927cc57c78_*.png
    Saving 761th image f72255bfade29bcd0eda6aa002a21d6a_*.png
    Saving 762th image 8b982a438ebddeabda0c9ee00fcda914_*.png
    Saving 763th image 2e7ba39eba519809188b96e835f79e77_*.png
    Saving 764th image f0387031da7462480436db21af7ee058_*.png
    Saving 765th image d6bd8b0fa9e2bb74527eabd78e2dd2d4_*.png
    Saving 766th image ca5c54046722831babee52912107d2df_*.png
    Saving 767th image 6d4a916bbd6511b3c2ab2d7ee60afd47_*.png
    Saving 768th image dae0ee2015f18c480eba3f766e59199c_*.png
    Saving 769th image 77c259bb2e2f81394ca7621dda374907_*.png
    Saving 770th image 57377a3092083f40d132444b4ab2dfb1_*.png
    Saving 771th image e2dab8e844423746753c1939cd070a8a_*.png
    Saving 772th image 15e87cd71bc30514047066817a6e948c_*.png
    Saving 773th image 53ccad37d31a21a662629f855042f67e_*.png
    Saving 774th image 7583534adfc997c2c4d758e81ea50230_*.png
    Saving 775th image 9346cd5b556bab63cc704023873f6fcb_*.png
    Saving 776th image 75e288bb3c227db8ebed24e823891f9b_*.png
    Saving 777th image 590270f187248a39b754b0a1030caa68_*.png
    Saving 778th image 51410bc28e410f715e7eb8be9f5129e2_*.png
    Saving 779th image 1e2677a7c663101eba7ba603d60b3188_*.png
    Saving 780th image 422a18316ad4eef55548eec10c5add3e_*.png
    Saving 781th image 548d448d2a6babe2f79e56ad4bf42d80_*.png
    Saving 782th image f7e70d6f346ac9afd728e2f1b309cdb5_*.png
    Saving 783th image e4762b3db3f67c5427631fe8cb085e02_*.png
    Saving 784th image 7984d0559439be03cc1073367ce13e47_*.png
    Saving 785th image fddcdcd73fcfcec35a37e35260b94f64_*.png
    Saving 786th image 1943a93b5d26f6432dd21c233c80587a_*.png
    Saving 787th image abd2932dc8ff9fe8736118b012df3795_*.png
    Saving 788th image 9cb62476811827d469c205cff7b83c3c_*.png
    Saving 789th image 898009d96c072e78e3235b4a608750c4_*.png
    Saving 790th image 250d138dfe370c0ef0503efa63016de3_*.png
    Saving 791th image d12fe7eef14f1c076e28b1274091d515_*.png
    Saving 792th image a795abff11ac2c312bbcb62005703a6b_*.png
    Saving 793th image c28820f40dc1ce91d5a8fff62f57cbee_*.png
    Saving 794th image 307a59b06b33eb70e517937e11421692_*.png
    Saving 795th image 471cedc5b9bc47a0071198b1010d994a_*.png
    Saving 796th image d7b5537785d4a5ea6f8265367d330d21_*.png
    Saving 797th image 4d988c0ff9d596a46ec724ab77b648aa_*.png
    Saving 798th image 5808a39cca9f669cf451d26d7e1a40c2_*.png
    Saving 799th image f53f6b4c4d522c2af3702a1aecdf1167_*.png
    Saving 800th image cfe9a5aed3cea6d07e2c9cc06e6c9e4a_*.png
    Saving 801th image 30316224c21ec4ce2d1101a2fdcf3d6b_*.png
    Saving 802th image f8f5437ecbaab1a46baa502400ba09cb_*.png
    Saving 803th image 279a2791a316ca93dedcd025c3f6244d_*.png
    Saving 804th image f193345afd3747e5d3dc5e149dba4f6b_*.png
    Saving 805th image 6c20f540aac5ca4cedd5530f394ba8c3_*.png
    Saving 806th image bf791c3aa2d65f64f6d19f2f165a8657_*.png
    Saving 807th image 539684ebb0debe30cfa78b86efcd60b9_*.png
    Saving 808th image 2fa1365636b408af0bf3c01674b5c1f0_*.png
    Saving 809th image d4f57d964f69cb4165bf49b3f64f7e5f_*.png
    Saving 810th image 5ccc659c0bdb23a31b219bf90106836e_*.png
    Saving 811th image db356bc61cf4a1aee3f74becdb1bdb91_*.png
    Saving 812th image 7e94a616009ad09ce699c636355fad94_*.png
    Saving 813th image 6aaf46339ab6e47ccc503dced1b2ce1f_*.png
    Saving 814th image 32ee567ab0529bf9753a0c6809a32810_*.png
    Saving 815th image 7f2edd47a3e1ff56f3343f1f04109a3a_*.png
    Saving 816th image 19aff698607435d8f53313beea1820da_*.png
    Saving 817th image 0d397143c5ff32f61bc040661ba220b4_*.png
    Saving 818th image 9cc21619ddf45d97fe72da2094164d02_*.png
    Saving 819th image d00173e8a54f7aeeecd2fc289e844759_*.png
    Saving 820th image e2ffef052864c442d671188a8d4a9de7_*.png
    Saving 821th image ba6550d9314ac689043abb22d6ca76c9_*.png
    Saving 822th image c563e1ef3dcee28c4a9a4b14990d0585_*.png
    Saving 823th image db5064d5ac0fd3dbef4424db7183daca_*.png
    Saving 824th image bb450e99f396ef8af0b17e2f41407957_*.png
    Saving 825th image 22ab4136aded96f2a641dd320dee6a11_*.png
    Saving 826th image 2e52eb9e87fc2d1d0534bf157acbd2fb_*.png
    Saving 827th image a7d3fc0bb0d1f60c95f1557be1865e46_*.png
    Saving 828th image d1169c77d12fbd0dfab6f691ed7458ed_*.png
    Saving 829th image 41d83f8681de854f49ed5b6147674835_*.png
    Saving 830th image 4c3e881946d311caaa7b4f98ffd9fd7b_*.png
    Saving 831th image 059e60b40b67d38bc0469cedc493edfd_*.png
    Saving 832th image 747ab70d0a322dc6706f0d339fff221e_*.png
    Saving 833th image a37b8bd0f0863d78d9a148c484175b4a_*.png
    Saving 834th image 48170e6ef3e5fe840cd3de204727164e_*.png
    Saving 835th image 3e5b28cba766abc0160940263e351ed6_*.png
    Saving 836th image f1fe899d3346470246370a53cffc4f6a_*.png
    Saving 837th image a704bb16466e4510b14429659fca4e71_*.png
    Saving 838th image 758973024a112c6882b9949f74e88e3f_*.png
    Saving 839th image ee150c823a2136f78d498680e1ad9410_*.png
    Saving 840th image d5ad1dc7695ebabe67d6654c4f5cd0b9_*.png
    Saving 841th image 26026dbc99b0f7d1670314c27450bbf9_*.png
    Saving 842th image a8c4a71e319af73e50351801d1630653_*.png
    Saving 843th image 06477292992f511d5b597503637c9996_*.png
    Saving 844th image 7275f772ac31b07b8d780bb653a0e131_*.png
    Saving 845th image a500cbc74b5add6b4e7cc19117382574_*.png
    Saving 846th image 3747dded1c27c8faa88d487ea2d14279_*.png
    Saving 847th image 990be40edc57fcb71335bb4b81b6860d_*.png
    Saving 848th image 6fad840ea23b5267a89bd23131c538d9_*.png
    Saving 849th image af99fbddf97595ff8f5e4150b440bc34_*.png
    Saving 850th image 411a1e5e11e547788807f3b3c402d433_*.png
    Saving 851th image 23751ca3624ac051504ba0776cd078e6_*.png
    Saving 852th image 54cc3570721a799270e9b74653a7e2fa_*.png
    Saving 853th image 75cee8f2d6b0f75ba4bbd7fa2f35712c_*.png
    Saving 854th image 06e1945c5b0e8523b39a3aa6ea228609_*.png
    Saving 855th image a3d8be66b8bd2c550915ec0bc60f546e_*.png
    Saving 856th image 2d2b0bba1424dd40df2f30243ae7e2e1_*.png
    Saving 857th image 23a4076404dc65c3ff35a69181940f04_*.png
    Saving 858th image f52c28c775f27d9c9fc84795becb3d88_*.png
    Saving 859th image 7664a1480c61081a45b4ea7d4ec86111_*.png
    Saving 860th image 7220cb98235e1a1cf39246256bb95647_*.png
    Saving 861th image 6a45c214aa86de7d51570bb598480e85_*.png
    Saving 862th image a290e7bb5866a69463b96535f4cd9fec_*.png
    Saving 863th image 6df83a9429c7446e3a49114c123c0335_*.png
    Saving 864th image 39e3035acdf75878515f88cdadd4354a_*.png
    Saving 865th image 985dde33ff11ae3f4752f9e204838bea_*.png
    Saving 866th image 7e569707a20bd5f11f8658ee71c7073f_*.png
    Saving 867th image f665d686dcc4ebb7a1d99b1325e429d5_*.png
    Saving 868th image c61d6b19b70f4cadc10336fea77b6c72_*.png
    Saving 869th image 6d245a88ffa93cfef28d2baf0bdf7e9a_*.png
    Saving 870th image cec6f57f351ff398f1bc3c0522ea62b7_*.png
    Saving 871th image 27875ec88767bbfb57423e6dd19739c0_*.png
    Saving 872th image f013859ddfc4c9efc6eb5a08ffff7fb1_*.png
    Saving 873th image fd746877c2950d4ba05e38aeb7d340e9_*.png
    Saving 874th image 568e5c1fb55ecba3d7205eb92d529438_*.png
    Saving 875th image 787e0886a56757e00924c5531b16942d_*.png
    Saving 876th image 96bcfb221be3443b1aae54f513e3fb5c_*.png
    Saving 877th image d25692d31d35c6fdd5e5697ab32e3398_*.png
    Saving 878th image d90b3a98884d374d5b1035d83de5c53c_*.png
    Saving 879th image df630438bfae7fd80d4fae6015adf654_*.png
    Saving 880th image fe4d6914fc9b6460326e2083913b64f6_*.png
    Saving 881th image 2bd2874953130dc606080b14b0faa533_*.png
    Saving 882th image b25e1ef2ab40e8f3d2c061ebb8596458_*.png
    Saving 883th image 3fe0aebc0eaff282613236efab9e8cbb_*.png
    Saving 884th image e6cf46e951d8ee51e99c152e805580da_*.png
    Saving 885th image c0989d2d0d2232143eaa2203c2cc0574_*.png
    Saving 886th image 41080c42b524f36987b6b5dd821b5783_*.png
    Saving 887th image 2121752b8283749850307e4390a02cc7_*.png
    Saving 888th image d87c9751609ed8beaa478e8e43f6e8b4_*.png
    Saving 889th image 1600f0c80238f4c2c7b0a8e712a0938e_*.png
    Saving 890th image 5c2e7e9f7e67f3133a0bdf803cf51cf4_*.png
    Saving 891th image 4023ef8b2c807d99d3c8cf2ab3112701_*.png
    Saving 892th image ed706c6aa7add80a2093412e637d9850_*.png
    Saving 893th image 7ff88ac8b46915a97d30bcc2045676a0_*.png
    Saving 894th image 0ab89c76164bde35eaa5d42ee76b4370_*.png
    Saving 895th image 0bd19e4514202044015ae1f7da9dc132_*.png
    Saving 896th image 8ebdfec2c5e8b2c438048c65c2c65cf0_*.png
    Saving 897th image f1a784a848aa41f4759bbab6d94b2e49_*.png
    Saving 898th image 1e7e75a870caff201d90f09006351794_*.png
    Saving 899th image 8c133a4c9e7f8e76fc5dd71c8b09667c_*.png
    Saving 900th image b9eb7436cc00c1a2526b4c181106017f_*.png
    Saving 901th image 12d9ba31c434e6ac03a06e9241e46766_*.png
    Saving 902th image 005b02bcc91204a4cc1a115f74dc386c_*.png
    Saving 903th image cf1c5cbd088505c22a6b728471d3018d_*.png
    Saving 904th image 1c30bc645fba8146f9d1a8f64e86dc33_*.png
    Saving 905th image d51ec69d6781b003dfb94a18f678665a_*.png
    Saving 906th image e7157c7c3de096397d75e7025910ff1d_*.png
    Saving 907th image aed1616bab7db3667667094cc99595ea_*.png
    Saving 908th image 9630f3839a949e02fce526a186955c05_*.png
    Saving 909th image dbeee186ef430a6f0e1292fd4afdd4c2_*.png
    Saving 910th image 1164d5f3059f1c0f8257de942fe6bc8b_*.png
    Saving 911th image f28172857b55f9011da7cecc05b99cce_*.png
    Saving 912th image 098763e3f518276cc34e797c015e68c1_*.png
    Saving 913th image 46938ddf36c71801cd4a67ebacd4bd57_*.png
    Saving 914th image 0a9134c3223643138c3302c417f1e5d5_*.png
    Saving 915th image 2f9c6cba804cb24eeb4e1a1e870cdeb7_*.png
    Saving 916th image 3502f4074dc139e12f0cc4c636859f13_*.png
    Saving 917th image 48af109a06aff6d250f427e8a01050c5_*.png
    Saving 918th image 7ca0c1f1f879f5755fad85272174af26_*.png
    Saving 919th image 7315e8eba92f13657c74df52e5cad887_*.png
    Saving 920th image a067ba006a550ca872639c362199fbc1_*.png
    Saving 921th image 3f3f6dd37674b11cd5c7b032f2baff6e_*.png
    Saving 922th image 6aea72bfba9fb5508413414c93b5188e_*.png
    Saving 923th image 6fd0899ff04690e55f31be003bc495f1_*.png
    Saving 924th image 3d3f6f38e8fad357c3040a23f8d7cd69_*.png
    Saving 925th image 611bd3a743468daf408064722f5ea28a_*.png
    Saving 926th image d76a01721453c7911a823c6d9f61ac8b_*.png
    Saving 927th image 7be8d19e66a9fb397bd9a903c30940fb_*.png
    Saving 928th image e8a39ddb1bbb5748a2d5998c7c99dbb1_*.png
    Saving 929th image 7f85f901f39955b220d87aeab7a45533_*.png
    Saving 930th image 80b84f15995a194498d52a0974bb9a64_*.png
    Saving 931th image 9b95c5f4ff3f121e4532be61321abbb3_*.png
    Saving 932th image fdfa8a32c3fcc46137c1eb9404d4816c_*.png
    Saving 933th image a04820e4208ef7d7ea3d3c98710d5374_*.png
    Saving 934th image bfc9c5f6ce57121e0b6d643fb9f9b9fa_*.png
    Saving 935th image 9b1bb4a0f00fac732fcb48096293ca60_*.png
    Saving 936th image a379ca1420960268566812ca37dc3520_*.png
    Saving 937th image b17405557b3a3bad1abf6a06347d5306_*.png
    Saving 938th image c001fa0fdf474d0ccbc980df71c90ec7_*.png
    Saving 939th image 9552c43bee7a393ba357ce367685fb29_*.png
    Saving 940th image ed5cdadbc5935487b1a5647e941f059d_*.png
    Saving 941th image 8822f66e26070943931547541d6b5405_*.png
    Saving 942th image 993c9f79fb6916b7c05725711a17b63d_*.png
    Saving 943th image 0e3ffee27455c70d956d99985fb552db_*.png
    Saving 944th image e43abd5d9f0821aaae339fd5e338891d_*.png
    Saving 945th image 29d1a87646bc714c452e115fe3f2a82b_*.png
    Saving 946th image 1916ad4e31dbee88a4efc6fbb252bcd9_*.png
    Saving 947th image c4b925823aeb009b3bc1fce55a8c9559_*.png
    Saving 948th image 4c861a27d896ce7a806256a03efa97d5_*.png
    Saving 949th image 2fcf5d1012df8f50889156ae83a62313_*.png
    Saving 950th image e0801ad60bb5825a1bbb436637eaf73c_*.png
    Saving 951th image eba7e63f2c9bd2f3228c189c6c172703_*.png
    Saving 952th image 36c060fee6094b7dfb883d20e3f59cb2_*.png
    Saving 953th image b077980c6de4ab410f1c123835270b58_*.png
    Saving 954th image e46e3564df7714b717f2ba030104c077_*.png
    Saving 955th image 19fbdc7c56d85421b78d04712bd2b134_*.png
    Saving 956th image b3659ada63140e6adf95940a00cf63f5_*.png
    Saving 957th image b8554d135716a8d830a92ae446f4008b_*.png
    Saving 958th image 3eeb997f83d0182e690bd0805b2bb9fe_*.png
    Saving 959th image 9f4e497de07c0dbfc6433ba4bba6546e_*.png
    Saving 960th image 3e3c6523fb977fc76d0254dab8718aa6_*.png
    Saving 961th image e356b367f0a9246962a7425a6da15901_*.png
    Saving 962th image 444ccc5b665ee0c91d66a952136f7b3a_*.png
    Saving 963th image bdd0f65b5e2bc92cccce9ead32e98e3a_*.png
    Saving 964th image bc0b73728a96e9ab55d79ef01ff6ddcf_*.png
    Saving 965th image cc70f8ff9fae10ce96cde860e3a8f4e7_*.png
    Saving 966th image 2de4b5f97bc379599825cae337a1c35d_*.png
    Saving 967th image cafb9716954925a0a0412e4bb5c207d3_*.png
    Saving 968th image 0ee432e8692b2fadf5287c362e09152f_*.png
    Saving 969th image 3b90dfbcdff519965c2018f24cc88b23_*.png
    Saving 970th image ad2655ae7c5acd99c0658d83c7818330_*.png
    Saving 971th image a8c405e442b8b8220e47f18a3b09e7d8_*.png
    Saving 972th image 99654a11c3ead26f647b376342e87c66_*.png
    Saving 973th image 5a9f45714735de8e733837b4ec66b3a0_*.png
    Saving 974th image 791b7d09205f1ffc4b35e1f78babd4b7_*.png
    Saving 975th image aedfc3a488cccc8fbaf451066c79c355_*.png
    Saving 976th image 9e25a207154c3d109ccc3616e299bb44_*.png
    Saving 977th image 3281307625c26bf25700de66bbc8df22_*.png
    Saving 978th image 6328688a6d97fe38c3c4f4431e789a96_*.png
    Saving 979th image 275d78767ebc087bdfdf4656e22bc71d_*.png
    Saving 980th image fd2bc56d63cdbc0821030c5709b8c284_*.png
    Saving 981th image 4cc0f4e00466a8881521add810780a88_*.png
    Saving 982th image 4afa53bd00b81590981e18450aebe0f6_*.png
    Saving 983th image fbd8642a74d1ad28131fbda1224ee4e0_*.png
    Saving 984th image 6c05466464471a8182cdea55aef16674_*.png
    Saving 985th image 8732a9e03a7430f247dbe581040443f1_*.png
    Saving 986th image e5ca462da9e9b5130ba492d4b894f502_*.png
    Saving 987th image b2716c934761b582545f94697212bde9_*.png
    Saving 988th image 95308aa7c7788220d34db680f342fb09_*.png
    Saving 989th image 986abb837e88c4abaaea00c93ed129c5_*.png
    Saving 990th image 392755b99b3d5896830ffc1e9f718b49_*.png
    Saving 991th image caf9754270ce847fb5eb1bfc876b13ee_*.png
    Saving 992th image 277d6f0e021f7669235b306f4c69ea3f_*.png
    Saving 993th image ff520730fa960e14668cf8ba8bdc7298_*.png
    Saving 994th image 6bef82b2aa197877c02c28aafb5dbe9f_*.png
    Saving 995th image c94350dcd5c8e624618a59b3dcbb58bf_*.png
    Saving 996th image 64e6ce417e4cd1f9b185bd813194e7c6_*.png
    Saving 997th image ac9751c39f4baab74241cd6dce137f0b_*.png
    Saving 998th image cc31d9df86dba89e5e87c5e6066ea459_*.png
    Saving 999th image ba6ceedb654f968aae352c9f002056d2_*.png
    Saving 1000th image 592f6c8568f6f394e3cc48b171c884e5_*.png
    Saving 1001th image abf428aba868c9f622ca02e1d4b64751_*.png
    Saving 1002th image 7c6a126de8d56d7ef6208ceb6306f5c1_*.png
    Saving 1003th image 1c3b33f69033f2af506b0baf8da10524_*.png
    Saving 1004th image 3c6784c36398d6c84e0763ad6ed49cc8_*.png
    Saving 1005th image 4ec1f09a0c646f22bb1e5d1fb2efc185_*.png
    Saving 1006th image d7a298400518c25acbf60b27acb0445c_*.png
    Saving 1007th image c842ed62db72a06208d2c9e4d825498f_*.png
    Saving 1008th image 334268048d83788cc66ca43f687d9e1c_*.png
    Saving 1009th image 2b9bc98326ef419436820eadf44f0497_*.png
    Saving 1010th image fabc8aeff31a6a1317715dae07173211_*.png
    Saving 1011th image a5614e90a7980a2c70f100297b80c892_*.png
    Saving 1012th image 3d334a19e14ae7475baee2168dfc4cd5_*.png
    Saving 1013th image ac2a15d4807cb9bc05a6355083ffad25_*.png
    Saving 1014th image 1fedf13a9fafc75551e807ea9870f622_*.png
    Saving 1015th image 678b6a5e433856218edc9636cc2ec3b9_*.png
    Saving 1016th image 0f633209362ad9c5ef0b6b3030cb57b7_*.png
    Saving 1017th image 80a61c15ab6928e2f03c5b75c3c5ae2f_*.png
    Saving 1018th image e7669dbb71a4c5608a49969f2324cb9b_*.png
    Saving 1019th image 83b8652f6dfdac0235051b47f8d29a17_*.png
    Saving 1020th image 7d12131e081155a22b130cfef73d2206_*.png
    Saving 1021th image fd72bff1d7086cadc433617242992372_*.png
    Saving 1022th image 73516ad96b0a7ffad5c105c14bbf07e4_*.png
    Saving 1023th image 997706803316876b507414304c46e25e_*.png
    Saving 1024th image 6453b7ab8eeaf2351b9844ad46fa9ebc_*.png
    Saving 1025th image db90a97b4786bd1596b5cdbd7f85ee11_*.png
    Saving 1026th image 8385d568eed6bebe8260ac1cbda61be5_*.png
    Saving 1027th image a2a57c28b3cf0cd75ee6e7872186b0fe_*.png
    Saving 1028th image 44d4df1d4a5276d8536bf33ca023954e_*.png
    Saving 1029th image db2d48838b97f40c6922a8215621ffe1_*.png
    Saving 1030th image a116f3224db9c3659ed21a0242147cf3_*.png
    Saving 1031th image fe33ba28f35d14d7f1d26eadccd3d642_*.png
    Saving 1032th image 91b32e62ee3b8d7e3c4b671c7d5050d9_*.png
    Saving 1033th image 41081feed022f11a37d32a6725eb2f65_*.png
    Saving 1034th image 7949e630961c8331305c8657ab42fcc2_*.png
    Saving 1035th image 1f84415718fa42f988f1a45dc26882b0_*.png
    Saving 1036th image 7e5cc4627488fcb99d99840263aa3352_*.png
    Saving 1037th image e82374d1159df60abd4195b9cf25096a_*.png
    Saving 1038th image 4ea816213837ec3a4a83ee123bf17187_*.png
    Saving 1039th image aaf71b9f99b6347cad658ad305c32280_*.png
    Saving 1040th image 78385e8c74fd6223cdbb1d12d1503536_*.png
    Saving 1041th image 96a461004015aae9820ed2e849445b40_*.png
    Saving 1042th image 82c14a5936d9193a48f58c2bde2548fc_*.png
    Saving 1043th image 49b41b64a87ab79a2e692f804467c1a7_*.png
    Saving 1044th image 068161a5ecdcab2c53e2cbca7fd505b8_*.png
    Saving 1045th image b3616529d6515b214232c86940a88cb8_*.png
    Saving 1046th image 52e728a1db6ab7126ea57184cc1aef80_*.png
    Saving 1047th image 9e38e82c1c6942f51caea453d0ea444f_*.png
    Saving 1048th image fb856c68f169302790fad4ba59dc6160_*.png
    Saving 1049th image ec62a2b38d21d4b2b44a915b30427756_*.png
    Saving 1050th image aad6024353bd941a712cd0724daf770f_*.png
    Saving 1051th image 2c2024e9c3f542c128d4259518f52087_*.png
    Saving 1052th image 7c8faa48d79724c02053789df1aacc39_*.png
    Saving 1053th image 402ca6c5d6c0e3683a794c968dc460c1_*.png
    Saving 1054th image 4286b0e3b22aca5a4705a5d0648b2904_*.png
    Saving 1055th image e1c36eec289071b8eda3bd65b5ee6346_*.png
    Saving 1056th image 1ffd054018ee29da2d217bf9ab8d55da_*.png
    Saving 1057th image 99860f4012dcac91af302e64ed228932_*.png
    Saving 1058th image a1f63c885daa41708e39ab28d97c5c42_*.png
    Saving 1059th image 1dec049c7c13b48be7f6858588985220_*.png
    Saving 1060th image aff39f6237799ada6580a8512bfa008f_*.png
    Saving 1061th image a4b9d0bdd03da01193fbc977b0bbca57_*.png
    Saving 1062th image e255f0415c92a199fd7703af189c9c1b_*.png
    Saving 1063th image a5231deb2725c58cfc0ad2a1a65a1711_*.png
    Saving 1064th image e918a7eb126b20844a8e67c830ea4d7c_*.png
    Saving 1065th image 91a5e6773ecf3e18423e461df31626a3_*.png
    Saving 1066th image 32602d8f195ea46af086d6b733692492_*.png
    Saving 1067th image ac43b34a4318fddbf5158811660ae2b3_*.png
    Saving 1068th image ab7aa3079a376e9d45a6048c89bbe53d_*.png
    Saving 1069th image 0f2dcd593060b16402da96ead345a108_*.png
    Saving 1070th image 43f708582db5c26a7900d86c3cb8de3d_*.png
    Saving 1071th image 3a686a6c5d21d933ebb8552cf9861ea9_*.png
    Saving 1072th image 19bc1c19f27e2ea570fbfbfa558d44eb_*.png
    Saving 1073th image 7a4bbf715d0f17f2c7471ec09f64f098_*.png
    Saving 1074th image 7969a66b6ecc1010190b669f07d9c177_*.png
    Saving 1075th image db60ac4146c1ca6853077bacd2ea739a_*.png
    Saving 1076th image 10bf83983959ec552fce842c55bd3b04_*.png
    Saving 1077th image dfa1ecc080b64a18cc9a79c9920b78ba_*.png
    Saving 1078th image 0b40cbb892da3a6170f47df4563a54a5_*.png
    Saving 1079th image d7317a11d9e648108c37dff786b0fea7_*.png
    Saving 1080th image 31dd86596a5d2e3227140840a81b5742_*.png
    Saving 1081th image f9dbcd43289e7716af89daea09186e79_*.png
    Saving 1082th image 555feec5f7c17d22698922dc40818d1e_*.png
    Saving 1083th image b0710a44026ffebd68ea4720b8e10ab0_*.png
    Saving 1084th image 0d15b45df137af2d6b1ea34101ab20ec_*.png
    Saving 1085th image ee06136d7cfa76cd1cdb02eeebcb8f09_*.png
    Saving 1086th image a0c1b8f2e274e0d0b9ff4a9963d11936_*.png
    Saving 1087th image a6bdccef25609163617b1f252a2bb161_*.png
    Saving 1088th image 50b80577e17bf5d828c8f0b601d3bbac_*.png
    Saving 1089th image 306c9ebfca4172cfa6d9a1411fb17a2e_*.png
    Saving 1090th image 4de8602edaa3e9b67ab838b4dd1d5294_*.png
    Saving 1091th image c934b7a5fc9cd92ea4912dfe6ddca37d_*.png
    Saving 1092th image 550f605b3344aefac731d9449b731b29_*.png
    Saving 1093th image e74efb323fa7bda6f946d6da0923eb3f_*.png
    Saving 1094th image 49534dec5ab32a20a47f62fcdcf9ed6a_*.png
    Saving 1095th image 746cfcc1aedd9955e4772f2b4bdeaa07_*.png
    Saving 1096th image c8bce5f6e8407875be402ad97ec434d4_*.png
    Saving 1097th image 32cdb951859af5eaddf042b3a6708214_*.png
    Saving 1098th image 6e24bb267cbc17797653f2a8236d7cc2_*.png
    Saving 1099th image 9a8a7c097d39321640985303ea140ad7_*.png
    Saving 1100th image 3efebeaa2a0d10d5b1cf4ca45e1405bd_*.png
    Saving 1101th image e85cfd8c00879abd434b0016c4241352_*.png
    Saving 1102th image 98e1a6a55f591be660bda9fa3a97aa6a_*.png
    Saving 1103th image b803d7bcf8ccc621a96e4c9680a906c4_*.png
    Saving 1104th image 045f6f71632e1a938aed03e5bd269951_*.png
    Saving 1105th image c69344a7fb66abd928033864251c0ec2_*.png
    Saving 1106th image 295d861749ed98d5a04ab8435b778472_*.png
    Saving 1107th image 7755320ec60f5a7b64982dc9ae473d4a_*.png
    Saving 1108th image f3cc15c015ca3524923fefa2e5ead684_*.png
    Saving 1109th image 59b83edc5f2161f856f9ecf00a3af3f6_*.png
    Saving 1110th image e4c3ad8ec983e16f37de67c31a3b75ac_*.png
    Saving 1111th image d71a31d69e1869f960a4a8a49be80f43_*.png
    Saving 1112th image 28e0199f8871c0563c858a54fc33a266_*.png
    Saving 1113th image 6c300d7546aff76c31f4d47e54436816_*.png
    Saving 1114th image 627d88b670e5721ebc25d29ad5fa92ca_*.png
    Saving 1115th image 78ab0875973464c49321de73cd0c58b6_*.png
    Saving 1116th image 64377371f2c5dff9ffcba63ba096e4e6_*.png
    Saving 1117th image 15c6c471b8c3fe540e8b51504ada2635_*.png
    Saving 1118th image e629a234bf3f9158372a605cb1427beb_*.png
    Saving 1119th image ff5469fa8be671b1a519d0e7dcbf0360_*.png
    Saving 1120th image 7403c75dffba4200f56fa9cdf79e15fc_*.png
    Saving 1121th image cd755dd92b7f6522711b3749019d0588_*.png
    Saving 1122th image 75f126fad38aabae56f878ea4be7b6bf_*.png
    Saving 1123th image 6ac56a58a7e6fcebf989183f300c5379_*.png
    Saving 1124th image 4bc641fdcd6e2451da88273421a254d5_*.png
    Saving 1125th image 9dec0adb95236b021c6d8eb5b8da289b_*.png
    Saving 1126th image 9ad1f9eaf993760b22e42a569c98307c_*.png
    Saving 1127th image be471d9d179d5fbf8cc79f3f5b835516_*.png
    Saving 1128th image d195dc3fda9f566339f351c8caaae777_*.png
    Saving 1129th image b56727f0dd222dfb2743ca384f4038c9_*.png
    Saving 1130th image 556903d6b56c19f2bc501f4033129185_*.png
    Saving 1131th image 3a88d89dca46caea9c0421f52db98047_*.png
    Saving 1132th image 9cc4443616d0c73c041046fb047c38e6_*.png
    Saving 1133th image 4cd7cafe1d687b811d843c5e5a1bf923_*.png
    Saving 1134th image 03ff45dfe493d98308ff7dc4ced5883a_*.png
    Saving 1135th image d7ac4aa33890f9c2cfa432af901ecb50_*.png
    Saving 1136th image 916b23e930a0879dcf39309dd790087b_*.png
    Saving 1137th image efeab1982b24916902da11e53ff7c589_*.png
    Saving 1138th image a84327cd7543b259a7bb1c33edd799f9_*.png
    Saving 1139th image ef738a6537db9fe7a0fea381181f7b80_*.png
    Saving 1140th image 5a22d36c0805b6ad26653604d9150526_*.png
    Saving 1141th image c2dc00c3cec771883fedd5afb24a0b8b_*.png
    Saving 1142th image 2aaaeb4d55292bbbd7f4be758ed26b28_*.png
    Saving 1143th image b5f64264ef2310ba4a42992a36030fe8_*.png
    Saving 1144th image af69dd2e3829ab979594ee377c818837_*.png
    Saving 1145th image f5519f1c4bc5c06570fcbad1317f4357_*.png
    Saving 1146th image 0c11a48828f36dc3951c72c1afe5b019_*.png
    Saving 1147th image e1e21ea523322cbede0130814c190bd6_*.png
    Saving 1148th image 5bd3996ad4f98312ed5cf894d0f75342_*.png
    Saving 1149th image 6b555b9114228a617d6951f297642c6b_*.png
    Saving 1150th image a862f2a2692685bfb0bf94e29b97bdb2_*.png
    Saving 1151th image 3acec3bb0947b76d1ce2c8510bdecde1_*.png
    Saving 1152th image c601cdb745d443e71c54ee4d5b6e3a13_*.png
    Saving 1153th image 423c051dfe4452a0d780a9e97097fc64_*.png
    Saving 1154th image 5f533ef713d1081ad090aa0aec223822_*.png
    Saving 1155th image 2ca07a4fb995d4789bdde4504f255e07_*.png
    Saving 1156th image ba5a7fe4bc9e8bb52afc04ac64253674_*.png
    Saving 1157th image 396d3bd06839a66d2e238b68f387518e_*.png
    Saving 1158th image b3170d01071e998aaf4fc1ee6db21a41_*.png
    Saving 1159th image cf930726b4b8446dd200286368c3d13e_*.png
    Saving 1160th image a793e1892a067ee1add34529f1d6a7c8_*.png
    Saving 1161th image 831afd190b30efbf0800bfabdc1c2914_*.png
    Saving 1162th image ec211706c936facaa01a15338c812423_*.png
    Saving 1163th image bbb0be6716102a885de3296d80af3b46_*.png
    Saving 1164th image 03ffd706858b46d5a64d5baf4d263aef_*.png
    Saving 1165th image 752fca239c4671d0c5bb804921b6b4fb_*.png
    Saving 1166th image 50e68007c2e1d7b8b8c766791878360c_*.png
    Saving 1167th image 01482f314d3e06130fea91915496a870_*.png
    Saving 1168th image 09e06f34aeb946384e9bd7285897e26b_*.png
    Saving 1169th image d0a606d9c0fca7a60951e8d854ea5adf_*.png
    Saving 1170th image aa300a176faf504e5002cf1b42d422d2_*.png
    Saving 1171th image 3ffa4e3dd975b01ed905c0801ba8ce54_*.png
    Saving 1172th image aede8c7e2415dd877680af898ce52a22_*.png
    Saving 1173th image f1a04a021fdb990b7854d2e15518bdb7_*.png
    Saving 1174th image a09273d4e507358899daaedba3dba463_*.png
    Saving 1175th image bdc421249bbcbfae5f9c8e0bef5426c6_*.png
    Saving 1176th image e3802d7611a667afc10b16edc6fd39e1_*.png
    Saving 1177th image 3b21705b8546109a460a55d4335316e4_*.png
    Saving 1178th image b6ad8e4494dad55d2c6a1eeec4e9cc1a_*.png
    Saving 1179th image 670048241870ca88a808b2328c2e29bc_*.png
    Saving 1180th image 0f9b3ceae6fb4e757ce1cc5b385553be_*.png
    Saving 1181th image f732a5c0d358ea3bd8b3891f68153c58_*.png
    Saving 1182th image 1ce4dc57b508b22c86a420ce33f31925_*.png


    /home/harsh/anaconda3/envs/datascience/lib/python3.6/site-packages/skimage/color/colorconv.py:985: UserWarning: Color data out of range: Z < 0 in 6 pixels
      warn('Color data out of range: Z < 0 in %s pixels' % invalid[0].size)


    Saving 1183th image 862be8c4f2a40d4e86f8a7913f5b7bd6_*.png
    Saving 1184th image 133b1a18057a8d31e6a2944b6c36e3e2_*.png
    Saving 1185th image f3b4f0248e6c7677eb3aa2b52c92c885_*.png
    Saving 1186th image b4e45d058aba61850838742659a05181_*.png
    Saving 1187th image e482e553dfde43de80d01c2cd8d89e5d_*.png
    Saving 1188th image 0996810901da53495dbfb4fb50ff0b4f_*.png
    Saving 1189th image ff4e25459296d5732b5dcbee95b9af0e_*.png
    Saving 1190th image 661c95af82a7e09c3c200c2ebb081223_*.png
    Saving 1191th image 4a8629ae4213f51def1714bd599d4948_*.png
    Saving 1192th image 79683e1e0df3067d70f99d65dd7b8386_*.png
    Saving 1193th image 873dee08955d7c3eb2242502c1575c59_*.png
    Saving 1194th image 483eda6cb5af616e15a1c32c530bd3d3_*.png
    Saving 1195th image 314e65d844bd78cbaf5636eae702d2f0_*.png
    Saving 1196th image cd919c6255efbb2a19dc66c90cacab7b_*.png
    Saving 1197th image 281117506960a0d3fa6ff2bf9b002c72_*.png
    Saving 1198th image bbd324321f53259acde604001cf807bf_*.png
    Saving 1199th image 93587964403507137ffa3952647d1e0d_*.png
    Saving 1200th image d4ae53133071f8703109f0e36d31647d_*.png
    Saving 1201th image 6503d36e2d4db21b76c0c00729bb33b9_*.png
    Saving 1202th image dd0055c66b49ac738794f980bb199047_*.png
    Saving 1203th image 84ba65a6b225e78cffbdcc2b6e58c99f_*.png
    Saving 1204th image 5456440124c38b3e6027fdbdacd1dafb_*.png
    Saving 1205th image 62feffc15a363eaa99eb6062f8f1cd7a_*.png
    Saving 1206th image f1411b38aa36a5df1a0bdfe1ba7514c9_*.png
    Saving 1207th image c866c39b9100cef2c0a5cb87d678e2b3_*.png
    Saving 1208th image 6ae705b6205f5828163769842f64705e_*.png
    Saving 1209th image 3858cc58238ef55392812f968fa17ec5_*.png
    Saving 1210th image 88a95f486339650ea36331bdac8f2c9c_*.png
    Saving 1211th image 66a0f3294330e05e54663dab4fa54a60_*.png
    Saving 1212th image be052252319bf4f45c7565a15dc9651c_*.png
    Saving 1213th image b97b98d294768543f98f2df6fb299d0d_*.png
    Saving 1214th image 7a87f36035a7453ff7a06dc028dd9f37_*.png
    Saving 1215th image c271e3c64938ca744df91f07c946c91c_*.png
    Saving 1216th image 583fe5e222b4c06c1ed4457384ce103c_*.png
    Saving 1217th image 8f15bdaaaad2147033c1f11215b86cbd_*.png
    Saving 1218th image 84f68d790c9b019d642d7fbfee46c134_*.png
    Saving 1219th image cfe9856aa150507d1c13286f5c91c223_*.png
    Saving 1220th image 8527075d713bc5ce753028766efddbe3_*.png
    Saving 1221th image 3f7fdb0aabdeddfcea792b134e092e6d_*.png
    Saving 1222th image b1c2bc6e2c98d4df875668b8abfd3433_*.png
    Saving 1223th image daabcd05bd6efafea4e12f4adad5121c_*.png
    Saving 1224th image 34749fb4e8473a5db1cb43bb81cf76e4_*.png
    Saving 1225th image 0edd96448c1a9f6123c4b4e14abc6cdf_*.png
    Saving 1226th image c2973f82f51b2db583bdc92809a5077f_*.png
    Saving 1227th image 4a1df7cf5d7c80df46ade620895f346f_*.png
    Saving 1228th image 3135693a412ab2b6c38bc868ed187d4a_*.png
    Saving 1229th image e7f56dfde6b9053c9a737f573806be92_*.png
    Saving 1230th image 416117dbb4f5360be9bc675371f2ecbe_*.png
    Saving 1231th image a15ec3bb5be849c99fdb3e35fdcd00ab_*.png
    Saving 1232th image 32c07da9c0b06a99c795187f6675902c_*.png
    Saving 1233th image d62c0398c5efbcc68dc04cc4cdf4954e_*.png
    Saving 1234th image a706b95082f1b98034685d2e0bf62612_*.png
    Saving 1235th image cccc1426b8387913a849d8dfd1cafa0d_*.png
    Saving 1236th image 66493ebe4c533005940016f689f6e6a8_*.png
    Saving 1237th image 07690e53e1758459895f56398d1ae2c9_*.png
    Saving 1238th image 1cb8996d481859489aa60ea2f97a48d6_*.png
    Saving 1239th image d14b6dcb97ad9bf4b1b27737e8095276_*.png
    Saving 1240th image 4896e428deb956f70b6d1a49558b0080_*.png
    Saving 1241th image a3dc64c56bf3897eb9191d20ff898e53_*.png
    Saving 1242th image 9281e830dcacffc2889e9f116e7b96e2_*.png
    Saving 1243th image cf1c4e22f02463e11e947812d270ec75_*.png
    Saving 1244th image f8b28fd9285e6497082fb57b0ff589b4_*.png
    Saving 1245th image 8e6fbce044c55b402156837f7745adf7_*.png
    Saving 1246th image 1b2241922856bd45f41942375521758e_*.png
    Saving 1247th image e15e16e8cce591f51de7d6b32eff8ddf_*.png
    Saving 1248th image 61623b48c807ec8afcf0093cbdbfcec1_*.png
    Saving 1249th image 4f6a6b6e989073579df399563df3a3eb_*.png
    Saving 1250th image 87eb68abfc03a9e657e695bc2158aed6_*.png
    Saving 1251th image 260e22858c1cd44161afcfc38624238c_*.png
    Saving 1252th image f9e34fcf3cebb67c29023ac7a423c6df_*.png
    Saving 1253th image 098ba8cce7dd7b8e1d949e9e462f1f0e_*.png
    Saving 1254th image 6651cd2e409452d1a5dcf2b0167a9cb9_*.png
    Saving 1255th image b05e002b2a66042e65e4d128befd42cb_*.png
    Saving 1256th image 5f9854bcf9a127ba8d6a633853ec3df8_*.png
    Saving 1257th image dbbeb3682bcee6bc6bf4de4b9658eb1b_*.png
    Saving 1258th image 18e219578a9d534f22e11cbc8157543a_*.png
    Saving 1259th image be43a658f4e0a9c907487225381045f8_*.png
    Saving 1260th image 5e4c4682c2277662f0320cec5425136b_*.png
    Saving 1261th image 05dffe572340b5a6fd7b8b688fa2b9f1_*.png
    Saving 1262th image a8c6300d23f47faac0cb8bcf223b508d_*.png
    Saving 1263th image d8b25ca9fb1fd6a58ed0b8a463de5bc1_*.png
    Saving 1264th image b3f29532da8e8c10c2ae166cc28b09f1_*.png
    Saving 1265th image 47338b6bb2ea73c1e882f23a52575317_*.png
    Saving 1266th image 7d1117e6295cbb594efc51bedad79921_*.png
    Saving 1267th image 1c67d6ea0272da47e94a67dd8c7ba0ec_*.png
    Saving 1268th image b199f9acaffedb53166e3dc363349186_*.png
    Saving 1269th image e70fbd1949cbd0c47372b3809da0ceb7_*.png
    Saving 1270th image 221cc9516e2a976ff9cb546bc43ee011_*.png
    Saving 1271th image 4f948e422c93b7c69ed7165d431878c5_*.png
    Saving 1272th image 2117f1aa1068a962220a4ef7103f0be6_*.png
    Saving 1273th image 81818f2b148fa6848ae2589edbe9e988_*.png
    Saving 1274th image b4d3e0093f3e19fb71fd95b579bad764_*.png
    Saving 1275th image 5623893673c612412ee1317829bf2b6a_*.png
    Saving 1276th image 1b833121c385b2a6c91a75cbd3200b3c_*.png
    Saving 1277th image fbe93e4842a658d085cd9011e2fdc876_*.png
    Saving 1278th image 834cfb98ab3710a7508890a2678357b1_*.png
    Saving 1279th image 431e9de000f69b569178fef444432d2b_*.png
    Saving 1280th image 7f11a1ef751fac2be20b15f103e05183_*.png
    Saving 1281th image 1622e885eacc92de0e86d824324694a4_*.png
    Saving 1282th image 7845c1121bac577a7ea065b46bbc2f35_*.png
    Saving 1283th image 26a603132aa4c19a0cd1c13d20baf792_*.png
    Saving 1284th image 8c7a5fb33afe5e7b8ee4c5a08991fef8_*.png
    Saving 1285th image 247bd9d202212db661e8aaf8dddf12af_*.png
    Saving 1286th image 32b6c1964c99d30827fd91cedd45f9e9_*.png
    Saving 1287th image 384d781d67186d3f3727d1e6ce162064_*.png
    Saving 1288th image c8c5e97fbdd54b7b1cf6c70c1af5acae_*.png
    Saving 1289th image c98d466cee0124eb9b3e75bdc2bdde75_*.png
    Saving 1290th image e3236aacc912a78f0dd2535f65ac4ec7_*.png
    Saving 1291th image ba10cf601a6af2289b3f61a5ec00163a_*.png
    Saving 1292th image 9ca83a7941150bf17cf5272eccccf799_*.png
    Saving 1293th image 56696b1d7bb5957904fde5f1ab4acfc5_*.png
    Saving 1294th image bf635afc7e316348c81c2f421e1f4090_*.png
    Saving 1295th image 396f0f747cd2e916c941b06650e2c7b3_*.png
    Saving 1296th image ae57259cd269e1c65587dd69d1a991e4_*.png
    Saving 1297th image 9395c3100ee5e9a5fc0fa474965cfee3_*.png
    Saving 1298th image abf435e23b8e87d70b1083ccf5080b69_*.png
    Saving 1299th image 0acfc6c2326f92ca0636adcd29c2f198_*.png
    Saving 1300th image 089bbc930fb33884522c5397947ee2df_*.png
    Saving 1301th image 7b11b3be5152d3fb3e27d485c95acd04_*.png
    Saving 1302th image 6bb832e931bd6cfd866236ecfc378d3b_*.png
    Saving 1303th image aed001040db385130d97471aa44ebd27_*.png
    Saving 1304th image 225a24f316a80eb22dd456954f4a0e65_*.png
    Saving 1305th image f1c2fba4d2096ba1b7844b9bcf52f527_*.png
    Saving 1306th image 4541b3a29b395d6269660b6be9f336be_*.png
    Saving 1307th image 11ba255a96bed9fd4b70e07efbfc0fe1_*.png
    Saving 1308th image 8d89852b1699dd71ebc608bbfc38b865_*.png
    Saving 1309th image 1e83b8915a238b9ec23f1748edf3815b_*.png
    Saving 1310th image 838dd6e5b8a192e394400302fce0d9c1_*.png
    Saving 1311th image e5dc635bcbb636beced91484f3244a7f_*.png
    Saving 1312th image d592ab01b2e8e7d13591b8f2702ea795_*.png
    Saving 1313th image f9054a1f50c66ccb532b94f8dd4ab11f_*.png
    Saving 1314th image 0844b0da5fb7b86733a4eb6f4791b256_*.png
    Saving 1315th image 7f99537de6cbaefe54f7039dcd69227e_*.png
    Saving 1316th image 135a3e4f9b13ee6356fd79ceca627ef6_*.png
    Saving 1317th image 00c3c64630a5835da80b4a6145e49554_*.png
    Saving 1318th image 782a2220dff658e948690ab72435c936_*.png
    Saving 1319th image ee833c137da22c511c6759101ea2511c_*.png
    Saving 1320th image 86548f215960c8b1ccd166d142194f2f_*.png
    Saving 1321th image 07de0c78e41447ca83362535852c1bb6_*.png
    Saving 1322th image 9c8c01c1b5bf7ef53485f8293c470f67_*.png
    Saving 1323th image 4cf165d3c01deee7b224ea9c22e1c59b_*.png
    Saving 1324th image 241ee2a2957bc81f189bb3d5a1748bf9_*.png
    Saving 1325th image 5392e2f1326e28401c870cce6aa0d5a8_*.png
    Saving 1326th image 420a27017029fdde9c45bb3edb1bd39c_*.png
    Saving 1327th image 834ea79db9007bde42a10f40eaff9011_*.png
    Saving 1328th image 27d2cb4cac2a897f13d72c909b3df504_*.png
    Saving 1329th image 4d8bf8ab20b349745143be7e4c9142ed_*.png
    Saving 1330th image c7d284ea162c17f46f5d21dc85042767_*.png
    Saving 1331th image 5ed04218b070be2217d8de259ff90186_*.png
    Saving 1332th image 83d2de5c8bcae3fab5bcfa1c56876a8d_*.png
    Saving 1333th image 8ea402b26af12870df01620cd779129f_*.png
    Saving 1334th image 80f2e61b51cdd8d39fa0bdcee592f74a_*.png
    Saving 1335th image 701096ef2ec5b4f0edd5eb8aa1f8b3e6_*.png
    Saving 1336th image 7e391b54c5073b37f5e0fbf447861b66_*.png
    Saving 1337th image 90b976509a219ab0f65c1bb52d8b9ee1_*.png
    Saving 1338th image 9192f90811231146c89df83103c35123_*.png
    Saving 1339th image 051785a431a3388049e06e768b841c71_*.png
    Saving 1340th image a9bf0385341ee5f611b2ba9c784d8838_*.png
    Saving 1341th image d64ad743cc8ee71b5e13b7c3643fffc4_*.png
    Saving 1342th image b3847b04ffcc281751776f03296050ef_*.png
    Saving 1343th image e1e7c1d46fdf72fa4366a44e5975d97a_*.png
    Saving 1344th image 45d73e678e17e4b55d41b7d88ea6d0cc_*.png
    Saving 1345th image d985911ac6e51a077d7c363b5454cfc4_*.png
    Saving 1346th image e71295f6b5c0379e68021ce481f94e42_*.png
    Saving 1347th image fc4d92a2a0125f992500bf5778b002dc_*.png
    Saving 1348th image e385cc7100af305c756d97f081b9b5b9_*.png
    Saving 1349th image 4fb367f17ed22a37eab84b297edf6215_*.png
    Saving 1350th image e1172ef181db793e084b4c72e278b847_*.png
    Saving 1351th image b4c8944bf951512607d9bc3c92004051_*.png
    Saving 1352th image 29ceb1d7d62b52ccfa8d9d0dbd33246b_*.png
    Saving 1353th image ab8629fdd6ffeb78069071c34c802027_*.png
    Saving 1354th image 1163968237e0fe18ab06ff654f2cb670_*.png
    Saving 1355th image d16f0a54f2618b48750479b1c2819776_*.png
    Saving 1356th image 31e7a5349f26731853fe882a4f561154_*.png
    Saving 1357th image 9dd7592dc0eea1fdca48af26baf6eb6f_*.png
    Saving 1358th image 63faa33cbd81e85003946075558ee4c0_*.png
    Saving 1359th image 1aaa7bb515108f7a0c258e9d7d06b8d3_*.png
    Saving 1360th image b54219c3d83c9daaae8fcd0367b57ea6_*.png
    Saving 1361th image 9fe0b73ba65ebc40f7a2d6487eb5acfe_*.png
    Saving 1362th image 2ef90508c1d4563ab8fb5fe27d0447d1_*.png
    Saving 1363th image 80ba7126724aee1038b81cd077d38a7e_*.png
    Saving 1364th image 788448a8f07017b163da1793b1ef45f0_*.png
    Saving 1365th image 069c004a4b1d58fa1dc8212f104f5760_*.png
    Saving 1366th image d55ef85304eab552aa9a8e88205df340_*.png
    Saving 1367th image 461d380ed6f3169dae2b165d37dc5db3_*.png
    Saving 1368th image 67a18599968f72c9d36b3a963819754f_*.png
    Saving 1369th image 3da2e4967543d6983e364c4cf617032d_*.png
    Saving 1370th image a3de84639888442da36a24be2e25f92d_*.png
    Saving 1371th image 633979a3d09cfb134b263f0e7ec9a11b_*.png
    Saving 1372th image 1bba933e748d68ea5d50deb59c56e53c_*.png
    Saving 1373th image 36ef750c3ec0f5b626a97c725e4637a7_*.png
    Saving 1374th image 1e47327e14297a9a7b85a0a097b60406_*.png
    Saving 1375th image 8e86a894c3c2725c4940eb01b46a4f8e_*.png
    Saving 1376th image a43d2a463bdae5add312e2e6621b54b5_*.png
    Saving 1377th image 2d59e222b99d77c0950172afc9fdcc37_*.png
    Saving 1378th image dd5116ae425ba8defd3f162c99b31287_*.png
    Saving 1379th image fb423de190545d5027c827e0f4939348_*.png
    Saving 1380th image 674e64cd06655f13e9affc93add7ba60_*.png
    Saving 1381th image dd6e73a456ddac95f182ab64e58984da_*.png
    Saving 1382th image 051aa0395dfb16db394153bd6d3c5ff4_*.png
    Saving 1383th image 137a805c988c58d237c007f277180228_*.png
    Saving 1384th image 607ed4060fecfa48a34a3a80124239ce_*.png
    Saving 1385th image 780ae062a0649bec4a31cb1af1529a44_*.png
    Saving 1386th image ec6fb4d7b7d7a9ded162a924533f0733_*.png
    Saving 1387th image 423e7145383bec651306f46eab851510_*.png
    Saving 1388th image 52246f882206b588ad96d03805937e37_*.png
    Saving 1389th image 9d58b244f1d8242c22904f721ad296c7_*.png
    Saving 1390th image aa5a1a53dc9bbd77a991c853c7a01fa5_*.png
    Saving 1391th image 0601a81706860cf3b4e7ea4a08472818_*.png
    Saving 1392th image 03c1d31ee1879ee373d50fb568db470c_*.png
    Saving 1393th image f46c227bdfaa3c9d6416e606b56a268f_*.png
    Saving 1394th image dc12367e2aea00b15760dcf8fd619221_*.png
    Saving 1395th image d3261f54e3446921f6cfedd5d123543f_*.png
    Saving 1396th image 32f20cca2a0303093baf8cdb5fc3aa5c_*.png
    Saving 1397th image fca1d256cdd65af146188b3ee1379cc2_*.png
    Saving 1398th image ea991af5903b7a567fe3c057f4abbd05_*.png
    Saving 1399th image be3b3edb5578265f6515c7ba562c7244_*.png
    Saving 1400th image 2d59e4a13a784e26c86049eaf1851385_*.png
    Saving 1401th image 17c3131096beba3f23c3146d6c282790_*.png
    Saving 1402th image d62e22a60029b4f9a900b2487084e58d_*.png
    Saving 1403th image 542591986638c301e676158703fe8d6d_*.png
    Saving 1404th image 97768433ea4b89a13c7aeec6221f346b_*.png
    Saving 1405th image 92e5093aed3a61461c6027059665d59b_*.png
    Saving 1406th image 56641605222ca10a910fda897bff76f8_*.png
    Saving 1407th image ab750b967659249729e0f77fbf4192b4_*.png
    Saving 1408th image 72a30fdac6309632f6b3f49d968259b4_*.png
    Saving 1409th image 17cd2d9b55b2316ea55085cf797efe74_*.png
    Saving 1410th image 7e1e8e886d50744e1e13d1e96d27add6_*.png
    Saving 1411th image bd84dd7017820e55ecea50a19d220a15_*.png
    Saving 1412th image c4a857c9c6ee6803254b8f2cff070b9b_*.png
    Saving 1413th image 030f5b1d312f537cbcc061f1a043cca7_*.png
    Saving 1414th image 8950ea69204b2b260403b97dbfb5eda0_*.png
    Saving 1415th image de2b3992ed901f934e78cd73a782edc2_*.png
    Saving 1416th image 8927bfc280ab3f9ce0c48170a6f39321_*.png
    Saving 1417th image 3e79717ac62af647c6b52d83fa3aa164_*.png
    Saving 1418th image 8d6a3cd682e91a1b1e715b17cbad8e53_*.png
    Saving 1419th image 74e04cea5a5ce0e887584005929700db_*.png
    Saving 1420th image 49b637c05fc1df962897673e859aaa3c_*.png
    Saving 1421th image 9176438ab8a80838e60a55a0f65e8491_*.png
    Saving 1422th image b688812b82f3bd1230f0b7724e526717_*.png
    Saving 1423th image cfe9767225466b2f3fb63456979e1d2d_*.png
    Saving 1424th image 52a739ba81ea2d0282fe98a26a382dfc_*.png
    Saving 1425th image df4f3c67132e985444a9716934310e3c_*.png
    Saving 1426th image 7d44dd327ef4e2dbce477ebe0777df71_*.png
    Saving 1427th image e50e8a96b54ea3e5709260bba605c1d7_*.png
    Saving 1428th image 4c5963b1cd7d3873198aafa64004f3f2_*.png
    Saving 1429th image da82517abf877918e6509cad7bdf53e4_*.png
    Saving 1430th image 3bc15bd78a1b51ce46257da961704d97_*.png
    Saving 1431th image 5a232e2a678f4165ab7c2a034c24b4c9_*.png
    Saving 1432th image e240ebbfbae3f2384ab5928acf00e8ee_*.png
    Saving 1433th image 9b1f7fcdaa1107c9bd81ff1dbc9c95f7_*.png
    Saving 1434th image 721724706b08e8d7529a420349678548_*.png
    Saving 1435th image 7200be8243def59902de7d335498ff9c_*.png
    Saving 1436th image 8226671a4e0189ab2938eea60fcf174b_*.png
    Saving 1437th image 7e58d509aeeda1e6e5255fc83f3ed01b_*.png
    Saving 1438th image 1598cb096052e9ffa614e168bd994468_*.png
    Saving 1439th image dc516b5e3c743012ea3ea9c10ffbcda8_*.png
    Saving 1440th image fa0d2d30d3f2da299e29c9b47dd41b57_*.png
    Saving 1441th image 5bd5baca940d4dc45d44f1bd1d70cd09_*.png
    Saving 1442th image 8f60921ef69f938648388685b4597063_*.png
    Saving 1443th image b6f38fe948075a2a5053f30e1bfcf011_*.png
    Saving 1444th image 18d6c4a8a4976d405f6fcc7b1c90e7a6_*.png
    Saving 1445th image ddc82146ea154cbfde4ccba3ab086d17_*.png
    Saving 1446th image 8c40d53d0c02f0925cf8b6ed281738b6_*.png
    Saving 1447th image 9a2b30e454947453effd49b7989fbb97_*.png
    Saving 1448th image 04c02608e1ff7437d9162863c896438c_*.png
    Saving 1449th image 9d38554ec20bdb42e47d9551c467b2ec_*.png
    Saving 1450th image 37101eede80de86d9eb4168cab7ec2a1_*.png
    Saving 1451th image 3d57b3caa5d39e199d7baa326ac28a57_*.png
    Saving 1452th image 7600f8441b8f68ac6833b24973b59092_*.png
    Saving 1453th image cf9747b57b24cde5a1683d6a0295671f_*.png
    Saving 1454th image 02c326f2a1dd52f6c233b43a7f1c6e34_*.png
    Saving 1455th image 172c0ced845ce0dd2fcd0ed08419e4cd_*.png
    Saving 1456th image 672dda784be0a09c728c892295e282c3_*.png
    Saving 1457th image 01b51162867a8c5de18d9538b74ec568_*.png
    Saving 1458th image 5eea45d279dfe1fa64b9d498dc02d826_*.png
    Saving 1459th image fd0b94b74786c4107334089f39c90678_*.png
    Saving 1460th image 9497b4f8f22caac484e1f4e8cd892640_*.png
    Saving 1461th image 2ab003d56a8128c3f8fc86debb83502f_*.png
    Saving 1462th image d4b4a28d815638d7bfc9fdabdd84d9cd_*.png
    Saving 1463th image 9d6acf10d020010509315c0aaa339774_*.png
    Saving 1464th image 9830c059e78d055a3b86b3f8f3ade3c2_*.png
    Saving 1465th image d9d3505ea4abf0dae14591d11f46d3b5_*.png
    Saving 1466th image d26f0daa7a8008cc7c1b70c66f65e72c_*.png
    Saving 1467th image 8a32e7a0171b0ef5adc3cad0fa9daf79_*.png
    Saving 1468th image 1bac39cda87c66016519a125f244ca78_*.png
    Saving 1469th image 60b97580e482ce7ea3dfcab19df940ed_*.png
    Saving 1470th image 47199f4143c926be0cb6d396437ef9e8_*.png
    Saving 1471th image c2647ed019e7f0b901f1e16cc376401a_*.png
    Saving 1472th image 7ce4126fd0bd2cb92bdc73114cea48c6_*.png
    Saving 1473th image 011ce78eccf251dac0984eaffe2d896f_*.png
    Saving 1474th image 62b17851f45581d862ed0e35a4e06094_*.png
    Saving 1475th image aaafc031998eaa4bf9643eb362142dd9_*.png
    Saving 1476th image 5f441cbd81b354fde0e86a6053d4fc50_*.png
    Saving 1477th image 754fde044cc38fbd22e9aa392568838b_*.png
    Saving 1478th image 9afd017393660fc5366342355c350d42_*.png
    Saving 1479th image d92db715f5d213392bb502dec340178c_*.png
    Saving 1480th image a948ddfaace59f37cd7ee0eac9502252_*.png
    Saving 1481th image a73286c465354078786a591b5d3c9d0f_*.png
    Saving 1482th image c01b0aade164987e386e1a3791b5973c_*.png
    Saving 1483th image 8cfcb2caedb325fb3ae764ebc44412b0_*.png
    Saving 1484th image 01e0d1337eb023727058abc0fd96827c_*.png
    Saving 1485th image 2d91d87bca03be835a1c989d14daa349_*.png
    Saving 1486th image c7cf6d56d2ec061ebb42ce731c91860f_*.png
    Saving 1487th image 9478c10d3e8018e3430e9f0221110d88_*.png
    Saving 1488th image ad486306c33b62cd4c518c15893f7009_*.png
    Saving 1489th image 1a5b23fcd3f0a99dd1d2d9de8294083e_*.png
    Saving 1490th image 6f9123e1728f3dc6ff41dc5bdc47f1f0_*.png
    Saving 1491th image 6220c2aa673d2ee5e4281af3a057916c_*.png
    Saving 1492th image f84d5731f734463ff2dd80f6d2fd00e3_*.png
    Saving 1493th image 3d8230ceba2388e831d2f4a702a4781f_*.png
    Saving 1494th image 7df54568293ec1739288a906576f1319_*.png
    Saving 1495th image 2a399cda4c567392b5b5623de921dd1e_*.png
    Saving 1496th image a920f52828a0de9e0deceaf110d731f4_*.png
    Saving 1497th image 2e9bb2d8ed5f843521d812e0e86befdf_*.png
    Saving 1498th image 06d36eb87d64e67f7904f79c5e8d6b17_*.png
    Saving 1499th image bf13406c44e066202a82a7f603c1ab7d_*.png
    Saving 1500th image ca7730f682dcd7ee8e93751485f3d874_*.png
    Saving 1501th image 51c85979f6a97df88be3d7171320fe33_*.png
    Saving 1502th image d72c5f58d26d981967f0fe500866cdc7_*.png
    Saving 1503th image d76c9981fb633665411c5bd5148e67f9_*.png
    Saving 1504th image 916c4175ad0263d1f656ce2c527a59e1_*.png
    Saving 1505th image e6ed91cb5dee75341e3fbfecc9d8f245_*.png
    Saving 1506th image 8a625e2b2ed3f434ff20f280eb08ac46_*.png
    Saving 1507th image 893bce5759c5ddca7051b9794220bf39_*.png
    Saving 1508th image 04fcc582ef18c7783732b64234c97315_*.png
    Saving 1509th image 8cc8b5c3f3c2d91ecbbd20446fe0314b_*.png
    Saving 1510th image 5a49a8a5d047c84f2fff3eff1f5b6d9c_*.png
    Saving 1511th image a7587a2a7be2ea1cde5e9c8db3ef0ffd_*.png
    Saving 1512th image 86b0e7037381606544f1060c4b2e3b0f_*.png
    Saving 1513th image 47666c0f9bb5c9990353fcc02c703637_*.png
    Saving 1514th image 334cc957e7def018ea1598cc1beb2d66_*.png
    Saving 1515th image d226248e9f53bf361be79337f12b834a_*.png
    Saving 1516th image 2f4361b3f3e76724393505b0c0f6736e_*.png
    Saving 1517th image 2becf185e9160ec0f73343680b618b7f_*.png
    Saving 1518th image 4ecda9be1362eb5629cecc95aa686150_*.png
    Saving 1519th image 9f81be2d4092755167ed15d02f07bbfb_*.png
    Saving 1520th image 434465383617023a8ba4df72757185bc_*.png
    Saving 1521th image f7d30e41c72931e8087c1b4acb16d680_*.png
    Saving 1522th image 1293b10d036806f6e675e906a22b13c8_*.png
    Saving 1523th image 179e1aa91f2bac488baa37233d87142d_*.png
    Saving 1524th image 6ac731d39cf71a2f8415c418004c2dba_*.png
    Saving 1525th image a445b98bf4f4b24536d9e81e52a62be8_*.png
    Saving 1526th image ffecc6bf4358e8899d2a466449fbfcad_*.png
    Saving 1527th image c3f2ebf6081e91cec6375d02d87dea49_*.png
    Saving 1528th image ce849ddfdf8e8913095a09ff764603e3_*.png
    Saving 1529th image dbefd932bd5f6d19f26e5dccd5dce7fe_*.png
    Saving 1530th image 062ed9e13036a6a017831ab9901261b6_*.png
    Saving 1531th image a53aec7fef1e23e66ca8c50b932f1837_*.png
    Saving 1532th image 7010bf578c7d14c703073432c1ed4977_*.png
    Saving 1533th image 11d9f12450d7f217d2d6a4145303b5fa_*.png
    Saving 1534th image 515d365c3219b7a76c13fbad8c62d58b_*.png
    Saving 1535th image d45c364903de96f9d834bd6e886c3b6d_*.png
    Saving 1536th image 6983cf2724bd5ee088e39f58ae63169d_*.png
    Saving 1537th image b90648b9e4759d124f4efaf181b5135e_*.png
    Saving 1538th image 846d33c7fbd3b725c1afcb4cf039a626_*.png
    Saving 1539th image 1dc03391246a133a4918e75623800a9c_*.png
    Saving 1540th image 9ae7f16c4dcda571f5e9bcb86ef218ef_*.png
    Saving 1541th image 22a6a64cea0deefc3ae2a69cc8f742af_*.png
    Saving 1542th image 9b7ebb03572a09e128ff5b6b5a474cac_*.png
    Saving 1543th image 11718b42d2c3bb550fc8480d14c03bea_*.png
    Saving 1544th image c1a5b0bdaa0320eaecc5fb633d362e0a_*.png
    Saving 1545th image e47a4aeb024eb273f4110943734a38ec_*.png
    Saving 1546th image 3b12fbfea9eca1103ee87b7aaededbcc_*.png
    Saving 1547th image 8f094d1b5d7087fa5f7eb9d04727a012_*.png
    Saving 1548th image f5e6555dd3bba0bb5d54340af5996110_*.png
    Saving 1549th image ed4865f5f3f094ed6016d0f611e4f556_*.png
    Saving 1550th image 9f068f476266fc80b9bab83f6910cbc8_*.png
    Saving 1551th image b2b3a153bae55c1d11808ab3645be0bf_*.png
    Saving 1552th image 21270a000ebe3230caf29174accae851_*.png
    Saving 1553th image 089d995ee3a81c0ca9762c4a1d68d3d4_*.png
    Saving 1554th image 955cd420b30366042080b62ccd8c7ea4_*.png
    Saving 1555th image 5881b81e5c4de3e9907bced3ef35f350_*.png
    Saving 1556th image 54ce734ca1cfa2ab9641ea243d0c4f20_*.png
    Saving 1557th image c80a5ef82ccdbceb3b260cd0ba336bec_*.png
    Saving 1558th image 34ed8e87fe7cc6798ba59920c68f5b72_*.png
    Saving 1559th image f4fc80199be789d181db925813691acf_*.png
    Saving 1560th image 79f906356fc2a7b67fe65949e7df8a3c_*.png
    Saving 1561th image 795debddb8cf43b980f2f567402225ee_*.png
    Saving 1562th image 7fab050bd6afd729ee3221fd3ef2c526_*.png
    Saving 1563th image 9ba3996c8d05031dbb4f0f948877b93e_*.png
    Saving 1564th image e76d68ae814831d105aa87cc0e4967fb_*.png
    Saving 1565th image 54970533b2197cc5436fe760c78e1096_*.png
    Saving 1566th image f2bde03e8bf2cf8af600a17d272dbf7d_*.png
    Saving 1567th image cb432f054918dc7d4dde1a1b7fedd732_*.png
    Saving 1568th image a94ca1e409f21746a0b03fe0eed75701_*.png
    Saving 1569th image 382fb76bb63d4ae21e3f6ec701c26cc0_*.png
    Saving 1570th image 43f95a7beb9332c81592d0836eb1b154_*.png
    Saving 1571th image f284107b405f899db8ca82cfb0a81cfa_*.png
    Saving 1572th image f7a8efef0b73d8df556cfdb6ef4d50e0_*.png
    Saving 1573th image 3204fd2a5750b96494b3ed57ff8004c6_*.png
    Saving 1574th image 96c8d42294039cdb6df1019935b9d790_*.png
    Saving 1575th image cadd0f1367150ce7d9e06cd376567e6e_*.png
    Saving 1576th image fd6f360194ed8641288f6330fbb4310b_*.png
    Saving 1577th image f091c2e61c9bf791e8600fe60ea49aed_*.png


    /home/harsh/anaconda3/envs/datascience/lib/python3.6/site-packages/skimage/color/colorconv.py:985: UserWarning: Color data out of range: Z < 0 in 11 pixels
      warn('Color data out of range: Z < 0 in %s pixels' % invalid[0].size)


    Saving 1578th image a07cf5602b4afe1f487926f18693bb96_*.png
    Saving 1579th image fc56d8fc114d36aa5716f7026d3ce521_*.png
    Saving 1580th image a4a9472dbb48a1e27e6f8cc8e8b6e3bf_*.png
    Saving 1581th image e2685ba9b5ca6577e95f3cefd2a7518c_*.png
    Saving 1582th image 79e657709a82f255e4924f87273fc0bd_*.png
    Saving 1583th image fc46f484046c3142b5f11a77dc225087_*.png
    Saving 1584th image 8793e2896033652d6d0f91b8a0794735_*.png
    Saving 1585th image cb6ccd52a4bc82c341d21d627a149da7_*.png
    Saving 1586th image e4ed078b44df91a4936e3e232ac224fc_*.png
    Saving 1587th image 2fe677a67b69cc7c621af776c210fdde_*.png
    Saving 1588th image 765226ced03b7f3f364a3d96a777453c_*.png
    Saving 1589th image d23cc7e2dd21a2a275d673216fb9c8c5_*.png
    Saving 1590th image 580159861f106b0f2aac32f9372d84a1_*.png
    Saving 1591th image 894192769c7a4d2852141b19b9f5618a_*.png
    Saving 1592th image 8ab957b92849e2439b5488c2d3d1d309_*.png
    Saving 1593th image 44b26560a29b12dd812d058159d48adb_*.png
    Saving 1594th image 9391415c040a502ff993370090588e28_*.png
    Saving 1595th image c3d2f3d0f46402a00d675867cdda4ae2_*.png
    Saving 1596th image 4a9f8667d7f339b84e9f401f0eb7a7a7_*.png
    Saving 1597th image ee5e85d3c1a1e3ec0a70305d7736543c_*.png
    Saving 1598th image 0cde3399886837f1d47be41f9f88b8ff_*.png
    Saving 1599th image 8fa251677255163b5e56ee957bf12dc3_*.png
    Saving 1600th image 7ece9d468e6863c32ab7873b7147fcfb_*.png
    Saving 1601th image ed32405ed20793b8a69e5e486c86edac_*.png
    Saving 1602th image 308948c1e725c6a74737842635fbf544_*.png
    Saving 1603th image dd95056c460ef7915f34ed95ee974a12_*.png
    Saving 1604th image a3e8b0df6dbaa58b6a261972c2387efe_*.png
    Saving 1605th image f5b8d83164d3d03687edad5ef1c12f67_*.png
    Saving 1606th image 71191c8b939ffc873beaaf130aca9c12_*.png
    Saving 1607th image 6cff023ce25dda602540da04d27e3740_*.png
    Saving 1608th image b691c67fe844f04429b12e990131858a_*.png
    Saving 1609th image 5f5009862089e540766ea2fc60b6caae_*.png
    Saving 1610th image 64890364dd640d6e899dc53445a10d0f_*.png
    Saving 1611th image 8fc3a6694e5302f9a79a80b1948ca51d_*.png
    Saving 1612th image c5579b77e7df1f7cb08285f2566abf8e_*.png
    Saving 1613th image 3c51533d1ad5695973611a9960d421e1_*.png
    Saving 1614th image 8b0c5065b6b55a20f7fc57e2b7758826_*.png
    Saving 1615th image ae375eaa9075d99bb06eb72468a87398_*.png
    Saving 1616th image db9d483eed93328b652cf61aa3f7769e_*.png
    Saving 1617th image 5bbea2f31629f5d6aefe9a3241393c7c_*.png
    Saving 1618th image 5f3c2d1ba1f0e6f1ef352b7b2402b70f_*.png
    Saving 1619th image 3cfef3448d5b733f9c173b8095fa7e7c_*.png
    Saving 1620th image 06cf09c0815b13eb741ca369c9a3db0e_*.png
    Saving 1621th image ac3e6f6b08753445fb4a5ca947917017_*.png
    Saving 1622th image 21f42f80bc752a7e7ef8f0bb9e2c9024_*.png
    Saving 1623th image bf60ff204b845ccde07bf5e8abd70af3_*.png
    Saving 1624th image 3303d2e6ad93ea241f7d93057037ad35_*.png
    Saving 1625th image c6eca33aa26b1f67d7ffd347f31aa45d_*.png
    Saving 1626th image 7fe2f88cc778f3f8d5f92e1d05c44196_*.png
    Saving 1627th image 8a1ab11f01d60931e460d5b48e31fdc3_*.png
    Saving 1628th image b8a10f5aca39f6dc068f54bc5106a054_*.png
    Saving 1629th image a6dd7f75ae0ddd646ae7f9cd371adc48_*.png
    Saving 1630th image 846a9910d9229f4c740557a22c3cb81f_*.png
    Saving 1631th image a69dbc0e4e8769896e0834d440c6380f_*.png
    Saving 1632th image d28c568d90c23bc1ca794cb9b38f25a6_*.png
    Saving 1633th image 36a9b354eac5a30b7e7cbb80a80b2db4_*.png
    Saving 1634th image 4fc7ddc00256ccc4438dbba06f2aff3d_*.png
    Saving 1635th image d3166407bb41435bf3c7133ed6e10465_*.png
    Saving 1636th image 0591e4e1ca05201f4fe2c1f7ae14fb83_*.png
    Saving 1637th image 473be080f2531d67d849744cd7c4251b_*.png
    Saving 1638th image ac5d7492f564a75a3c26959a0b5f2645_*.png
    Saving 1639th image f3dd975d35cc81e1d52d480655cde660_*.png
    Saving 1640th image 712d44f6b8a455baedefb08a5751ddda_*.png
    Saving 1641th image d3c7bdfdcb95ad549ca73a8214d2e7d4_*.png
    Saving 1642th image a8f3dff60f2ab8fa00756dd3c00ef58c_*.png
    Saving 1643th image e8c3b524d2964a0bd506544a629ba8db_*.png
    Saving 1644th image a5e59f7b5d891770e573d58a6138b7a1_*.png
    Saving 1645th image affad075ea3754ab429345d04ed79a6b_*.png
    Saving 1646th image f878ccc4a7266f97e4f89e5620f82e92_*.png
    Saving 1647th image b08e8eb6370297ae40d362fb6a11e7b9_*.png
    Saving 1648th image fd35f2a90840b287ce8e955d5f2f07ad_*.png
    Saving 1649th image f5fe28310b53ed209c06576c11b6b972_*.png
    Saving 1650th image 6667e8ccca8ee3eb1f8dcecc9eec09cb_*.png
    Saving 1651th image ed16bda7020a3da229ad8ed7145723b3_*.png
    Saving 1652th image ed21bf2f15896fd078ad07832b541245_*.png
    Saving 1653th image 3cf61cc058a3e1e2a25d9df2e8e5f6d8_*.png
    Saving 1654th image 528555723cf947d2330d4432136a6c0c_*.png
    Saving 1655th image edfc8ae295158970cf69abbeafaa0f1d_*.png
    Saving 1656th image 92b87c7b89a47364c33e0a35b5854ce2_*.png
    Saving 1657th image e07acaccae7a6649f44c48f79fbd4437_*.png
    Saving 1658th image 8e5c4267b053eb13efffb96e64dfc94c_*.png
    Saving 1659th image fd9f6d4b9001eb45219e69bc0ef5bb22_*.png
    Saving 1660th image 4443519506a02ea4e48c87f2f7a960da_*.png
    Saving 1661th image 6c8bb16d36d321516d23a70212fb2ea2_*.png
    Saving 1662th image ab864321c7d70bd94371db40b5ef5f87_*.png
    Saving 1663th image bc391046664c3ede589bd66fc0a523c6_*.png
    Saving 1664th image 41e7589010167518a93a0ebde6b2c065_*.png
    Saving 1665th image f785baf5b3e60414a7930847c7fe0ff5_*.png
    Saving 1666th image ef5a1c88713d0cdbd3cc4211f30e4a85_*.png
    Saving 1667th image 87a697d8185ea087f214baafdc12b3d6_*.png
    Saving 1668th image 6bbeb216ccbc8accc17b8e36fb4f35a4_*.png
    Saving 1669th image 4b66ee9b3c42d714a49216a84d442b7a_*.png
    Saving 1670th image 6fa291c88f231d6eea88064335e6a7d5_*.png
    Saving 1671th image ece6cc025cf2e31b2c643d6cfe41f9ec_*.png
    Saving 1672th image cd9ba265341131e52388423d99e88dd5_*.png
    Saving 1673th image 3262c6acc52acb8f0890b4c9ed09cd6b_*.png
    Saving 1674th image d264e7ebcb733da430000e07bb11e7d1_*.png
    Saving 1675th image eea4a020c3a9ecd5fe72c47afc283773_*.png
    Saving 1676th image 6f5bacedb80790c14f117d3985e29827_*.png
    Saving 1677th image 4d3c720a56e950a3cba0c0ac990cd6c5_*.png
    Saving 1678th image d6e3f88ba3f42d5b524b1304068628cd_*.png
    Saving 1679th image c555a9a58e1b4b656527a85fe2a87198_*.png
    Saving 1680th image 4c7d7801f058dc58cd09d966f5128dfa_*.png
    Saving 1681th image 3b3c06cf54cfababefd7673afd843724_*.png
    Saving 1682th image ec8c492ea1f5fd89130bfb72d031bd88_*.png
    Saving 1683th image 15e0467626d18776663ad8466168bfc3_*.png
    Saving 1684th image 00dfe72bf8f7af4b9dce75d8f64e3af5_*.png
    Saving 1685th image c2f70b750caec8d0371822129b228e93_*.png
    Saving 1686th image 61e0e1615644e0ff07e3c74701e5cc90_*.png
    Saving 1687th image 594d5a0a29e43d8472d7144038ca9f52_*.png
    Saving 1688th image 7db87d5a653de49d4b7de2d18db6e240_*.png
    Saving 1689th image 8832e6d1fba38f7d1f703437341086d7_*.png
    Saving 1690th image 5abb1f0882742e13c59113dcbd6d6dc9_*.png
    Saving 1691th image 1773f69b607b0376cf82c4e652a9c920_*.png
    Saving 1692th image 484487da79854d962c7d2eec1a9377c7_*.png
    Saving 1693th image 7902ac4447cad116286bdfa9824afd9d_*.png
    Saving 1694th image ee020be6dfbe9b15f1402980fce96f04_*.png
    Saving 1695th image 2369bae72c68eeaaaf7a00b79e55ce81_*.png
    Saving 1696th image d9c4e8c327d0e36350dac8e213d9b6d1_*.png
    Saving 1697th image 3b6f34cb0159f023cba0a0ad4b9f1c32_*.png
    Saving 1698th image a80ee8d221c2a55077aaf39d3e125abe_*.png
    Saving 1699th image 260390fc0c87b422b2065d7e1a2e0226_*.png
    Saving 1700th image 2939b7fe79daee7d9109c49c62e66504_*.png
    Saving 1701th image 562a849db4ec81d958739c27a72a5989_*.png
    Saving 1702th image 984999bd6cd4f7521a6d2e307487ed96_*.png
    Saving 1703th image b6ecbdae5151ab153b92f17968f4abc2_*.png
    Saving 1704th image 95a85dc1d0bf8588d8e4cad80c70bf83_*.png
    Saving 1705th image 2f102aedf7fe9073cf04f7d7e8f73bdc_*.png
    Saving 1706th image 792df539e695c683ba41474c3d80bcd0_*.png
    Saving 1707th image b0e1f103b62117ea3c026ff02cb934f4_*.png
    Saving 1708th image 67b9473d2d70916b6065cca479a4da8c_*.png
    Saving 1709th image 56d4dd37e77286f2a8ee1b0811339aad_*.png
    Saving 1710th image 494934ed2c59e621de821b52caf0e694_*.png
    Saving 1711th image 2420674032a5b2ff0107a9bf043b6f91_*.png
    Saving 1712th image 5e48943e09da88bf00eedefb674e76b1_*.png
    Saving 1713th image 5cef08761a6d50be4881db0fe2432a17_*.png
    Saving 1714th image b3c9ee066768206a7fd3bdc0fd8845e3_*.png
    Saving 1715th image 132ee297d468d2c556b06ce9f6ab6f43_*.png
    Saving 1716th image 89723f4b79b38ca40b930ae903f1c3f2_*.png
    Saving 1717th image cdc4c15a7b8d1881005d4b11bc3ede4f_*.png
    Saving 1718th image f6230112c500a507058c38382b0f536b_*.png
    Saving 1719th image d46277d566a4b0c20e76b97fd01ad5bf_*.png
    Saving 1720th image ec8e38f0afc56fe133d14d87d84cac1d_*.png
    Saving 1721th image 649d796c2bf662c87027b42c8f158605_*.png
    Saving 1722th image 7c9325eb6c287a7f1a42675e3b26f432_*.png
    Saving 1723th image cf57e965f7621372bd28e48bca338662_*.png
    Saving 1724th image 7c90a6e0c6dc1c2d45f2d3bf0b2f25d4_*.png
    Saving 1725th image 4a599c8c3000a993328d2ccb0b116c32_*.png
    Saving 1726th image 37c703178c6adb34aec2d5efbf191a20_*.png
    Saving 1727th image 9bbe78a3c6fe9e448784a7a85e08285f_*.png
    Saving 1728th image 884f36ee1dd6bedf711a776564f6ec85_*.png
    Saving 1729th image 2d736c6b616207168eefe0382eeae7c9_*.png
    Saving 1730th image 4cf12d75ad6ae8a95ca0bc6166e35f33_*.png
    Saving 1731th image a5ac469615c9499bd30456dcfce5f827_*.png
    Saving 1732th image 1b04b61dd64e46659f1042feead78d63_*.png
    Saving 1733th image 451215449007a0ba6a941474bdb585e0_*.png
    Saving 1734th image de19652c52291d3ede3464445f2da276_*.png
    Saving 1735th image c20a1d5a93bab79c0271addacfd3ba7c_*.png
    Saving 1736th image e6934d7f7d609498b8d315da8c35f8de_*.png
    Saving 1737th image 94fb78b2be07373fce3090dc20c3c640_*.png
    Saving 1738th image cac23a2c1a0f1958fc49a0d2b89312d1_*.png
    Saving 1739th image 8b93a4df934d633d108d24eeb6c86357_*.png
    Saving 1740th image d9dc0275acb72914e494d7bac1b400d2_*.png
    Saving 1741th image 584d244b210f11fe1a10c73beca796bd_*.png
    Saving 1742th image 202787f4a7a0ccfb43cc463b459bde1d_*.png
    Saving 1743th image eb56d7c8440ec2ba0ec95fb9838b5332_*.png
    Saving 1744th image b07cf078d1865732ba8fd88ec2dc195c_*.png
    Saving 1745th image 41020a82cb62ee00f4824770fb363ae6_*.png
    Saving 1746th image 1c40bdffa246c27f3b77a5189572f2b4_*.png
    Saving 1747th image eb794ecc58ac333a5e4534c4fe58b816_*.png
    Saving 1748th image 1354a5a9d0997dfed29db6bbe0750b7d_*.png
    Saving 1749th image aa2e9efa0e080525b7b20a40080fe263_*.png
    Saving 1750th image 22a4f636357b475581e6025e5f691f20_*.png
    Saving 1751th image e280782da89e0195a9c9ce8af7dc53f9_*.png
    Saving 1752th image 199eeb38cf069cfee244a7fefa7839ca_*.png
    Saving 1753th image 7e003954ea3afa7ed592032e9e94fd3d_*.png
    Saving 1754th image 1bd95237b9152aa80d0a5dd4594cbabb_*.png
    Saving 1755th image eb96fef0bd5dc4bc89f14ae05d5678e1_*.png
    Saving 1756th image 88efff2da1e565761c3b3d5325d08105_*.png
    Saving 1757th image fb2d0536b7fd9468c4bb804c8df3f033_*.png
    Saving 1758th image e412fb8f42acb552f1076a1ed4db206b_*.png
    Saving 1759th image 73fb3c6aed7ce28c0754448467a00aa4_*.png
    Saving 1760th image e2c85b9a6d347add15e522c66817efa9_*.png
    Saving 1761th image a52b7fcad09192e58ae7c1d5edbdfd18_*.png
    Saving 1762th image ed86b620abcce60b23ac8ce67097d1c9_*.png
    Saving 1763th image 5b66335f29411cc8567dd3bea8ff9065_*.png
    Saving 1764th image 90a470a047558eb278daad7885845138_*.png
    Saving 1765th image a380e24c7b44c888987e42e3ecb45f58_*.png
    Saving 1766th image aea44c04bf3359083a6a39f81301edff_*.png
    Saving 1767th image 8b476b5ba151f3e201daccb457b4946d_*.png
    Saving 1768th image 5ae8f48e7fecd8ce3f72bb661372110a_*.png
    Saving 1769th image 9c55db36b9e1f96a8952476d5d532096_*.png
    Saving 1770th image fb5f33fb4ec81626443702c48f2b939b_*.png
    Saving 1771th image 237a5d72c877070af70b9b9a2dcc0860_*.png
    Saving 1772th image da8598a7b253155a05ab38f5e85bd1f4_*.png
    Saving 1773th image b0bbd7c18dceacfedaa9984cd5b069a8_*.png
    Saving 1774th image b22db58965760d130d0053b975837c13_*.png
    Saving 1775th image 84ddb26d407649cff5e7c88a6591db4a_*.png
    Saving 1776th image c006422869572f0c40d682c878f3ebbf_*.png
    Saving 1777th image 5f3b372b80ae56029aefe7395be4f395_*.png
    Saving 1778th image ee50e13824e82cdbc4e2058e9ede2e49_*.png
    Saving 1779th image d6f82c7fae46314118726e92bf1993f8_*.png
    Saving 1780th image 1a07bbac32f7e9d0b8756881b9e6c376_*.png
    Saving 1781th image 78c18bbaa0399c1f6b198855edb94fd2_*.png
    Saving 1782th image f98a968141dfefb25b41779753cad87f_*.png
    Saving 1783th image a7d2111978c44fe45812a16f10a0ba11_*.png
    Saving 1784th image 18ab473966ef714ce316e37e8ab2ff01_*.png
    Saving 1785th image 8fbb2f488f310f36c9ebb63cc042f223_*.png
    Saving 1786th image 75a98800cee063d3367b15f5a1f7c4f6_*.png
    Saving 1787th image feb5c494b321161c90283a414065f567_*.png
    Saving 1788th image 9d858fe444ca6ae6a0136af4d4b55761_*.png
    Saving 1789th image b33429af5652ff2316415268d807db97_*.png
    Saving 1790th image 251a342df21b720421ad8725e8a4e79b_*.png
    Saving 1791th image b8a6fddb7e7bfe5d3de3be7564ed0e21_*.png
    Saving 1792th image 328de0e2a933ed37c4e147e0995c7b00_*.png
    Saving 1793th image 75a3f542223758e12c8929380803581e_*.png
    Saving 1794th image 24a064df3b2bd61999f04d3200f529b9_*.png
    Saving 1795th image 82210cda2ee56b28a579bab3c9aeb6fa_*.png
    Saving 1796th image 57b06b40b1b216e6392c64b7be118021_*.png
    Saving 1797th image f4db74312b15042f275b9779b9dd67f2_*.png
    Saving 1798th image df490f949b2bdb7e3b638e10c8fc73b7_*.png
    Saving 1799th image 91a74f6ac9f0930257a4948adba8ffb9_*.png
    Saving 1800th image 81c3c551668b966596095b9886e86e10_*.png
    Saving 1801th image cc36b373f188bd9a6569307442d8005a_*.png
    Saving 1802th image 7991422befb26de2e592674f5ff5b1f9_*.png
    Saving 1803th image 596c0a2902b7f4d1f14d9ebd852240e3_*.png
    Saving 1804th image 29af6e61e94db271ea8021addedb7b74_*.png
    Saving 1805th image 038d4a5a97f9f677d6771962d80bb183_*.png
    Saving 1806th image 6c2e1bbc7a04be5eb73297130e24071b_*.png
    Saving 1807th image 019bfdb566ec880ffda1c120d4451da4_*.png
    Saving 1808th image 5dd3d97cdc8215ec66868a5c9ebe5480_*.png
    Saving 1809th image 5f8e9b6c9f03e614fcbb85daeb84408f_*.png
    Saving 1810th image 6d6b2cdaf9c149c31677da91843c6c07_*.png
    Saving 1811th image fd650ce753eb217dfba2ca486e6e0b59_*.png
    Saving 1812th image b4f355f98dbec36f9f5068d8d357a2fa_*.png
    Saving 1813th image 7cf75ed5fcafc8be39d66b02826f77ff_*.png
    Saving 1814th image 4fa35fb0227df06da0858b7d1d073503_*.png
    Saving 1815th image 27392708ec38ad99f4c16b2e9cfc7d94_*.png
    Saving 1816th image 34da039cab3147f69968b5522b45feec_*.png
    Saving 1817th image 2291e8a5f4dd534bc6ba21eb75ff1de5_*.png
    Saving 1818th image 8bfa7e7ff1a5273ef92484484c08e9e6_*.png
    Saving 1819th image e09c37f327a3efc079498e7baf03772e_*.png
    Saving 1820th image 3acd56047ee0b36262caca3ccbb06602_*.png
    Saving 1821th image ad4bdbf36d7c429706364898500e1017_*.png
    Saving 1822th image 7922b7b23b44c38021aa8ffed2ace23c_*.png
    Saving 1823th image 8c781b8601697e31dfbe1565b53b02ca_*.png
    Saving 1824th image 7b854eea98f222a81a354b72ea26c35f_*.png
    Saving 1825th image 2ef20a2e954753a108cef545facf4756_*.png
    Saving 1826th image 1f0afde3e3e8ede5dedfb10bb3c0a360_*.png
    Saving 1827th image 562b0e5f21e42b5a8f83a8d705bd8620_*.png
    Saving 1828th image 9a42159d53937a88f18f0d6767eeaca3_*.png
    Saving 1829th image 370713f0cebdbab6afda4ffc4957b5c8_*.png
    Saving 1830th image 97110c1e788cc1dc7992a8640ffcfce1_*.png
    Saving 1831th image 32e83bc317c607ca4394c083ff5be852_*.png
    Saving 1832th image 5ffa45fd9e7d8785b1287f8e71e41a63_*.png
    Saving 1833th image 19b781f10472c67877d01582161835a2_*.png
    Saving 1834th image da23e7e74c96c59161dcba95f76645b0_*.png
    Saving 1835th image 4c704c8e27253d087b5c6e89e4a3f7a6_*.png
    Saving 1836th image 33ab3fc5fde1a0a1d215d232894499ba_*.png
    Saving 1837th image 98a1f252d955bead9a9e95611d14ba53_*.png
    Saving 1838th image 1373bd9b01af4da2bd4011bda522def8_*.png
    Saving 1839th image b65a675d4fd8a95d0c094afc9b1353c6_*.png
    Saving 1840th image 00c397728b4a60bdbe754b1a7be4bd17_*.png
    Saving 1841th image d7717113fd3b92eee964c8a519f30d4b_*.png
    Saving 1842th image 13bf351a80c6f6429e97ad19fb8c8b55_*.png
    Saving 1843th image d087bae28015ea0e4fcbedd769535ffb_*.png
    Saving 1844th image 71335c7d1f18607a5d2774a07eabc842_*.png
    Saving 1845th image c0d83fddeb3258bc652cf129aa2f9ad1_*.png
    Saving 1846th image f3d4593eeccbfcf4889b1dbfdbe30fbe_*.png
    Saving 1847th image 4a0bd80ddb3a6990d6545680928f8bc6_*.png
    Saving 1848th image d6e917f18c822147377e552302d2006c_*.png
    Saving 1849th image ae7f27d8fd57dcdefad3f5e365b3ae51_*.png
    Saving 1850th image 9650faf4644dc5ea35e93cfdcac659a1_*.png
    Saving 1851th image 6be60ec152d30313a1bb954e8834ecf8_*.png
    Saving 1852th image 1e01ca967059423f69f2d374e57d6088_*.png
    Saving 1853th image 795c12b777041206c77f8cd2f8fd7dd8_*.png
    Saving 1854th image 335719742bcfaa572bfcb2c7a79f2f28_*.png
    Saving 1855th image a422e0254c17ec9917789a7bae47fd57_*.png
    Saving 1856th image c1a1a65f711d45af866e79ff6c5e815d_*.png
    Saving 1857th image d18f94ea7aea2231e57f204d55fb1c20_*.png
    Saving 1858th image 280165c604cf36a0586504a8d55e895c_*.png
    Saving 1859th image 85903df5c47e01593a26c9819d8c15fd_*.png
    Saving 1860th image 8d867b3475f8c70c80dbfd162c68650d_*.png
    Saving 1861th image 42b7d184333dbe0fd49576777a3e2548_*.png
    Saving 1862th image 69e5f0c324bd28146dafebfe25747dbe_*.png
    Saving 1863th image 1c427a7d267b5d89ff9278614403c884_*.png
    Saving 1864th image 7a90b0bf64040a8605532ffa80dc8789_*.png
    Saving 1865th image 2b770bde3015dbf86566f5d9e50dcaa9_*.png
    Saving 1866th image 5b45c364421ee30d2778d6d8b4c74894_*.png
    Saving 1867th image a56a041070b5ed165e9bb3781c691005_*.png
    Saving 1868th image ed6389ed3ea5dbb42c3bbe675c685e25_*.png
    Saving 1869th image 854f59245546d68a354c6b8f7322e858_*.png
    Saving 1870th image 44c0c9a8fdd5c603c5f516110d83c242_*.png
    Saving 1871th image 822c72cd3e67ba838249e6a126fe2044_*.png
    Saving 1872th image cbfbe799e6974439b3d3e812e4abcea9_*.png
    Saving 1873th image 27414e79458fbe8356cfcb672a84b424_*.png
    Saving 1874th image ff8dab55ee27975f1115d908bd17805e_*.png
    Saving 1875th image 4ec28402d3529a49e046f316827a51ea_*.png
    Saving 1876th image cd805e465215aef5628f06c7e8ce64ab_*.png
    Saving 1877th image af870dc85241a334c2228de86951d5d6_*.png


### Results

Selecting 4 Random results in output and displaying.

Order of images:

    Input (gray scale), Predicted Color, Ground Truth


```python
# Select random 4 results to display.
# Order: Input, Predicted, Ground Truth
# Assumes all three results folder have the 4 files

filelist = shuffle(os.listdir('result/predicted/'))
filelist = filelist[:4]

fig, ax = plt.subplots(4, 3, figsize=(16,16))
row = 0
for filename in filelist:
    folder = 'result/bnw/'
    image_in = img_to_array(load_img(folder + filename))
    image_in = image_in/255
    ax[row,0].imshow(image_in)
    
    folder = 'result/predicted/'
    image_in = img_to_array(load_img(folder + filename))
    image_in = image_in/255
    ax[row,1].imshow(image_in)
    
    folder = 'result/actual/'
    image_in = img_to_array(load_img(folder + filename))
    image_in = image_in/255
    ax[row,2].imshow(image_in)
    
    row += 1
```


![png](output_34_0.png)


### Learning Graphs

|Loss|Learning Rate|Validation Loss|
|:----:|:----:|:----:|
| <img src="resources/loss.png" style="height:150px; vertical-align:bottom; display:inline-block;"/> | <img src="resources/lr.png" style="height:150px; vertical-align:bottom; display:inline-block;"/> | <img src="resources/val_loss.png" style="height:150px; vertical-align:bottom; display:inline-block;"/> |

### Future Work

* Above network is made to run on a very small dataset due to time and resource contraints, In future, we will train on a larger dataset and compare result
* As seen in the Learning graphs, this network reaches learning saturation with quite a few training samples and doesn't generalize well just yet. We will do further enhancements to improve this situation.
* Generalize network input size so as to enable it to train on multiple resolution images
* Experiment using other pretrained models
* Can be extended to videos

### References

1. *Baldassarre, Morin and Lucas Rodes-Guirao.* __Deep Koalarization: Image Colorization using CNNs and Inception-Resnet-v2.__ [link](https://github.com/baldassarreFe/deep-koalarization/blob/master/paper.pdf)
2. *Iizuka, Satoshi, Edgar Simo-Serra, and Ishikawa Hiroshi.* __Let there be Color!__ *SIGGRAPH 2016*
3. *Larsson, Maire, and Shakhnarovich.* __Learning Representations for Automatic Colorization.__ *ECCV 2016*
4. *Zhang, Richard, Phillip Isola, and Alexei A. Efros.* __Colorful Image Colorization.__ *ECCV 2016*
5. http://www.whatimade.today/our-frst-reddit-bot-coloring-b-2/
