``` python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import categorical_crossentropy
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
%matplotlib inline
```

``` python
# go inside dogs-and-cats folder. Change it to the name of your dataset folders.
os.chdir('./dogs-and-cats')

#create 3 folders for train, test, and validation
if os.path.isdir('train/dog') is False:
    os.makedirs('train/dog')
    os.makedirs('train/cat')
    os.makedirs('test/dog')
    os.makedirs('test/cat')
    os.makedirs('valid/dog')
    os.makedirs('valid/cat')
    # I will use 500 images for testing and training and 100 images for validation
    for c in random.sample(glob.glob('cat*'), 500):
        shutil.move(c, 'train/cat')
    for c in random.sample(glob.glob('dog*'), 500):
        shutil.move(c, 'train/dog')
    for c in random.sample(glob.glob('cat*'), 100):
        shutil.move(c, 'valid/cat')
    for c in random.sample(glob.glob('dog*'), 100):
        shutil.move(c, 'valid/dog')
    for c in random.sample(glob.glob('cat*'), 500):
        shutil.move(c, 'test/cat')
    for c in random.sample(glob.glob('dog*'), 500):
        shutil.move(c, 'test/dog')
    
```

``` python
# go back to the main directory
os.chdir('..')
```

``` python
train_path = './dogs-and-cats/train'
valid_path = './dogs-and-cats/valid'
test_path = './dogs-and-cats/test'
```

``` python
# preprocess your data using VGG16
# Our target size is 224 by 224, 
# classes are dogs and cats because this is just a binary classification problem
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=train_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=valid_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=test_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=10, shuffle=False)
```

``` python
# we get the next batch and try to visualize it

imgs, labels = next(train_batches)

def plotImages(images_arr, labels_arr):
    fig, axes = plt.subplots(1,10, figsize=(20,20))
    axes = axes.flatten()
    index = 0
    label_names = ['Cats', 'Dogs']
    for img, ax in zip (images_arr, axes):
        ax.imshow(img.astype(np.uint8))
        ax.set_xlabel(label_names[np.argmax(labels_arr[index])])
        index += 1
    plt.tight_layout()
    plt.show()
    
plotImages(imgs, labels)

# you will see that the images are already preprocessed
```

``` python
# now will create are simple conv2d model
model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',input_shape=(224,224,3)),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2)),
    Flatten(),
    Dense(2, activation='softmax')
])

model.summary()
```

``` python
# we will compile it using Adam as optimizer and categorical_crossentropy for our loss function
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

``` python
# then we will train it for 10 epochs
history = model.fit(x=train_batches, validation_data=valid_batches, epochs=10, verbose=2)
```

``` python
# we will plot are models to see how our loss and validation loss perform
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

``` python
# our model does overfit so bad, we will fix it later
# for now we will make our prediction and see how this model performs
predictions = model.predict(x=test_batches, verbose=0)
```

``` python
from sklearn.metrics import classification_report
print(classification_report(test_batches.classes,np.argmax(predictions, axis=-1)))
```

``` python
# we have achieve 51% accuracy for our simple model
# lets add additional Conv2d layer and additional dense layer
model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(224,224,3)),
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

model.summary()
```

``` python
# let's compile it again
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(x=train_batches, validation_data=valid_batches, epochs=5, verbose=2)
```

``` python
# we will plot again are results to see how our loss and validation loss perform
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

``` python
# We make a slight improvement vs our previous model
predictions = model.predict(x=test_batches, verbose=0)
print(classification_report(test_batches.classes,np.argmax(predictions, axis=-1)))
```

``` python
# now we will use a pretrained model called VGG16 for our dogs and cats image classification problem
vgg16_model = tf.keras.applications.vgg16.VGG16()
vgg16_model.summary()
```

``` python
# we will add our vgg16 model to our model except the last layer
model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)
    
model.summary()
```

``` python
# we will set trainable to False because it is a pretrained model
# we don't want to train it again
for layer in model.layers:
    layer.trainable= False
```

``` python
# add 2 units dense layer for our last layer
model.add(Dense(2, activation='softmax'))
```

``` python
# Let's compile and train it
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_batches, validation_data=valid_batches, epochs=5, verbose=2)
```

``` python
# lets predict our test data and see how it performs
predictions = model.predict(x=test_batches, verbose=0)
print(classification_report(test_batches.classes,np.argmax(predictions, axis=-1)))
```

``` python
# We already got the best result which is an accuracy of 98%
# But if for some reason we got did not get what we want
# Our next option is to fine tune the model by unfreezing layers and let it train on our current dataset
```
