'''
Name: Daniel Nguyen
Date: April 23, 2025
Program: CIFAR-10 Image Classification using Convolutional Neural Network (CNN)
Description:
This program builds and trains a Convolutional Neural Network (CNN) using the CIFAR-10 image dataset.
It uses Keras and TensorFlow to classify images into 10 different classes. The model architecture includes 
multiple convolutional layers, dropout for regularization, and fully connected dense layers.

The dataset used is the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, 
with 6,000 images per class.

The CNN is trained using categorical cross-entropy loss and evaluated with accuracy metrics. 
Model performance is visualized using matplotlib.

Acknowledgement:
This program is inspired and based on the tutorial by Joseph Lee:
https://medium.com/intuitive-deep-learning/build-your-first-convolutional-neural-network-to-recognize-images-84b9c78fe0ce

All credits for the instructional content and structure go to the author, Joseph Lee.
'''

from tensorflow.keras.datasets import cifar10

from keras.datasets import cifar10
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Display the first image in the training set
img = plt.imshow(x_train[1])

plt.show()

print("The label is: ", y_train[1])

import keras
y_train_one_hot = keras.utils.to_categorical(y_train, 10)
y_test_one_hot = keras.utils.to_categorical(y_test, 10)

print("The one-hot encoded label is: ", y_train_one_hot[1])

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

model = Sequential()

# This code initializes a Sequential model and adds a 2D convolutional layer with 32 filters,
# a kernel size of (3, 3), ReLU activation, and same padding. The input shape is set to (32, 32, 3), 
# which corresponds to the dimensions of the CIFAR-10 images.    
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(Dropout(0.25)) # Prevents overfitting by randomly setting a fraction of input units to 0 during training

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2))) # Reduces the spatial dimensions of the output volume
model.add(Dropout(0.25)) # Prevents overfitting by randomly setting a fraction of input units to 0 during training

model.add(Flatten()) # Flattens the input, i.e., converts it to a 1D array
model.add(Dense(512, activation='relu')) # Fully connected layer with 512 units and ReLU activation
model.add(Dropout(0.5)) # Prevents overfitting by randomly setting a fraction of input units to 0 during training
model.add(Dense(10, activation='softmax')) # Output layer with 10 units (one for each class) and softmax activation
model.summary() # Prints a summary of the model's architecture

# Compile the model with categorical crossentropy loss, Adam optimizer, and accuracy metric
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(x_train, y_train_one_hot, batch_size=32, epochs=20, validation_split=0.2) # Train the model on the training data

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

# Evaluate the model on the test data
model.evaluate(x_test, y_test_one_hot)[1]

# Save the model to a file
model.save('my_cifar10_model.keras')

# Load the model from a file
from keras.models import load_model
model = load_model('my_cifar10_model.h5')

my_image = plt.imread("cat.jpg") # Load an image

from skimage.transform import resize
my_image_resized = resize(my_image, (32, 32, 3)) # Resize the image to (32, 32, 3)

img = plt.imshow(my_image_resized) # Display the image

import numpy as np
probabilities = model.predict(np.array([my_image_resized])) # Predict the class probabilities
print(probabilities) # Print the class probabilities

number_to_class = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
                     5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'} # Mapping of class numbers to class names
index = np.argsort(probabilities[0,:])
print("Most likely class:", number_to_class[index[9]], "-- Probability:", probabilities[0,index[9]])
print("Second most likely class:", number_to_class[index[8]], "-- Probability:", probabilities[0,index[8]])
print("Third most likely class:", number_to_class[index[7]], "-- Probability:", probabilities[0,index[7]])
print("Fourth most likely class:", number_to_class[index[6]], "-- Probability:", probabilities[0,index[6]])
print("Fifth most likely class:", number_to_class[index[5]], "-- Probability:", probabilities[0,index[5]])

