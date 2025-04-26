'''
Name: Daniel Nguyen
Date: April 23, 2025
Purpose: Build a simple neural network model using Keras to predict house prices based on randomly generated features. 
This exercise is based on the tutorial from Joseph Lee on Intuitive Deep Learning.
'''

# Import necessary libraries
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# Generate dummy data
x_train = np.random.random((1000, 10))                  # 1000 training samples with 10 features each
y_train = np.random.random((1000, 1))                   # 1000 target prices
x_test = np.random.random((100, 10))                    # 100 test samples
y_test = np.random.random((100, 1))                     # 100 target prices

# Build the model
model = Sequential()
model.add(Dense(64, input_dim=10, activation='relu'))   # Hidden layer with 64 units
model.add(Dense(1, activation='linear'))                # Output layer with 1 unit (predict price)

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Evaluate the model
score = model.evaluate(x_test, y_test, batch_size=32)
print('Test loss:', score)
