'''
Q6 — Convolutional Neural Network with Built-in Dataset (10
points)
Convolutional Neural Networks (CNNs) are widely used for image classification because they can
automatically learn spatial patterns from pixel data. In this question, you will train a basic CNN
using a built-in TensorFlow dataset.
Use the Fashion MNIST dataset available in TensorFlow.
Tasks:
• Load the Fashion MNIST dataset.
• Normalize the pixel values to the range [0,1].
• Reshape the images to include the channel dimension.
• Build a CNN that includes at least:
– one Conv2D layer,
– one MaxPooling2D layer,
– one Dense output layer.
• Train the model for at least 15 epochs.
• Report the test accuracy.
In Python comments, briefly explain:
• why CNNs are generally preferred over fully connected networks for image data, and
• what the convolution layer is learning in this task.
'''
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.datasets import fashion_mnist

# loadng dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# normalizing pixel values to range [0,1] and reshaping images to include channel dimension
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

#CNN that has Conv2d, MaxPooling2D, and Dense
model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# getting model and training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=15, batch_size=64, validation_split=0.1)
test_loss, test_accuaracy = model.evaluate(X_test, y_test)
print('test accuracy:', test_accuaracy)

'''
- CNN are preferred for image data because they capture patterns, edges, and shapes.

- the convulation layer is learning filters that can detect features like patterns, edges and clothing.
'''