'''
Q7 — CNN Error Analysis and Misclassification Study (10 points)
In practical machine learning systems, analyzing model errors is just as important as reporting
accuracy.
Using your trained CNN from Question 6:
Tasks:
• Generate predictions on the test set.
• Compute and display the confusion matrix.
• Identify and visualize at least three misclassified images.
• For each displayed image, clearly show:
– the true label,
– the predicted label.
In Python comments, briefly discuss:
• one pattern you observe in the misclassifications, and
• one realistic method to improve the CNN performance.
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
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
    Dense(10, activation='softmax')])

# getting model and training with predictions
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=15, batch_size=64, validation_split=0.1)
y_predictions_prob = model.predict(X_test)
y_prediction = np.argmax(y_predictions_prob, axis=1)

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_prediction)

plt.figure(figsize=(8,6))
plt.imshow(conf_matrix, cmap='Blues')
plt.colorbar()
plt.title('confusion matrix')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()

# 3 misclassified images
misclassified_indices = np.where(y_prediction != y_test)[0]
for i in range(3):
    indx = misclassified_indices[i]
    plt.imshow(X_test[indx].reshape(28,28), cmap='gray')
    plt.title(f"true: {y_test[indx]}, predicted: {y_prediction[indx]}")
    plt.axis('off')
    plt.show()
'''
- i observed that it usually occurs when there are cisually similar classes, for example the two pairs of shoes.

- one realistic method to improve the CNN performance is to add more filters or convolutional layers, like brightness and rotations.
'''