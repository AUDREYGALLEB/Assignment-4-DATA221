'''
Q4 — Neural Network for Binary Classification (10 points)
Neural networks approach learning differently, relying on optimization rather than rule-based splitting.
Using the same training and testing data:
• Standardize the input features.
• Train a neural network with at least one hidden layer and a sigmoid output unit.
• Report training accuracy and test accuracy.
In Python comments, explain:
• why feature scaling is necessary for neural networks, and
• what an epoch represents during neural network training.
'''
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#load dataset
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target

# train and standardize features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

# neural networks
# 1 with 10neurons
# and sigmoid output unit
model = MLPClassifier(hidden_layer_sizes=(10,), activation='logistic', max_iter=1000, random_state=42)
model.fit(scaled_X_train, y_train)

# predictions and accuracies
y_train_prediction = model.predict(scaled_X_train)
y_test_prediction = model.predict(scaled_X_test)
train_accuracy = accuracy_score(y_train, y_train_prediction)
test_accuracy = accuracy_score(y_test, y_test_prediction)
print('train accuracy:', train_accuracy)
print('test accuracy:', test_accuracy)

'''
- it is necessary for neural networks becuase inputs with varying ranges can slow down the learning. standardizing ensures that 
all features contribute equally and fastens optimization.

- epoch represents one complete training of data through networks.
'''