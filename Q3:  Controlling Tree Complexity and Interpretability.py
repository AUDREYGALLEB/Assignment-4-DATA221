'''
Q3 — Controlling Tree Complexity and Interpretability (10 points)
Unconstrained decision trees can easily overfit training data.
Modify the Decision Tree model by introducing at least one constraint (e.g., max depth, min samples split,
or a similar parameter):
• Train the constrained model and report training and test accuracy.
• Display the top five most important features according to the model.
In Python comments, briefly discuss:
• how controlling model complexity affects overfitting, and
• how feature importance contributes to the interpretability of decision trees.
'''
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target

# training and tree to avoid more overfitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
the_model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
the_model.fit(X_train, y_train)
#predictions and accuracies
y_train_prediction = the_model.predict(X_train)
y_test_prediction = the_model.predict(X_test)
train_accuracy = accuracy_score(y_train, y_train_prediction)
test_accuracy = accuracy_score(y_test, y_test_prediction)
print('train accuracy: ', train_accuracy)
print('test accuracy: ', test_accuracy)

#features importance (top 5)
importances = the_model.feature_importances_
top_5_indices = np.argsort(importances)[::-1]
print("top 5 incidces: ")
for i in range(5):
    index = top_5_indices[i]
    print(data.feature_names[i], ": ", importances[index])

'''
- it affects it by limiting tree depth and reduces overfitting. since the train and test accuracy are very close, it can
be considered as good generalization.

- it contributes by showing which feature contributes the most, making it so that if one feature is more dominant
it makes it more reliant on interpreting the model more than any other features.
'''