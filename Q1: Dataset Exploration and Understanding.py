'''
Q1 — Dataset Exploration and Understanding (5 points)
A correct understanding of the dataset is essential before applying any machine learning model.
Using the provided dataset:
• Construct the feature matrix X and target vector y.
• Report the shape of X and y.
• Report the number of samples belonging to each class.
In Python comments, provide a brief discussion addressing:
• whether the dataset is balanced or imbalanced, and
• why class balance is an important consideration for classification models.
'''
# loading the dataset
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

# feature matrix X and the target vector of y
# showing the shape too
X = data.data
y = data.target
print("shape of X - the features: ", X.shape)
print("shape of y - the target: ", y.shape)

# occurrences of each class
malignant_count = 0
benign_count = 0
for value in y:
    if value == 0:
        malignant_count += 1
    else:
        benign_count += 1
print("malignant count: ", malignant_count)
print("benign count: ", benign_count)
'''
- the dataset is imbalanced since there are more benign samples.

- it is important because if it is imbalanced, and leans towards one class more than the other, it will be more
biased in predicting the dominant class instead. this can be misleading and tamper with accuracy.
'''