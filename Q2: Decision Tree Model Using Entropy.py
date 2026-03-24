'''
Q2 — Decision Tree Model Using Entropy (10 points)
Decision trees aim to reduce uncertainty in the target variable by selecting splits that maximize
information gain.
Using an 80/20 train–test split with stratification:
• Train a Decision Tree classifier using entropy as the splitting criterion.
• Report the training accuracy and test accuracy of the model.
In Python comments, explain:
• what entropy represents in the context of decision trees, and
• whether the observed results suggest overfitting or good generalization.
'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#import dataset
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target

# training with 80/20 strat
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
the_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
the_model.fit(X_train, y_train)

# making my predictions and accuracies
trained_y_prediction = the_model.predict(X_train)
test_y_prediction = the_model.predict(X_test)
train_accuracy = accuracy_score(y_train, trained_y_prediction)
test_accuracy = accuracy_score(y_test, test_y_prediction)

print('train accuracy:', train_accuracy)
print('test accuracy:', test_accuracy)

'''
- entropy in decision trees represents the mixed classes in a node. low entropy indicates a node with one class while high entropy 
indicates a node with mixed classes. with this model it will just split the data reducing entropy, which is information gain.

- from what i am getting, training accuracy is slightly higher than test accuracy making it overfitting.
'''