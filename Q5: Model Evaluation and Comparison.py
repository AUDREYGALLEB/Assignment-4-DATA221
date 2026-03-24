'''
Q5 — Model Evaluation and Comparison (5 points)
Different models may perform similarly in terms of accuracy, yet behave very differently in practice.
For the constrained Decision Tree and the Neural Network:
• Compute and display the confusion matrix for each model.
In Python comments, provide a concise comparison addressing:
• which model you would prefer for this task, and
• one advantage and one limitation of each model.
'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# loading dataset
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target

#train with the 80/20 and tree
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
the_model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
the_model.fit(X_train, y_train)
y_test_prediction = the_model.predict(X_test)

#neural networks and standardize
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)
model = MLPClassifier(hidden_layer_sizes=(10,), activation='logistic', max_iter=1000, random_state=42)
model.fit(scaled_X_train, y_train)
y_train_prediction = model.predict(scaled_X_train)

#confusion matrices and tree
#tree
confusion_matrix_tree = confusion_matrix(y_test, y_test_prediction)
print(confusion_matrix_tree)
decision_tree_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_tree, display_labels=data.target_names)
decision_tree_display.plot()
plt.show()
#neural network
confusion_matrix_neural = confusion_matrix(y_test, y_test_prediction)
print(confusion_matrix_neural)
neural_network_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_neural, display_labels=data.target_names)
neural_network_display.plot()
plt.show()

'''
i would prefer the decision tree since it is simpler to interpret and is more acceptable in these classes which are breast cancer.

decision tree:
- pros: easy to interpret, able to show which features has the most influence,.
- cons: can overfit and prone to changes within data.
neural network:
- pros: can model complex models
- cons: harder to interpret (blackbox)
'''