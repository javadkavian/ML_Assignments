import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

# Define your data points
class1_x = [0, 0.5, 1, 1, 1.5, 1.5, 2, 2, 2.5]
class1_y = [0.5, 2, 1, 3, 0, 2, 1, 3, 2]

class2_x = [-2, -1.5, -1.5, -1, -0.5, 0.5, 0.5, 1, 1.5, 1.5]
class2_y = [-1, 0, 1, -1, 0.5, 0.5, -0.5, -1, -0.5, 0.5]

# Combine data points
X = np.array(list(zip(class1_x + class2_x, class1_y + class2_y)))
y = np.array([1] * len(class1_x) + [2] * len(class2_x))  # Class labels (1 or 2)

# Define the risk matrix
a = 0.5
risk_matrix = np.array([[0, 2 * a], [a, 0]])

# Calculate the posterior probabilities with adjusted priors
clf = GaussianNB(priors=[0.66, 0.34])
clf.fit(X, y)

# Compute the decision boundary
def predict_min_risk(x):
    posterior_probs = clf.predict_proba(x)
    risks = np.dot(posterior_probs, risk_matrix)
    return np.argmin(risks, axis=1) + 1

# Plot the decision boundary
def plot_decision_boundary(predict_func):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = predict_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)

# Plot the decision boundary for minimum risk Bayes
plot_decision_boundary(predict_min_risk)
plt.scatter(class1_x, class1_y, color='blue', label='Class 1')
plt.scatter(class2_x, class2_y, color='red', label='Class 2')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.title('Minimum Risk Decision Boundary')
plt.legend()
plt.show()
