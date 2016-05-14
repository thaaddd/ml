# taken from:
# http://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html
# comments are just to help me stupid my way through it

from sklearn import datasets, neighbors, linear_model

# gets the dataset, splits the data and the target
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

n_samples = len(X_digits)

# splits the data into training and testing 
X_train = X_digits[:.9 * n_samples]
y_train = y_digits[:.9 * n_samples]
X_test = X_digits[.9 * n_samples:]
y_test = y_digits[.9 * n_samples:]

# declares the estimators
knn = neighbors.KNeighborsClassifier()
logistic = linear_model.LogisticRegression()

# fits the training data
k = knn.fit(X_train, y_train)
lgr = logistic.fit(X_train, y_train)


# returns the score of the model: 
# number of correctly identified samples / samples
print k.score(X_test, y_test)
print lgr.score(X_test, y_test)

