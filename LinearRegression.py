import numpy as np
import random
from statistics import mean
from sklearn import linear_model


# Again, this is based on sentdex's tutorials, I just made it a class and made some slight changes.
# the create_dataset() is taken from him verbatim.
# his tutorial:
# https://www.youtube.com/watch?v=OGxgnH8y2NM&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v&index=1
# the first 13 videos in the playlist


class LinearRegression(object): # if the data isn't linear, don't use linear regression

	def __init__(self, **kwargs):
		self.m = 0.0
		self.b = 0.0

	def fit(self, X, y):

		# take in the X values, and the y values, sets m & b

		Xy = [X[i] * y[i] for i in xrange(len(X))] # Just gets the x * y list 
		xSqaure = [_**2 for _ in X] # the x^2 list 
		self.m = ((mean(X) * mean(y) - mean(Xy))  / ( mean(X)**2 - mean(xSqaure) )) # calculates m 
		self.b = mean(y) - self.m*mean(X) # calculates b

		return [(self.m*x) + self.b for x in X] # just returns the line of the model y = mx + b for each x


	def predict(self, X):
		return [self.m * x + self.b for x in X]

	def getRsquared(self, y, yhat):

		y_mean_line = [mean(y) for _ in y]
		squared_error_reg = [(yhat[i] - y[i])**2 for i in xrange(len(yhat))]
		squared_error_reg = sum(squared_error_reg)
		squared_error_y_mean = [(y[i] - y_mean_line[i])**2 for i in xrange(len(y))]
		squared_error_y_mean = sum(squared_error_y_mean)

		return 1 - (squared_error_reg / squared_error_y_mean)
		# want the r^2 value to be high 


def create_dataset(hm, variance, step=2, correlation=False):

	val = 1
	ys = []

	for i in xrange(hm):
		y = val + random.randrange(-variance, variance)
		ys.append(y)
		if correlation and correlation == 'pos':
			val+=step
		elif correlation and correlation == 'neg':
			val -= step

	xs = [i for i in xrange(len(ys))]
	return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)



xs, ys = create_dataset(40, 40, 2, correlation='neg')

scikit_linear = linear_model.LinearRegression()
class_linear = LinearRegression()

xs_ = [x for x in xs]
ys_ = [y for y in ys]


class_line = class_linear.fit(xs_, ys_)
scikit_line = scikit_linear.fit(xs.reshape(-1, 1), ys.reshape(-1,1))


l = [8, 9, 42]
a = [[8], [9], [42]]


print class_linear.predict(l)
for x in scikit_line.predict(a):
	print x, 


# print scikit_line.score(xs.reshape(-1, 1), ys.reshape(-1,1)), class_linear.getRsquared(ys_,class_line)





