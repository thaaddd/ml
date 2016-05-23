# not the most beautiful work

# learned all of this from the scikit-learn site:
# http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
# and sentdex's tutorials on youtube

# you should check them both out
# this is my first attempt on my own (albeit borrowed heavily from scikit and sentdex)
# so if anyone does actually look at this please let me know if something looks off 
# thanks!

import numpy as np

from nltk.corpus import twitter_samples
from sklearn import cross_validation
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier

pos = twitter_samples.strings('positive_tweets.json')
neg = twitter_samples.strings('negative_tweets.json')

tweets = []

for x in neg:
	tweets.append((x, 'Negative')) 

for x in pos:
	tweets.append((x, 'Positive'))

X = dict(tweets).keys() # the tweets
y = np.array(dict(tweets).values()) # 0,1s

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=.25)
# splits into training and testing sets 

clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier()),])
# CountVectorizer() -> Handles text preprocessing, tokenizing, and filtering of stop words
# -> builds a dictionary of features (feature vectors) 

# TfidfTransformer() -> Handles the jump from occurrences to frequencies
# -> Term Frequency times inverse document frequency

# SGDClassifier() -> Trains the classifer, put in any classification algorithm, however

# these steps can be done seperately, but the Pipeline really streamlines it

clf = clf.fit(X_train, y_train) # training

accuracy = clf.score(X_test, y_test) # testing

print "Accuracy: ", accuracy

new = ['I dont care anymore forget the haters :(', 'God is love', 'I cant beleive Kim died']

predicted = clf.predict(new) # predicting

print; print 'Predicted:'

for x, y in zip(new, predicted):
	print x, '->' ,y






