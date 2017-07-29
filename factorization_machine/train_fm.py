import math
import csv
import numpy as np
import cPickle as pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from pyfm import pylibfm
from data_loader import loadData
from data_loader import loadTest

(train, y_train), (valid, y_valid) = loadData("../item_recom/train_info.tsv", 1.0)
dictv = DictVectorizer()
print "convert to one-hot represerntation"
X_train = dictv.fit_transform(train)
# X_valid = dictv.transform(valid)

fm = pylibfm.FM(
		num_factors=64, 
		num_iter=10, 
		verbose=True, 
		task="regression", 
		initial_learning_rate=0.001, 
		learning_rate_schedule="optimal")

# fm = pickle.load(open( "models/fm_64.pickle", "rb" ))  

print "Start training"
fm.fit(X_train, y_train)

pickle.dump(fm, open( "models/fm_64.pickle", "wb" ), -1)

# preds = fm.predict(X_valid)
# print("FM RMSE: %.6f" % math.sqrt(mean_squared_error(y_valid, preds)))

test = loadTest("../item_recom/test_info.tsv")
X_test = dictv.transform(test)
test_preds = fm.predict(X_test)

with open('../submissions/second.csv', 'w') as csvfile:
	fieldnames = ['uid#iid', 'pred']
	writer = csv.DictWriter(csvfile, fieldnames)
	writer.writeheader()
	for ind in xrange(len(test)):
		writer.writerow({'uid#iid': "%s#%s"%(test[ind]["user_id"], test[ind]["item_id"]) ,'pred': test_preds[ind]})

# pickle.dump(dictv, open( "models/fm_64_dictv.pickle", "wb" ), -1)
