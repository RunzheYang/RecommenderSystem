import math
import pywFM
import numpy as np
import pandas as pd
import csv
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from data_loader import loadData
from data_loader import loadTest

(train, y_train), (valid, y_valid) = loadData("../item_recom/train_info.tsv", 1.1)

dictv = DictVectorizer()

test = loadTest("../item_recom/test_info.tsv")

print "convert to one-hot represerntation"
_ = dictv.fit_transform(train+test)
X_train = dictv.transform(train)
# X_valid = dictv.transform(valid)
X_test = dictv.transform(test)
y_test = np.ones(len(test))*2.5

fm = pywFM.FM(task='regression', num_iter=1200, k2=48, learning_method='mcmc')

model = fm.run(X_train, y_train, X_test, y_test)
# print("FM RMSE: %.6f" % math.sqrt(mean_squared_error(y_valid, model.predictions)))
# 
with open('../submissions/sixteenth.csv', 'w') as csvfile:
	fieldnames = ['uid#iid', 'pred']
	writer = csv.DictWriter(csvfile, fieldnames)
	writer.writeheader()
	for ind in xrange(len(test)):
		writer.writerow({'uid#iid': "%s#%s"%(test[ind]["1_user_id"], test[ind]["2_item_id"]) ,'pred': "%f"%model.predictions[ind]})