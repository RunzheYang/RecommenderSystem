import csv
import recsys.algorithm
from recsys.algorithm.factorize import SVD
from recsys.datamodel.data import Data

def loadTest(dir):
	test = []
	cnt = 0
	print "Start loading test data..."
	with open(dir) as file:
		for line in file:
			if cnt % 10000 == 9999: 
				print "%d / 1524458 loaded\r"%(cnt+1),
			cnt += 1
			# if cnt == 100000: break
			(user, item, week, time, feat1, feat2)=line.split('\t')
			test.append(
				{"1_user_id": int(user),
				 "2_item_id": int(item)
				})		
	return test

recsys.algorithm.VERBOSE = True
print "loading data"
data = Data()
data.load('../item_recom/train_info.tsv',sep='\t', format={'col':0, 'row':1, 'value':6, 'ids': int})

topic = 48
print "compute svd"
svd = SVD()
svd.set_data(data)
svd.compute(k=topic, min_values=0.0, pre_normalize=None, mean_center=True, post_normalize=True)

print "loading test data"
test = loadTest('../item_recom/test_info.tsv')

print svd.predict(0,0)

print "creating submission"
with open('../submissions/recsys_3.csv', 'w') as csvfile:
	fieldnames = ['uid#iid', 'pred']
	writer = csv.DictWriter(csvfile, fieldnames)
	writer.writeheader()
	for ind in xrange(len(test)):
		writer.writerow(
			{
				'uid#iid': "%d#%d"%(test[ind]["1_user_id"], test[ind]["2_item_id"]),
				'pred': svd.predict(
					test[ind]["2_item_id"], 
					test[ind]["1_user_id"])
			})
