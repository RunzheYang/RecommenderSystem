import numpy as np

def getUserWatchedCnt(dir="../item_recom/train_info.tsv"):
	stats_user = [0]*94317
	stats_item = [0]*99782
	with open(dir) as file:
		for line in file:
			(user, item, week, time, feat1, feat2, rating)=line.split('\t')
			stats_user[int(user)] += 1
			stats_item[int(item)] += 1
	return stats_user, stats_item

def loadData(dir, train_ratio=0.9):
	train, train_rating = [], []
	valid, valid_rating = [], []
	cnt = 0
	print "Start loading data..."
	stats_user, stats_item = getUserWatchedCnt(dir)
	with open(dir) as file:
		for line in file:
			if cnt % 10000 == 9999: 
				print "%d / 5974450 loaded\r"%(cnt+1),
			cnt += 1
			# if cnt == 100000: break
			(user, item, week, time, feat1, feat2, rating)=line.split('\t')
			if np.random.rand() < train_ratio:
				train.append(
						{"1_user_id": str(int(user)),
						 "2_item_id": str(int(item)),
						 "5_week":	str(int(week)),
						 "6_time":	str(int(time)),
						 "3_feat1":	str(int(feat1)),
						 "4_feat2":	str(int(feat2)),
						 "7_uwatched": 1.0/float(stats_user[int(user)]),
						 "8_iwatched": 1.0/float(stats_item[int(item)])
						})
				train_rating.append(float(rating))
			else:
				valid.append(
					{"1_user_id": str(int(user)),
					 "2_item_id": str(int(item)),
					 "5_week":	str(int(week)),
					 "6_time":	str(int(time)),
					 "3_feat1":	str(int(feat1)),
					 "4_feat2":	str(int(feat2)),
					 "7_uwatched": 1.0/float(stats_user[int(user)]),
					 "8_iwatched": 1.0/float(stats_item[int(item)])
					})
				valid_rating.append(float(rating))			

	return (train, np.array(train_rating)), (valid, np.array(valid_rating))

def loadTest(dir):
	test = []
	cnt = 0
	print "Start loading test data..."
	stats_user, stats_item = getUserWatchedCnt()
	with open(dir) as file:
		for line in file:
			if cnt % 10000 == 9999: 
				print "%d / 1524458 loaded\r"%(cnt+1),
			cnt += 1
			# if cnt == 100000: break
			(user, item, week, time, feat1, feat2)=line.split('\t')
			test.append(
				{"1_user_id": str(int(user)),
				 "2_item_id": str(int(item)),
				 "5_week":	str(int(week)),
				 "6_time":	str(int(time)),
				 "3_feat1":	str(int(feat1)),
				 "4_feat2":	str(int(feat2)),
				 "7_uwatched": 1.0/float(stats_user[int(user)]),
				 "8_iwatched": 1.0/float(stats_item[int(item)])
				})		

	return test

