import csv

name_list = []
data_list = []

with open('tenth.csv') as file:
	for line in file:
		(name, rating)=line.split(',')
		if 'pred' not in rating:
			name_list.append(name)
			data_list.append(float(rating))

with open('eleventh.csv') as file:
	cnt = 0
	for line in file:
		(name, rating)=line.split(',')
		if 'pred' not in rating:
			data_list[cnt] = data_list[cnt]*0.2 + float(rating)*0.18
			cnt = cnt + 1

with open('twelveth.csv') as file:
	cnt = 0
	for line in file:
		(name, rating)=line.split(',')
		if 'pred' not in rating:
			data_list[cnt] = data_list[cnt] + float(rating)*0.11
			cnt = cnt + 1

with open('thirteenth.csv') as file:
	cnt = 0
	for line in file:
		(name, rating)=line.split(',')
		if 'pred' not in rating:
			data_list[cnt] = data_list[cnt] + float(rating)*0.11
			cnt = cnt + 1

with open('fourteenth.csv') as file:
	cnt = 0
	for line in file:
		(name, rating)=line.split(',')
		if 'pred' not in rating:
			data_list[cnt] = data_list[cnt] + float(rating)*0.2
			cnt = cnt + 1

with open('fifteenth.csv') as file:
	cnt = 0
	for line in file:
		(name, rating)=line.split(',')
		if 'pred' not in rating:
			data_list[cnt] = data_list[cnt] + float(rating)*0.2
			cnt = cnt + 1

with open('merged6.csv', 'w') as csvfile:
	fieldnames = ['uid#iid', 'pred']
	writer = csv.DictWriter(csvfile, fieldnames)
	writer.writeheader()
	for ind in xrange(len(name_list)):
		writer.writerow({'uid#iid': "%s"%(name_list[ind]) ,'pred': "%f"%data_list[ind]})
