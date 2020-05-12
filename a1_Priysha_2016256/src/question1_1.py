import glob
import cv2
import pickle
from sklearn.cluster import KMeans
import numpy as np

#<----------------------------------QUANTIZE COLOR SPACE----------------------------->
# colours = []
# for i in range(0,256,10):
# 	for j in range(0,256,10):
# 		for k in range(0,256,10):
# 			colours.append([i,j,k])
# kmeans = KMeans(n_clusters = 32, random_state = 42).fit(colours)
# print (kmeans.cluster_centers_)
# with open('kmeans.pkl', 'wb') as f:
# 	pickle.dump(kmeans, f)
# exit(0)

def prob(d, x, y, image):
	# neigh = [(x-d,y),(x+d,y),(x,y-d),(x,y+d),(x-d,y-d),(x-d,y+d),(x+d,y-d),(x+d,y+d)]
	samecol = 0
	exist = 0
	# for i in range(len(neigh)):
	for i in range(-d,d+1):
		neigh = [(x+i,y-d),(x+i,y+d)]
		for j in range(len(neigh)):
			if (neigh[j][0] >= 0 and neigh[j][0] < image.shape[0] and neigh[j][1] >= 0 and neigh[j][1] < image.shape[1]):
				exist = exist+1
				if image[x][y] == image[neigh[j][0]][neigh[j][1]]:
					samecol = samecol+1
	for i in range(-d+1,d):
		neigh = [(x-d,y+i),(x+d,y+i)]
		for j in range(len(neigh)):
			if (neigh[j][0] >= 0 and neigh[j][0] < image.shape[0] and neigh[j][1] >= 0 and neigh[j][1] < image.shape[1]):
				exist = exist+1
				if image[x][y] == image[neigh[j][0]][neigh[j][1]]:
					samecol = samecol+1
	return samecol,exist

def correlogram(image):
	sam = np.zeros((32,4))
	tot = np.ones((32,4))
	dis = [1,3,5,7]
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			for idx in range(len(dis)):
				d = dis[idx]
				same,total = prob(d, i, j, image)
				sam[image[i][j]][idx] = sam[image[i][j]][idx] + same
				tot[image[i][j]][idx] = tot[image[i][j]][idx] + total
	corr = np.divide(sam, tot)
	return corr

def start():
	with open('kmeans.pkl', 'rb') as f:
		kmeans = pickle.load(f)
		paths = glob.glob("../HW-1/images/*.jpg")
		dic = {}
		# images = []
		for i in paths:
			n = i.split("/")
			print (n[3])
			# dic[n[3]] = len(images)
			img = cv2.imread(i)
			img = cv2.resize(img, (int(img.shape[1]/4), int(img.shape[0]/4)), interpolation = cv2.INTER_AREA)
			# img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
#<-------------------------SAVE A SAMPLE CLUSTERED IMAGE---------------------------->		
			# image = np.zeros((img.shape[0], img.shape[1], 3))
			# for j in range(img.shape[0]):
			# 	for k in range(img.shape[1]):
			# 		image[j][k] = kmeans.cluster_centers_[kmeans.predict([img[j,k,:]])[0]]
			# cv2.imwrite("../Kmeans/" + n[3], image.astype(np.uint8))

			arr = kmeans.predict(img.reshape(img.shape[0]*img.shape[1],3))
			arr = np.asarray(arr).reshape(img.shape[0], img.shape[1])
			# print (arr)
			dic[n[3]] = correlogram(arr)

		with open('corrDic.pkl', 'wb') as f:
			pickle.dump(dic, f)

def sim(db, img):
	simi = []
	for i in db.keys():
		simi.append((i[:-4], np.sum(abs(np.subtract(img,db[i]))/(1+np.add(img,db[i])))/32))
	return simi

def read(qname, simi, num):
	f = open(qname[:14]+ "ground_truth/" + qname[20:-10] + "_good.txt", "r")
	q = f.readlines()
	q = list(map(lambda s: s.strip(), q))
#	print (q)
	f = open(qname[:14]+ "ground_truth/" + qname[20:-10] + "_ok.txt", "r")
	q1 = f.readlines()
	q1 = list(map(lambda s: s.strip(), q1))
	f = open(qname[:14]+ "ground_truth/" + qname[20:-10] + "_junk.txt", "r")
	q2 = f.readlines()
	q2 = list(map(lambda s: s.strip(), q2))
	good = 0
	ok = 0
	junk = 0
	for i,j in simi:
		if i in q:
			good = good + 1
		if i in q1:
			ok = ok + 1
		if i in q2:
			junk = junk + 1
	print ("good : ", good, ", ok : ", ok, ", junk : ", junk)
	pos = good+ok+junk
	precision = pos/(1.0*num)
	recall = pos/(1.0*(len(q)+len(q1)+len(q2)))
	f1 = (2*precision*recall)/(precision+recall)
	return precision, recall, f1
#	return good, ok, junk

def retrive():
	precision = []
	recall = []
	f1 = []
	query = glob.glob("../HW-1/train/query/*.txt")
	with open('corrDic.pkl', 'rb') as f:
		db = pickle.load(f)
		for i in query:
			print (i)
			f = open(i, "r")
			q = f.readline()
			q = q.split(" ")
			name = q[0][5:]
			print (name)
			bound = []
			bound.append(q[1]) # starting column
			bound.append(q[1]+q[3]) #end column
			bound.append(q[2]) # start row
			bound.append(q[2]+q[4]) #end row
			img = db[name+".jpg"]
			# print (bound)
			similarity = sim(db, img)
			similarity.sort(key = lambda x: x[1])
			a = similarity[:90]
			a.sort(key = lambda x: x[0])
#			print (a)
			p, r, f = read(i, similarity[:90], 90)
			precision.append(p)
			recall.append(r)
			f1.append(f)
	prec_avg = np.sum(precision)/(1.0*len(precision))
	rec_avg = np.sum(recall)/(1.0*len(recall))
	f1_avg = np.sum(f1)/(1.0*len(f1))
	print ("Average : Precision : ", prec_avg, ", Recall : ", rec_avg, ", F1 score : ", f1_avg)
	print ("Minimum : Precision : ", np.min(precision), ", Recall : ", np.min(recall), ", F1 score : ", np.min(f1))
	print ("Maximum : Precision : ", np.max(precision), ", Recall : ", np.max(recall), ", F1 score : ", np.max(f1))

# start()
retrive()


