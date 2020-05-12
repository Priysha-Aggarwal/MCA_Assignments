import glob
import cv2
import pickle
from sklearn.cluster import KMeans
import numpy as np
from scipy import ndimage
import math
import matplotlib.pyplot as plt
import pywt
from skimage.feature import blob
import timeit
import json
def padding(image):
	k = 3
	width = len(image[0])
	height = len(image)
	matrix = np.zeros((height+2*(math.floor(k/2)), width+2*(math.floor(k/2))))
	for i in range(math.floor(k/2), height+math.floor(k/2)):
		for j in range(math.floor(k/2), width+math.floor(k/2)):
			matrix[i][j] = image[i-math.floor(k/2)][j-math.floor(k/2)]
	return matrix

def inten1(p,x,y,padded):
	lis = []
	d=1
	neigh = [(x-d,y),(x+d,y),(x,y-d),(x,y+d),(x-d,y-d),(x-d,y+d),(x+d,y-d),(x+d,y+d)]
	im = padded[p]
	for i in neigh:
		lis.append(im[i[0]][i[1]])
	if p-1>=0:
		im = padded[p-1]
		for i in neigh:
			lis.append(im[i[0]][i[1]])
		lis.append(im[x][y])
	if p+1<len(padded):
		im=padded[p+1]
		for i in neigh:
			lis.append(im[i[0]][i[1]])
		lis.append(im[x][y])
	return lis

def inten(p,x,y,padded):
	lis = []
	if p-1>=0:
		# lis.extend(list(np.ravel(padded[p-1][x-1:x+2,y-1:y+2])))
		lis.append(np.max(padded[p-1][x-1:x+2,y-1:y+2]))
	if p+1<len(padded):
		# lis.extend(list(np.ravel(padded[p+1][x-1:x+2,y-1:y+2])))
		lis.append(np.max(padded[p+1][x-1:x+2,y-1:y+2]))
	# lis.extend(list(np.ravel(padded[p][x-1,y-1:y+2])))
	lis.append(np.max(padded[p][x-1,y-1:y+2]))
	# lis.extend(list(np.ravel(padded[p][x+1,y-1:y+2])))
	lis.append(np.max(padded[p][x+1,y-1:y+2]))
	lis.append(padded[p][x][y-1])
	lis.append(padded[p][x][y+1])
#	print (lis)
	return lis

def inten2(p,x,y,padded):
	lis = []
	if p-1>=0:
		lis.extend(list(np.ravel(padded[p-1][x-1:x+2,y-1:y+2])))
	if p+1<len(padded):
		lis.extend(list(np.ravel(padded[p+1][x-1:x+2,y-1:y+2])))
	try:
		lis.extend(list(np.ravel(padded[p][x-1,y-1:y+2])))
	except:
		a=0
	try:
		lis.extend(list(np.ravel(padded[p][x+1,y-1:y+2])))
	except:
		a=0
	try :
		lis.append(padded[p][x][y-1])
	except:
		a=0
	try:
		lis.append(padded[p][x][y+1])
	except : 
		a = 0
#	print (lis)
	return lis

def keypt(scales):
	padded = scales
	# padded = []
	# for img in scales :
	# 	padded.append(np.pad(img,(1,1)))
	# #	padded.append(padding(img))
	keypts = []
	for p in range(len(padded)):
		#print (np.max(padded))
		a = min(np.max(padded[p]), 0.1)
		for i in range(0,scales[0].shape[0]):
			for j in range(0,scales[0].shape[1]):
				if padded[p][i][j] >= a:
					intensities = inten2(p,i,j,padded)
					if padded[p][i][j] > np.max(intensities)+0.0001:
						keypts.append((i,j,math.pow(math.pow(2,0.25),p)*2))
	return keypts


def log(img):
	scales = []
	s = 2
	k = math.pow(2,0.25)
	
	for i in range(15):
		# ((math.pow(k,i)*s)**2)*
		image = ((math.pow(k,i)*s)**2)*(ndimage.gaussian_laplace(img, sigma=math.pow(k,i)*s, mode="constant"))
		scales.append(image)
	keypts = keypt(scales)
	return keypts

def display(keypts, img, name):
	fig, ax = plt.subplots()
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	nh,nw,_ = img.shape
	count = 0
	ax.imshow(img, interpolation='nearest')
	for blob in keypts:
		y,x,r = blob
		c = plt.Circle((x, y), r*1.414, color='red', linewidth=1, fill=False)
        print (r*1.414)
		ax.add_patch(c)
	ax.plot()  
	fig.savefig('./QueryBlobs/'+ name + '.png')

def harr(img, keypts):
	img = np.pad(img, (8,8))
	des = []
	for blob in keypts:
		lis = []
		x,y,r = blob
		x = int(x + 8)
		y = int(y + 8)
		region = img[x-8:x+8,y-8:y+8]
		for i in range(0,13,4):
			for j in range(0,13,4):
				grid = region[i:i+4,j:j+4]
				coeffs = pywt.dwt2(grid, 'haar')
				LL, (LH, HL, HH) = coeffs
				lis.extend([np.sum(LH),np.sum(HL),np.sum(abs(LH)),np.sum(abs(HL))])
		des.append(lis)
	return des

def sim(query, db):
	cnts = []
	for i in db.keys():
		count = 0
		for j in db[query]:
			simi = []
			for k in db[i]:
				simi.append(np.linalg.norm(np.array(j)-np.array(k)))
			simi.sort()
			if simi[0]*1.0/simi[1] <= 0.8:
				count = count + 1
		cnts.append((i,count))
	cnts.sort(key = lambda x: x[1], reverse=True)
	print (cnts[:10])

def prin(img, keypts):
	for k in keypts:
		x,y,r = k
		x = int(x)
		y = int(y)
		print (x,y, img[x][y])
		print (np.ravel(img[x-1,y-1:y+2]), np.ravel(img[x+1,y-1:y+2]), img[x][y-1], img[x][y+1])

def start ():
	# paths = glob.glob("../HW-1/train/query/*.txt")
	paths = glob.glob("../HW-1/images/*.jpg")
	dic = {}
	for i in paths:
		# f = open(i, "r")
		# q = f.readline()
		# q = q.split(" ")
		# name = q[0][5:]
		# print (name)

		n = i.split("/")
		# start = timeit.default_timer()
		print (n[3])
		# im = cv2.imread("../HW-1/images/" + name + ".jpg")
		# im = cv2.resize(im, (int(im.shape[1]/4), int(im.shape[0]/4)), interpolation = cv2.INTER_AREA)
		img = cv2.imread(i,0)
		img = img/255.0
		img = cv2.resize(img, (int(img.shape[1]/4), int(img.shape[0]/4)), interpolation = cv2.INTER_AREA)
		image = cv2.GaussianBlur(img, (3,3), 0)
		keypts = log(image)
		#print (keypts)
		#display(keypts,img,n[3][:-4])
		keypts = np.array(keypts)
		keypts = blob._prune_blobs(keypts,0.5)
		#prin(img,keypts)
		# display(keypts,im,name)

		# des = harr(img, keypts)
		dic[n[3][:-4]] = list(keypts.tolist())
		# print (len(des))
		# stop = timeit.default_timer()
		# print (stop-start)
	return dic
def retrieve(dic):
	query = glob.glob("../HW-1/train/query/*.txt")
	# with open('corrDic.pkl', 'rb') as f:
		# db = pickle.load(f)
	db = dic
	for i in query:
		f = open(i, "r")
		q = f.readline()
		q = q.split(" ")
		name = q[0][5:]
		bound = []
		bound.append(q[1]) # starting column
		bound.append(q[1]+q[3]) #end column
		bound.append(q[2]) # start row
		bound.append(q[2]+q[4]) #end row
		sim(name, db)

db = start()
with open('blobDic.json', 'w') as f:
	json.dump(db, f)
#retrieve(db)
