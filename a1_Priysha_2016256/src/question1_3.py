import glob
import json
import cv2
import pickle
import numpy as np
from skimage.feature import hessian_matrix
from skimage.feature import blob_dog
from skimage.feature import hessian_matrix_det
from skimage.feature import blob
import math
from skimage.transform.integral import integral_image
import matplotlib.pyplot as plt
import timeit
def subtract (matrix1, matrix2):
    sub = np.zeros((len(matrix1), len(matrix1[0])))
    matrix1 = matrix1.astype(np.int16)
    matrix2 = matrix2.astype(np.int16)
    for i in range(len(matrix1)):
        for j in range (len(matrix1[0])):
        	sub[i][j] = abs(matrix2[i][j] - matrix1[i][j])
            # if (matrix1[i][j] < matrix2[i][j]):
            #     sub[i][j] = matrix2[i][j] - matrix1[i][j]
            # else:
            #     sub[i][j] = matrix1[i][j] - matrix2[i][j]
    sub = sub.astype(np.uint8)
    return sub

def keypt(scales):
	#padded = scales
	padded = []
	for img in scales :
		padded.append(np.pad(img,(1,1),mode='constant'))
	#	padded.append(padding(img))
	keypts = []
	for p in range(len(padded)):
		a = min(np.max(padded[p]), 0.006)
		for i in range(0,scales[0].shape[0]):
			for j in range(0,scales[0].shape[1]):
				if padded[p][i][j] >= a:
					intensities = inten(p,i,j,padded)
					if padded[p][i][j] > np.max(intensities):
						keypts.append((i,j,1.2*p))
						# keypts.append((i,j,math.pow(math.pow(2,0.25),p)*2))
	return keypts

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

def hessian(img):
	scales = []
	s = 2
	k = math.pow(2,0.25)
	
	for i in range(12):
		image = hessian_matrix_det(img, sigma=1.2*i)
		# image = hessian_matrix_det(img, sigma=math.pow(k,i)*s)
		Hrr, Hrc, Hcc = hessian_matrix(integral_image(img), sigma=math.pow(k,i)*s, order='rc')
		det = Hrr*Hcc - np.power((0.9*Hrc),2)
		scales.append(image)
	keypts = keypt(scales)
	return keypts

def display(keypts, img, name):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	fig, ax = plt.subplots()
	nh,nw,_ = img.shape
	count = 0
	ax.imshow(img, interpolation='nearest')
	for blob in keypts:
		y,x,r = blob
		c = plt.Circle((x, y), r*1.414, color='red', linewidth=1, fill=False)
        print (r*1.414)
		ax.add_patch(c)
	ax.axis('off') 
	ax.plot()  
	#plt.imsave("temp.jpg")
	fig.savefig("./SURFNew/" + name + '.png',bbox_inches = 'tight', pad_inches = 0,facecolor=fig.get_facecolor(), edgecolor='none')

paths = glob.glob("../HW-1/images/*.jpg")
#paths = glob.glob("../HW-1/train/query/*.txt")
dic = {}
for i in paths:
	start = timeit.default_timer()
	n = i.split("/")	
	print (n[3])
	#f = open(i, "r")
	#q = f.readline()
	#q = q.split(" ")
	#name = q[0][5:]
	#print (name)
	#im = cv2.imread("../HW-1/images/" + name + ".jpg")
	#im1 = cv2.resize(im, (int(im.shape[1]/4), int(im.shape[0]/4)), interpolation = cv2.INTER_AREA)
	#print (im1.shape)
	#img = cv2.imread("../HW-1/images/" + name + ".jpg",0)
	img = cv2.imread(i,0)
	image = cv2.resize(img, (int(img.shape[1]/4), int(img.shape[0]/4)), interpolation = cv2.INTER_AREA)/255.0
	image = cv2.GaussianBlur(image, (3,3), 0)
	keypts = hessian(image)
	keypts = np.array(keypts)
	keypts = blob._prune_blobs(keypts,0.3)
	stop = timeit.default_timer()
	print (stop-start)
	#display(keypts,im1,name)
	dic[n[3][:-4]] = list(keypts.tolist())
with open('surfDic.json', 'w') as f:
	json.dump(dic, f)
