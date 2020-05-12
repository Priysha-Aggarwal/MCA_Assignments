from sklearn.svm import SVC
import pickle
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import random
from sklearn.preprocessing import MinMaxScaler

#<-------------------------------------SVM with noise------------------------------------>
# with open("trainingfeaturesSPEC.pkl", "rb") as f:
# 	train_x = np.array(pickle.load(f))
# with open("noisefeaturesSPEC.pkl", "rb") as f:
# 	noise = np.array(pickle.load(f))
# with open("traininglabelsSPEC.pkl", "rb") as f:
# 	train_y = np.array(pickle.load(f))
# print (np.shape(train_x))
# print (np.shape(train_y))

# xr = random.randint(1,100)
# for i in range(xr):
# 	a = random.randint(1,train_x.shape[0]-1)
# 	b = random.randint(1,train_x.shape[1]-1)
# 	c = random.randint(1,noise.shape[0]-1)
# 	d = random.randint(1,noise[c].shape[0]-1)
# 	# print (a,b,c,d)
# 	train_x[a][b] += noise[c][d]

# train = np.zeros((train_x.shape[0], train_x.shape[1]*train_x.shape[2]))

# for i in range(train_x.shape[0]):
# 	train[i] = np.ravel(train_x[i])
# print (train.shape)
# # clf = SVC(gamma = 'auto', kernel='rbf')
# clf = LinearSVC(random_state=0)
# # clf = OneVsRestClassifier(LinearSVC(random_state=0))
# clf.fit(train,train_y)

# with open("SVMspec.pkl", "wb") as f:
# 	pickle.dump(clf, f)

# with open("SVMspec.pkl", "rb") as f:
# 	clf = pickle.load(f)

# with open("validationfeaturesSPEC.pkl", "rb") as f:
# 	val_x = np.array(pickle.load(f))
# with open("validationlabelsSPEC.pkl", "rb") as f:
# 	val_y = np.array(pickle.load(f))
# val = np.zeros((val_x.shape[0], val_x.shape[1]*val_x.shape[2]))

# for i in range(val_x.shape[0]):
# 	val[i] = np.ravel(val_x[i])
# print (val.shape)
# print (np.shape(val_x))
# print (np.shape(val_y))
# print (clf.score(val, val_y))
# y_pred = clf.predict(val)
# tp = np.zeros(10)
# tn = np.zeros(10)
# fp = np.zeros(10)
# fn = np.zeros(10)
# precision = np.zeros(10)
# recall = np.zeros(10)
# for clas in range(10):
# 	pos = clas
# 	for i in range(len(y_pred)):
# 		pred = y_pred[i]
# 		actual = val_y[i]
# 		if pred == actual:
# 			if pred == pos:
# 				tp[clas] += 1
# 			else:
# 				tn[clas] += 1
# 		else:
# 			if pred == pos:
# 				fp[clas] += 1
# 			else:
# 				fn[clas] += 1
# 	precision[clas] = tp[clas]/(tp[clas] + fp[clas])
# 	recall[clas] = tp[clas]/(tp[clas] + fn[clas])
# print ("Precision", precision)
# print ("Recall", recall)
# print (classification_report(val_y, y_pred))


#<-------------------------------------MFCC with noise------------------------------------>
# with open("trainingfeaturesMFCC.pkl", "rb") as f:
# 	train_x = np.array(pickle.load(f))
# with open("noisefeaturesMFCC.pkl", "rb") as f:
# 	noise = np.array(pickle.load(f))
# 	# noise = noise.reshape((noise.shape[0], noise[1].shape[0], noise[1].shape[1]))
# with open("traininglabelsMFCC.pkl", "rb") as f:
# 	train_y = np.array(pickle.load(f))
# print (np.shape(train_x))
# print (np.shape(train_y))
# print (np.shape(noise))

# xr = random.randint(1,100)
# for i in range(xr):
# 	a = random.randint(1,train_x.shape[0]-1)
# 	b = random.randint(1,train_x.shape[1]-1)
# 	c = random.randint(1,noise.shape[0]-1)
# 	d = random.randint(1,noise[c].shape[0]-1)
# 	# print (a,b,c,d)
# 	train_x[a][b] += noise[c][d]

# train = np.zeros((train_x.shape[0], train_x.shape[1]*train_x.shape[2]))
# for i in range(train_x.shape[0]):
# 	train[i] = np.ravel(train_x[i])
# print (train.shape)

# with open("train_noise.pkl", "wb") as f:
# 	pickle.dump(train, f)


# # clf = SVC(gamma = 'auto', kernel='rbf')
# clf = LinearSVC(random_state=0)
# # clf = OneVsRestClassifier(LinearSVC(random_state=0))
# clf.fit(train,train_y)
# with open("SVMmfcc.pkl", "wb") as f:
# 	pickle.dump(clf, f)

# with open("SVMmfcc.pkl", "rb") as f:
# 	clf = pickle.load(f)

# with open("validationfeaturesMFCC.pkl", "rb") as f:
# 	val_x = np.array(pickle.load(f))
# with open("validationlabelsMFCC.pkl", "rb") as f:
# 	val_y = np.array(pickle.load(f))
# val = np.zeros((val_x.shape[0], val_x.shape[1]*val_x.shape[2]))

# for i in range(val_x.shape[0]):
# 	val[i] = np.ravel(val_x[i])
# print (val.shape)
# print (np.shape(val_x))
# print (np.shape(val_y))
# print (clf.score(val, val_y))
# y_pred = clf.predict(val)
# print (classification_report(val_y, y_pred))


#<-------------------------------------SPEC without noise------------------------------------>
with open("trainingfeaturesSPEC.pkl", "rb") as f:
	train_x = np.array(pickle.load(f))
with open("traininglabelsSPEC.pkl", "rb") as f:
	train_y = np.array(pickle.load(f))
print (np.shape(train_x))
print (np.shape(train_y))

train = np.zeros((train_x.shape[0], train_x.shape[1]*train_x.shape[2]))

for i in range(train_x.shape[0]):
	train[i] = np.ravel(train_x[i])
print (train.shape)
# clf = SVC(gamma = 'auto', kernel='rbf')
clf = LinearSVC(random_state=0)
# clf = OneVsRestClassifier(LinearSVC(random_state=0))
clf.fit(train,train_y)
with open("SVMspec_Nonoise.pkl", "wb") as f:
	pickle.dump(clf, f)

# with open("SVMspec_Nonoise.pkl", "rb") as f:
	# clf = pickle.load(f)

with open("validationfeaturesSPEC.pkl", "rb") as f:
	val_x = np.array(pickle.load(f))
with open("validationlabelsSPEC.pkl", "rb") as f:
	val_y = np.array(pickle.load(f))
val = np.zeros((val_x.shape[0], val_x.shape[1]*val_x.shape[2]))

for i in range(val_x.shape[0]):
	val[i] = np.ravel(val_x[i])
print (val.shape)
print (np.shape(val_x))
print (np.shape(val_y))
print (clf.score(val, val_y))
y_pred = clf.predict(val)
print (classification_report(val_y, y_pred))

#<-------------------------------------MFCC without noise------------------------------------>
# with open("trainingfeaturesMFCC.pkl", "rb") as f:
# 	train_x = np.array(pickle.load(f))
# with open("traininglabelsMFCC.pkl", "rb") as f:
# 	train_y = np.array(pickle.load(f))
# print (np.shape(train_x))
# print (np.shape(train_y))

# train = np.zeros((train_x.shape[0], train_x.shape[1]*train_x.shape[2]))

# for i in range(train_x.shape[0]):
# 	train[i] = np.ravel(train_x[i])
# print (train.shape)
# # clf = SVC(gamma = 'auto', kernel='rbf')
# clf = LinearSVC(random_state=0)
# # clf = OneVsRestClassifier(LinearSVC(random_state=0))
# # scale = MinMaxScaler()
# # train = scale.fit_transform(train)
# clf.fit(train,train_y)
# with open("SVMmfcc_Nonoise.pkl", "wb") as f:
# 	pickle.dump(clf, f)

# # with open("SVMmfcc_Nonoise.pkl", "rb") as f:
# 	# clf = pickle.load(f)

# with open("validationfeaturesMFCC.pkl", "rb") as f:
# 	val_x = np.array(pickle.load(f))
# with open("validationlabelsMFCC.pkl", "rb") as f:
# 	val_y = np.array(pickle.load(f))
# val = np.zeros((val_x.shape[0], val_x.shape[1]*val_x.shape[2]))

# for i in range(val_x.shape[0]):
# 	val[i] = np.ravel(val_x[i])
# print (val.shape)
# # scale = MinMaxScaler()
# # val = scale.fit_transform(val)
# print (np.shape(val_x))
# print (np.shape(val_y))
# print (clf.score(val, val_y))
# y_pred = clf.predict(val)
# print (classification_report(val_y, y_pred))