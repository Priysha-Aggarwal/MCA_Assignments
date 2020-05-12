import numpy as np 
import string 
from nltk.corpus import abc
from nltk.corpus import stopwords
import re
from scipy.special import softmax
import copy
import pickle
def preprocess(corpus, stopwords):
	data = []
	for sentences in corpus:
		sen = []
		for word in sentences:
			if word.lower() not in stopwords:
				word_new = re.sub(r"[?><,.{};@#!&$-9876543210()\"\']+", '', word.lower())
				if len(word_new) > 0:
					sen.append(word_new)
		data.append(sen)
	return data

def training_dic(data, window_size):
	dic = {}
	ids = {}
	rev_ids = {}
	id_num = 0
	for sentence in data:
		N = len(sentence)
		for i in range(N):
			word = sentence[i]
			neighbour = []
			for j in range(1, window_size+1):
				if i-j >= 0:
					neighbour.append(sentence[i-j])
				if i+j < N:
					neighbour.append(sentence[i+j])
			if word in dic.keys():
				if neighbour not in dic[word]:
					dic[word] = dic[word] + neighbour
			else:
				dic[word] = neighbour
				ids[word] = id_num
				rev_ids[id_num] = word
				id_num = id_num+1
	return dic, ids, rev_ids

def create_data(train_dic, ids):
	X = []
	Y = []
	vocab_size = len(ids.keys())
	for key in train_dic.keys():
		hoten = np.zeros(vocab_size)
		hoten[ids[key]] = 1
		X.append(hoten)
		lis = []
		for word in train_dic[key]:
			lis.append(ids[word])
		Y.append(lis)
	X = np.array(X)
	Y = np.expand_dims(np.array(Y), axis=0)
	# print (Y)
	print (X.shape, Y.shape)
	return X, Y, vocab_size

# def hotencode(train_dic, ids):
# 	X = []
# 	Y = []
# 	vocab_size = len(ids.keys())
# 	for key in train_dic.keys():
# 		hoten = np.zeros(vocab_size)
# 		hoten[ids[key]] = 1
# 		context = np.zeros(vocab_size)
# 		for word in train_dic[key]:
# 			context[ids[word]] += 1
# 		X.append(hoten)
# 		Y.append(context)
# 	return X, Y, vocab_size

def init(vocab_size, dim):
	W_1 = np.random.uniform(-1,1,(vocab_size, dim))
	W_2 = np.random.randn(dim, vocab_size)
	return W_1, W_2

def forwardpass(datapoint, W_1, W_2, dim, vocab_size):
	out1 = W_1[np.where(datapoint == 1)[1][0]]
	out2 = np.matmul(out1, W_2)
	output = out2.reshape((vocab_size, 1))
	softmax_out = softmax(output)
	return softmax_out, out1.reshape((dim,1)), output

def backwardpass(datapoint, softmax_out, context, out1, learning_rate, W_1, W_2, vocab_size):
	soft = copy.deepcopy(softmax_out)
	# print (soft)
	soft[context, 0] -= 1.0
	# print (soft)
	soft += (len(context)-1)*softmax_out
	# print (soft)
	# print ()
	diff = soft
	err1 = (1/vocab_size) * np.matmul(out1, diff.T)
	err2 = np.matmul(W_2, diff).T
	W_2 = W_2 - learning_rate*err1
	W_1 = W_1 - learning_rate*(np.matmul(datapoint.T, err2))
	return W_1, W_2

def loss(prev_loss, vocab_size, context, output, learning_rate):
	i = 0
	count = 0
	while i < vocab_size:
		if context[i] == 1:
			count = count + 1
			prev_loss = prev_loss - output[i][0]
		i = i + 1
	prev_loss = prev_loss + count * np.log(np.sum(np.exp(output)))
	return prev_loss

def cross_entropy(softmax_out, Y, extra = 0.001):
    a = np.sum(np.log(softmax_out[Y, 0] + extra))
    cost = -(1 / softmax_out.shape[1]) * a
    return np.squeeze(cost)

def train(max_epoch, X, Y, vocab_size, dim, learning_rate = 0.001):
	W_1, W_2 = init(vocab_size, dim)
	N = len(X)
	for epoch in range(1, max_epoch+1):
		prev_loss = 0
		for idx in range(N):
			datapoint = X[idx].reshape((1, vocab_size))
			context = Y[0,idx]
			# print (context)
			softmax_out, out1, output = forwardpass(datapoint, W_1, W_2, dim, vocab_size)
			W_1, W_2 = backwardpass(datapoint, softmax_out, context, out1, learning_rate, W_1, W_2, vocab_size)
			# prev_loss = loss(prev_loss, vocab_size, context, output, learning_rate)
			prev_loss += cross_entropy(softmax_out, context)
		print ("epoch :", epoch, "loss : ", prev_loss)
		if epoch % 5 == 1 :
			np.save("W1_" + str(epoch), W_1)
			np.save("W2_" + str(epoch), W_2)
			learning_rate = 0.98*learning_rate
	return W_1, W_2

def predict(ids, vocab_size,word,number_of_predictions, W_1, W_2, rev_ids, dim = 20): 
	if word not in ids :
		print ("Model doesn't recognize this word")
		return
	index = ids[word]
	X = np.zeros(vocab_size)
	X[index] = 1
	prediction, out1, out2 = forwardpass(X.reshape((1, vocab_size)), W_1, W_2, dim, vocab_size) 
	temp = []
	for i in prediction:
		temp = temp + list(i)
	temp = np.array(temp)
	words = list(np.argsort(-temp)[0:number_of_predictions])
	simwords = []
	for i in words:
		simwords.append(rev_ids[i])
	print (simwords)
	print ()
	return simwords

window_size = 2
dim = 20
max_epoch = 51
stopwords = stopwords.words('english')
corpus = abc.sents()
# corpus = [["The","earth","revolves","around", "the", "sun"],["The"," moon", "revolves", "around", "the" ,"earth"]]
print (corpus)
data = preprocess(corpus, stopwords)
train_dic, ids, rev_ids = training_dic(data, window_size)
X, Y, vocab_size = create_data(train_dic, ids)
print (vocab_size)
exit(0)
with open("word_ids.pkl","wb") as f:
	pickle.dump(ids,f)
with open("rev_word_ids.pkl","wb") as f:
	pickle.dump(rev_ids,f)
# X, Y, vocab_size = hotencode(train_dic, ids)
X, Y, vocab_size = create_data(train_dic, ids)
with open("vocab_size.pkl","wb") as f:
	pickle.dump(vocab_size,f)
print ("vocab_size", vocab_size)
W_1, W_2 = train(max_epoch, X, Y, vocab_size, dim)
np.save("W1", W_1)
np.save("W2", W_2)
predict(ids, vocab_size, "around" ,3, W_1, W_2, rev_ids)
