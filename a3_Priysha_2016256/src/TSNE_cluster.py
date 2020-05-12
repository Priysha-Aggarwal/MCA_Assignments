import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
from nltk.corpus import abc
from scipy.special import softmax
from scipy.spatial import distance

def load_emb(file):
	emb = np.load(file)
	return emb

def tSNE(embedding_clusters):
	tsne = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
	shape0 = embedding_clusters.shape[0]
	shape1 = embedding_clusters.shape[1]
	embedding_clusters = embedding_clusters.reshape(embedding_clusters.shape[0] * embedding_clusters.shape[1], embedding_clusters.shape[2])
	tsne_embedding = tsne.fit_transform(embedding_clusters)
	tsne_embedding = tsne_embedding.reshape(shape0, shape1, 2)
	return tsne_embedding

def forwardpass(datapoint, W_1, W_2, dim, vocab_size):
	out1 = W_1[np.where(datapoint == 1)[1][0]]
	out2 = np.matmul(out1, W_2)
	output = out2.reshape((vocab_size, 1))
	softmax_out = softmax(output)
	return softmax_out, out1.reshape((dim,1)), output


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
		temp.append(i[0])
	temp = np.array(temp)
	words = list(np.argsort(-temp)[0:number_of_predictions])
	simwords = []
	for i in words:
		simwords.append(rev_ids[i])
	print (simwords)
	print ()
	return simwords


def findSimilarWords(word, count, W1, ids, vocab_size, rev_ids):
	print (word)
	print ()
	given_word = W1[ids[word]]
	simWords = {}
	for i in range(vocab_size):
		dist = distance.cosine(given_word, W1[i])
		simWords[i] = dist
	# print (simWords[1])
	sort = sorted(simWords.items(), key=lambda x:x[1])
	# print (sort)
	sim_words = []
	for i in range(1,count+1):
		idx = sort[i-1][0]
		print(rev_ids[idx])
		sim_words.append(rev_ids[idx]) 
	print ()
	return sim_words


def tsne_plot_similar_words(labels, embedding_clusters, word_clusters, filename):
	plt.figure(figsize=(16, 9))
	colors = np.linspace(0, 1, len(labels))
	colors = cm.rainbow(colors)
	count = 0
	for count in range(len(labels)):
		x = embedding_clusters[count][:, 0]
		y = embedding_clusters[count][:, 1]
		words = word_clusters[count]
		plt.scatter(x, y, c=colors[count], label=labels[count])
		for i in range(len(words)):
			cord = (x[i], y[i])
			plt.annotate(words[i], xy=cord)
	plt.legend(loc=4)
	plt.title('TSNE plot of cluster formation')
	plt.grid(True)
	plt.savefig(filename)
	plt.show()

W_1 = load_emb("./WordEmb/W1_51.npy")
# W_2 = load_emb("./WordEmb/W2_51.npy")
with open("rev_word_ids.pkl", "rb") as f:
	rev_ids = pickle.load(f)
with open("word_ids.pkl", "rb") as f:
	ids = pickle.load(f)
vocab_size = len(ids.keys())

embedding_clusters = []
word_clusters = []
idx = [1,10,60,990,560,459,23000,5600,4300,11000,26000,3452, 9870, 13479,8425, 12345, 21845, 1780, 13279, 2347]
keys = []
for ind in idx:
	word = rev_ids[ind]
	keys.append(word)
	embeddings = {}
	words = {}
	# context_words = predict(ids, vocab_size, word, 20, W_1, W_2, rev_ids)
	# continue
	context_words = findSimilarWords(word, 20, W_1, ids, vocab_size, rev_ids)
	# continue
	for simword in context_words:
		if word in words.keys():
			words[word] = words[word] + [simword]
			embeddings[word] = embeddings[word] + [W_1[ids[simword]]]
		else:
			words[word] = [simword]
			embeddings[word] = [W_1[ids[simword]]]
	embedding_clusters.append(embeddings[word])
	word_clusters.append(words[word])
# exit(0)
embedding_clusters = np.array(embedding_clusters)

tsne_embedding = tSNE(embedding_clusters)

tsne_plot_similar_words(keys, tsne_embedding, word_clusters, 'similar_wordsW51.png')
