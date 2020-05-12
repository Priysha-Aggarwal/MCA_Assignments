import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle

def load_emb(file):
	emb = np.load(file)
	return emb

def tSNE(embedding):
	tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3500, random_state=32)
	tsne_embedding = tsne.fit_transform(embedding)
	return tsne_embedding

def plot(label, x, y, name):
	plt.figure(figsize=(16, 9))
	plt.scatter(x, y, c='red', alpha=0.1)
	plt.grid(True)
	plt.legend(loc=4)
	plt.title(label)
	plt.savefig(name + ".png")

lis = ["W1_1", "W1_6", "W1_11", "W1_16", "W1_21", "W1_26", "W1_31", "W1_36", "W1_41", "W1_46", "W1_51"]
for name in lis:
	embedding = load_emb("./" + name + ".npy")
	print (np.shape(embedding))
	tsne_embedding = tSNE(embedding)
	x = tsne_embedding[:,0]
	y = tsne_embedding[:,1]
	plot('t-SNE graph of word embeddings', x, y, name)
