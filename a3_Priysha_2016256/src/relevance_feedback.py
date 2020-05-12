import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import copy
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

def load_gt(file = 'data/MED.REL'):
    with open(file, 'r') as f:
        lines = f.readlines()
    lis = [(int(l.split()[0]), int(l.split()[2])) for l in lines]
    dic = {}
    for i in lis:
    	if i[0] in dic.keys():
    		dic[i[0]] = dic[i[0]] + [i[1]-1]
    	else:
    		dic[i[0]] = [i[1]-1]
    return dic


# def load_gt(file= 'data/MED.REL'):
#     with open(file, 'r') as f:
#         lines = f.readlines()
#     a = [(int(l.split()[0]), int(l.split()[2])) for l in lines]
#     truth = {}
#     for i in a:
#         if(i[0]-1 not in truth.keys()):
#             truth[i[0]-1] = [i[1]-1]
#         else:
#             truth[i[0]-1].append(i[1]-1)
#     return truth
# def relevance_feedback(vec_docs, vec_queries, sim, n=10):
# 	"""
# 	relevance feedback
# 	Parameters
# 		----------
# 		vec_docs: sparse array,
# 			tfidf vectors for documents. Each row corresponds to a document.
# 		vec_queries: sparse array,
# 			tfidf vectors for queries. Each row corresponds to a document.
# 		sim: numpy array,
# 			matrix of similarities scores between documents (rows) and queries (columns)
# 		n: integer
# 			number of documents to assume relevant/non relevant

# 	Returns
# 	-------
# 	rf_sim : numpy array
# 		matrix of similarities scores between documents (rows) and updated queries (columns)
# 	"""

# 	sim_matrix = copy.deepcopy(sim)
# 	for iterations in range(10):
# 		for i in range(sim_matrix.shape[1]):
# 			ranked_documents = np.argsort(-sim_matrix[:, i])
# 			top = ranked_documents[:n]
# 			bottom = ranked_documents[-1*n:]
# 			# print ('Query:', i+1, 'Top relevant 10 documents:', ranked_documents[:10] + 1)
# 			# print ('Query:', i+1, 'Least relevant 10 documents:', ranked_documents[-10:] + 1)
# 			d_R = vec_docs[top[0]]
# 			d_NR = vec_docs[bottom[0]]
# 			for j in range(1, len(top)):
# 				d_R = d_R + vec_docs[top[j]]
# 				d_NR = d_NR + vec_docs[bottom[j]]
# 			# print (vec_queries[i])
# 			vec_queries[i] = vec_queries[i] + (2.0/n)*d_R - (1.5/n)*d_NR
# 			# print (vec_queries[i])
# 		sim_matrix = cosine_similarity(vec_docs, vec_queries) # change
# 	return sim_matrix


def relevance_feedback(vec_docs, vec_queries, sim, n=10, alpha=0.75, beta=.15, iteration=15):
	"""
	relevance feedback
	Parameters
		----------
		vec_docs: sparse array,
			tfidf vectors for documents. Each row corresponds to a document.
		vec_queries: sparse array,
			tfidf vectors for queries. Each row corresponds to a document.
		sim: numpy array,
			matrix of similarities scores between documents (rows) and queries (columns)
		n: integer
			number of documents to assume relevant/non relevant

	Returns
	-------
	rf_sim : numpy array
		matrix of similarities scores between documents (rows) and updated queries (columns)
	"""
	gt = load_gt()
	sim_matrix = copy.deepcopy(sim)
	for iterations in range(iteration):
		for i in range(sim_matrix.shape[1]):
			ranked_documents = np.argsort(-sim_matrix[:, i])
			out = ranked_documents[:n]
			d_R = []
			d_NR = []
			for doc in out:
				if doc in gt[i+1]:
					d_R.append(vec_docs[doc])
				else:
					d_NR.append(vec_docs[doc])
			r = len(d_R)
			nr = len(d_NR)
			d_R = np.sum(np.array(d_R), axis = 0)
			d_NR = np.sum(np.array(d_NR), axis = 0)
			if r > 0 and nr > 0:
				vec_queries[i] = vec_queries[i] + (alpha/r)*d_R - (beta/nr)*d_NR
			else:
				if r == 0:
					vec_queries[i] = vec_queries[i] - (beta/nr)*d_NR
				elif nr == 0:
					vec_queries[i] = vec_queries[i] + (alpha/r)*d_R
				else:
					print ("HERE")
					vec_queries[i] = vec_queries[i]
			# arr = vec_queries[i].toarray()
			# idx = np.where(arr < 0)
			# arr[idx] = 0
			# vec_queries[i] = arr
		sim_matrix = cosine_similarity(vec_docs, vec_queries)
	return sim_matrix

# def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, n=10):
# 	"""
# 	relevance feedback with expanded queries
# 	Parameters
# 		----------
# 		vec_docs: sparse array,
# 			tfidf vectors for documents. Each row corresponds to a document.
# 		vec_queries: sparse array,
# 			tfidf vectors for queries. Each row corresponds to a document.
# 		sim: numpy array,
# 			matrix of similarities scores between documents (rows) and queries (columns)
# 		tfidf_model: TfidfVectorizer,
# 			tf_idf pretrained model
# 		n: integer
# 			number of documents to assume relevant/non relevant

# 	Returns
# 	-------
# 	rf_sim : numpy array
# 		matrix of similarities scores between documents (rows) and updated queries (columns)
# 	"""
# 	gt = load_gt()
# 	sim_matrix = copy.deepcopy(sim)
# 	for iterations in range(10):
# 		for i in range(sim_matrix.shape[1]):
# 			ranked_documents = np.argsort(-sim_matrix[:, i])
# 			out = ranked_documents[:n]
# 			d_R = []
# 			d_NR = []
# 			for doc in out:
# 				if doc in gt[i+1]:
# 					d_R.append(vec_docs[doc])
# 				else:
# 					d_NR.append(vec_docs[doc])
# 			d_R = np.sum(np.array(d_R), axis = 0)
# 			d_NR = np.sum(np.array(d_NR), axis = 0)
# 			vec_queries[i] = vec_queries[i] + (0.65/n)*d_R - (0.25/n)*d_NR
# 		# sim_matrix = cosine_similarity(vec_docs, vec_queries)

# 		vec = normalize(vec_docs.T)
# 		C = np.dot(vec, vec.T)
# 		for i in range(sim.shape[1]):
# 			top_words = np.argsort(-vec_queries[i])[:10]
# 			for top_word in top_words:
# 				temp = C[top_word].toarray()
# 				syn = np.argsort(-temp)[0][:n]
# 				typ = type(vec_queries[i])
# 				temp = vec_queries[i].toarray()
# 				temp[0][syn] = temp[0][top_word]
# 				vec_queries[i] = typ(temp)
# 		sim_matrix = cosine_similarity(vec_docs, vec_queries)
# 	return sim_matrix

def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, n=10, alpha = 0.75, beta = 0.15):
	"""
	relevance feedback with expanded queries
	Parameters
		----------
		vec_docs: sparse array,
			tfidf vectors for documents. Each row corresponds to a document.
		vec_queries: sparse array,
			tfidf vectors for queries. Each row corresponds to a document.
		sim: numpy array,
			matrix of similarities scores between documents (rows) and queries (columns)
		tfidf_model: TfidfVectorizer,
			tf_idf pretrained model
		n: integer
			number of documents to assume relevant/non relevant

	Returns
	-------
	rf_sim : numpy array
		matrix of similarities scores between documents (rows) and updated queries (columns)
	"""
	gt = load_gt()
	sim_matrix = copy.deepcopy(sim)
	for iterations in range(50):
		for i in range(sim_matrix.shape[1]):
			ranked_documents = np.argsort(-sim_matrix[:, i])
			out = ranked_documents[:n]
			d_R = []
			d_NR = []
			for doc in out:
				if doc in gt[i+1]:
					d_R.append(vec_docs[doc])
				else:
					d_NR.append(vec_docs[doc])
			r = len(d_R)
			nr = len(d_NR)
			d_R = np.sum(np.array(d_R), axis = 0)
			d_NR = np.sum(np.array(d_NR), axis = 0)
			if r > 0 and nr > 0:
				vec_queries[i] = vec_queries[i] + (alpha/r)*d_R - (beta/nr)*d_NR
			else:
				if r == 0:
					vec_queries[i] = vec_queries[i] - (beta/nr)*d_NR
				elif nr == 0:
					vec_queries[i] = vec_queries[i] + (alpha/r)*d_R
				else:
					print ("HERE")
					vec_queries[i] = vec_queries[i]
		# sim_matrix = cosine_similarity(vec_docs, vec_queries)

	# vec = normalize(vec_docs, axis = 0)
	# for i in range(sim.shape[1]):
			print ("Query:", i + 1)
			top_doc = gt[i+1][:n] #top 10 documents
			for doc in top_doc: #for each doc
				top_words = np.argsort(-vec_docs[doc].toarray()[0])[:n] #top 10 words in each doc
				typ = type(vec_queries[i])
				temp = vec_queries[i].toarray()
				temp[0][top_words] += vec_docs[doc].toarray()[0][top_words] #add tfidf value to query vector
				vec_queries[i] = typ(temp)
		sim_matrix = cosine_similarity(vec_docs, vec_queries) #cosine similarity
	return sim_matrix