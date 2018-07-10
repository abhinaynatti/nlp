import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from collections import defaultdict
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from scipy import sparse
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
stop_words=stopwords.words("english")

def stopword_removal(list1,list2): 
	list1=[word for word in list1 if word not in stop_words]
	list2=[word for word in list2 if word not in stop_words]
	return list1,list2


def svm_func(train_input,test_input,test_X,test_Y) :
	clf=svm.LinearSVC()
	clf.fit(train_input,test_X)
	print(clf.score(test_input,test_Y))

def naive_func(train_input,test_input,test_X,test_Y) :
	naive_obj=BernoulliNB()
	naive_obj.fit(train_input,test_X)
	print(naive_obj.score(test_input,test_Y))

def feedforw_func(train_input,test_input,test_X,test_Y):
	mlp=MLPClassifier(hidden_layer_sizes=(5,100),solver='adam')
	mlp.fit(train_input,test_X)
	print(mlp.score(test_input,test_Y))

def logisticregression_func(train_input,test_input,test_X,test_Y):
	lr=LogisticRegression()
	lr.fit(train_input,test_X)
	print(lr.score(test_input,test_Y))

def bow_func(list1,list2):
	vectorizer=CountVectorizer(binary=True)
	list_train=vectorizer.fit_transform(list1).toarray()
	list_test=vectorizer.transform(list2).toarray()
	return list_train,list_test

def normalized_tf_func(list1,list2):
	vectorizer=CountVectorizer(binary=True)
	list_train=vectorizer.fit_transform(list1).toarray()
	list_test=vectorizer.transform(list2).toarray()
	tfidf_transformer = TfidfTransformer(smooth_idf=False,use_idf = False, norm = 'l1')
	tfidf_bow_input = tfidf_transformer.fit_transform(list_train).toarray()
	tfidf_bow_output = tfidf_transformer.fit_transform(list_test).toarray()
	return tfidf_bow_input,tfidf_bow_output

def tfidf_func(list1,list2):
	vectorizer=CountVectorizer()
	list_train=vectorizer.fit_transform(list1).toarray()
	list_test=vectorizer.transform(list2).toarray()
	tfidf_transformer = TfidfTransformer(smooth_idf=False)
	tfidf_bow_input = tfidf_transformer.fit_transform(list_train).toarray()
	tfidf_bow_output = tfidf_transformer.fit_transform(list_test).toarray()
	return tfidf_bow_input,tfidf_bow_output


def word2vec_func(corpus,flag):
	with open("GoogleNews-vectors-negative300.txt","r") as f:
		d={}
		for line in f:
			line=line.split()
			y=[]
			for i in line[1:]:
				y.append(float(i))
			d[line[0]]=np.array(y)


		if(flag):
			tfidf_vectorizer=TfidfVectorizer()
			frequencies=tfidf_vectorizer.fit(corpus)
			dict2={}
			dict2 = defaultdict(lambda : max(frequencies.idf_)) 
			x=[]
			for w,i in frequencies.vocabulary_.items():
				dict2[w]=frequencies.idf_[i]
			for document in corpus:
				vect_list=[]
				for word in document.split():                 
					if word in d:
						vect_list.append(np.array(d[word]*dict2[word]))
					else:
						vect_list.append(np.zeros(300))
				x.append(np.mean(np.array(vect_list),axis=0))
			x=np.array(x)
			return sparse.csr_matrix(x)
		else:
			x=[]
			for document in corpus:
				vect_list=[]
				for word in document.split():
					if word in d:
						vect_list.append(np.array(d[word]))
					else:
						vect_list.append(np.zeros(300))
				x.append(np.mean(np.array(vect_list),axis=0))
			x=np.array(x)
			return sparse.csr_matrix(x)
			

def glove_func(corpus,flag):
		with open("./glove.6B/glove.6B.300d.txt","r") as f:
			d={}
			for line in f:
				line=line.split()
				y=[]
				for i in line[1:]:
					y.append(float(i))
				d[line[0]]=np.array(y)


			if(flag):
				tfidf_vectorizer=TfidfVectorizer()
				frequencies=tfidf_vectorizer.fit(corpus)
				dict2={}
				dict2 = defaultdict(lambda : max(frequencies.idf_)) 
				x=[]
				for w,i in frequencies.vocabulary_.items():
					dict2[w]=frequencies.idf_[i]
				for document in corpus:
					vect_list=[]
					for word in document.split():                 
						if word in d:
							vect_list.append(np.array(d[word]*dict2[word]))
						else:
							vect_list.append(np.zeros(300))
					x.append(np.mean(np.array(vect_list),axis=0))
				x=np.array(x)
				return sparse.csr_matrix(x)
			else:
				x=[]
				for document in corpus:
					vect_list=[]
					for word in document.split():
						if word in d:
							vect_list.append(np.array(d[word]))
						else:
							vect_list.append(np.zeros(300))
					x.append(np.mean(np.array(vect_list),axis=0))
				x=np.array(x)
				return sparse.csr_matrix(x)


def accuracies(list1,list2,test_X,test_Y):
	print("BOW:")

	train_input,test_input = bow_func(list1,list2)
	print("SVM:")
	svm_func(train_input,test_input,test_X,test_Y)
	print("LRF:")
	logisticregression_func(train_input,test_input,test_X,test_Y)
	print("FFOR")
	feedforw_func(train_input,test_input,test_X,test_Y)
	print("naive")
	naive_func(train_input,test_input,test_X,test_Y)

	print("normalized tf:")
	train_input,test_input =normalized_tf_func(list1,list2)
	print("SVM:")
	svm_func(train_input,test_input,test_X,test_Y)
	print("LRF:")
	logisticregression_func(train_input,test_input,test_X,test_Y)
	print("FFOR")
	feedforw_func(train_input,test_input,test_X,test_Y)
	print("naive")
	naive_func(train_input,test_input,test_X,test_Y)

	print("tfidf")

	train_input,test_input =tfidf_func(list1,list2)
	print("SVM:")	
	svm_func(train_input,test_input,test_X,test_Y)
	print("LRF:")
	logisticregression_func(train_input,test_input,test_X,test_Y)
	print("FFOR")
	feedforw_func(train_input,test_input,test_X,test_Y)
	print("naive")
	naive_func(train_input,test_input,test_X,test_Y)



	print("word2vec_func no weights")
	train_input=word2vec_func(list1,0)
	test_input=word2vec_func(list2,0)

	print("SVM:")
	svm_func(train_input,test_input,test_X,test_Y)
	print("LRF:")
	logisticregression_func(train_input,test_input,test_X,test_Y)
	print("FFOR")
	feedforw_func(train_input,test_input,test_X,test_Y)
	print("naive")
	naive_func(train_input,test_input,test_X,test_Y)

	print("word2vec_func with  weights")
	train_input=word2vec_func(list1,1)
	test_input=word2vec_func(list2,1)

	print("SVM:")
	svm_func(train_input,test_input,test_X,test_Y)
	print("LRF:")
	logisticregression_func(train_input,test_input,test_X,test_Y)
	print("FFOR")
	feedforw_func(train_input,test_input,test_X,test_Y)
	print("naive")
	naive_func(train_input,test_input,test_X,test_Y)

	print("GLOVE no weights")
	train_input=glove_func(list1,0)
	test_input=glove_func(list2,0)
	print("SVM:")
	svm_func(train_input,test_input,test_X,test_Y)
	print("LRF:")
	logisticregression_func(train_input,test_input,test_X,test_Y)
	print("FFOR")
	feedforw_func(train_input,test_input,test_X,test_Y)
	print("naive")
	naive_func(train_input,test_input,test_X,test_Y)

	print("GLOVE weights")
	train_input=glove_func(list1,1)
	test_input=glove_func(list2,1)
	print("SVM:")
	svm_func(train_input,test_input,test_X,test_Y)
	print("LRF:")
	logisticregression_func(train_input,test_input,test_X,test_Y)
	print("FFOR")
	feedforw_func(train_input,test_input,test_X,test_Y)
	print("naive")
	naive_func(train_input,test_input,test_X,test_Y)







directory = os.listdir("aclImdb/test/neg")
c=0	
list1=[]
output=[]
for file in directory:
	
	f=open("aclImdb/test/neg/"+file,"r")
	f=f.read()
	list1.append(f)
	output.append(0)


c=0
directory = os.listdir("aclImdb/test/pos")
for file in directory:
	f=open("aclImdb/test/pos/"+file,"r")
	f=f.read()
	list1.append(f)
	output.append(1)
 
train_output=np.array(output)




list2=[]
output1=[]

directory = os.listdir("aclImdb/train/neg")
c=0
for file in directory:
	f=open("aclImdb/train/neg/"+file,"r")
	f=f.read()
	list2.append(f)
	output1.append(0)

directory = os.listdir("aclImdb/train/pos")


for file in directory:
	f=open("aclImdb/train/pos/"+file,"r")
	f=f.read()
	list2.append(f)
	output1.append(1)


test_output=np.array(output1)
list1,list2=stopword_removal(list1,list2)
accuracies(list1,list2,train_output,test_output)






