import os
import gensim
import numpy as np
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

def move(list1,list2,y,head):

	if(y==[[1,0,0]]):
		list1.append(list2.pop(0))
	elif(y==[[0,1,0]]):#right arc
		head[list1[-2]].append(list1[-1])
		list1.pop(-1)
	elif(y==[[0,0,1]]):
		head[list1[-1]].append(list1[-2])
		list1.pop(-2)



def parser(classifier,path):
	print("loading model ...")
	with open("./glove.6B.300d.txt","r") as f:
			vocab={}
			for line in f:
				line=line.split()
				y=[]
				for i in line[1:]:
					y.append(float(i))
				vocab[line[0]]=np.array(y)
	print("loaded model ...")
	f=open(path,'r')
	f=f.read()
	Xtr = np.array([])
	train = f.split('\n\n')
	i=0
	j=0
	Y = []
	X = []
	print(len(train))
	for i,line in enumerate(train):
		print("train",i)
		words = line.split('\n')
		a = np.zeros([len(words)+1,300])
		tree = {}
		head={}
		for x in range(0,len(words)+1):
			tree[x] = []
			head[x] = []
		for word in words:
			word = word.split('\t')
			try:
				a[int(word[0])] = vocab[word[2]]
			except:
				tree={}
				i=i+1
				break
			try:
				tree[int(word[6])].append(int(word[0]))
			except ValueError:
				tree={}
				i=i+1
				break
		if tree == {}:
			continue
		temp = tree
		s = [0,1]
		b = [x for x in range(2,len(words)+1)]
		c=0
		x=[]
		
		while len(s)!=1 or len(b)!=0:
			c=c+1
			x=[]
			if(len(s)==1):
				s1 = a[s[-1]]
				s2 = np.zeros(300)
				b1 = a[b[0]]
				if(len(b)>=2):
					b2 = a[b[1]]
				else:
					b2 = np.zeros(300)
				try:
					ls11 = a[tree[s[-1]][0]]
				except:
					ls11 = np.zeros(300)
				x.append(np.concatenate([s1,s2,b1,b2]))
				X=np.array(x)
				y=clf.predict(X)
				y=y.tolist()
				if(y!=[1,0,0]):
					move(s,b,[1,0,0],head)
				else:
					move(s,b,y,head)		
			else:
				s1 = a[s[-1]]
				s2 = a[s[-2]]
				if(len(b)>=2):
					b1 = a[b[0]]
					b2 = a[b[1]]
				elif(len(b)==1):
					b1 = a[b[0]]
					b2 = np.zeros(300)
				else:
					b1 = np.zeros(300)
					b2 = np.zeros(300)
				x=[]
				x.append(np.concatenate([s1,s2,b1,b2]))
				X=np.array(x)
				y=clf.predict(X)
				y=y.tolist()
				move(s,b,y,head)

		print(head)		
		

def loaddata(path,path2):
	print("loading model ...")
	with open("./glove.6B.300d.txt","r") as f:
			vocab={}
			for line in f:
				line=line.split()
				y=[]
				for i in line[1:]:
					y.append(float(i))
				vocab[line[0]]=np.array(y)



	print("loaded model ...")
	f=open(path,'r')
	f=f.read()
	Xtr = np.array([])
	train = f.split('\n\n')
	i=0
	j=0
	Y = []
	X = []
	print("start")
	for line in train:
		words = line.split('\n')
		a = np.zeros([len(words)+1,300])
		tree = {}
		for x in range(0,len(words)+1):
			tree[x] = []
		for word in words:
			word = word.split('\t')
			try:
				a[int(word[0])] = vocab[word[2]]
			except:
				tree={}
				i=i+1
				break
			try:
				tree[int(word[6])].append(int(word[0]))
			except ValueError:
				tree={}
				i=i+1
				break
		if tree == {}:
			continue
		temp = tree
		s = [0,1]
		b = [x for x in range(2,len(words)+1)]
		y = []
		x = []
		while len(s)!=1 or len(b)!=0:
			if(len(s)==1):
				y.append([1,0,0])
				s1 = a[s[-1]]
				s2 = np.zeros(300)
				b1 = a[b[0]]
				if(len(b)>=2):
					b2 = a[b[1]]
				else:
					b2 = np.zeros(300)
				try:
					ls11 = a[tree[s[-1]][0]]
				except:
					ls11 = np.zeros(300)
				x.append(np.concatenate([s1,s2,b1,b2]))
				s.append(b.pop(0))
			elif(s[-1] in tree[s[-2]] and len(temp[s[-1]]) == 0):
				y.append([0,1,0]) # right arc
				s1 = a[s[-1]]
				s2 = a[s[-2]]
				if(len(b)>=2):
					b1 = a[b[0]]
					b2 = a[b[1]]
				elif(len(b)==1):
					b1 = a[b[0]]
					b2 = np.zeros(300)
				else:
					b1 = np.zeros(300)
					b2 = np.zeros(300)
				x.append(np.concatenate([s1,s2,b1,b2]))
				temp[s[-2]].remove(s[-1])
				s.pop(-1)
			elif(s[-2] in tree[s[-1]] and len(temp[s[-2]]) == 0):
				y.append([0,0,1])
				s1 = a[s[-1]]
				s2 = a[s[-2]]
				if(len(b)>=2):
					b1 = a[b[0]]
					b2 = a[b[1]]
				elif(len(b)==1):
					b1 = a[b[0]]
					b2 = np.zeros(300)
				else:
					b1 = np.zeros(300)
					b2 = np.zeros(300)
				x.append(np.concatenate([s1,s2,b1,b2]))
				temp[s[-1]].remove(s[-2])
				s.pop(-2)
			else:
				try:
					y.append([1,0,0])
					s1 = a[s[-1]]
					s2 = a[s[-2]]
					if(len(b)>=2):
						b1 = a[b[0]]
						b2 = a[b[1]]
					elif(len(b)==1):
						b1 = a[b[0]]
						b2 = np.zeros(300)
					else:
						b1 = np.zeros(300)
						b2 = np.zeros(300)
					x.append(np.concatenate([s1,s2,b1,b2]))
					s.append(b.pop(0))
				except IndexError:
					i=i+1
					y=[]
					break
		if(y!=[]):
			for i in range(len(y)):
				Y.append(y[i])
				X.append(x[i])
	print(len(train))
	print(i)
	print(len(X))
	print(len(Y))
	print(X[0])
	print(Y[0])
	X = np.array(X)
	Y = np.array(Y)
	X,Y = shuffle(X,Y)
	k = int(0.1*len(Y))
	Xtr = X[k:]
	Xts = X[:k]
	Ytr = Y[k:]
	Yts = Y[:k]
	clf = MLPClassifier(verbose = True)
	clf.fit(Xtr,Ytr)
	print(clf.score(Xts,Yts))
	# file=open('filename.pkl','w')
	# joblib.dump(clf, 'filename.pkl')
	parser(clf,path2)




basepath=os.getcwd()
path='modified.txt'
path2='test.txt'
path=basepath+'/'+path
# loaddata(path,path2)
clf=joblib.load('filename.pkl')

parser(clf,path2)



