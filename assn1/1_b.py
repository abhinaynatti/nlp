import re
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
f=open("file_2.txt",'r')
file=f.read()

def vector_generator(sentence,c):
	window_size=4
	feat=np.zeros(128)
	#print(feat,'\tinsdide func \n')
	for x in range(c-window_size,c+window_size+1):
		if(x>=0 and x<len(sentence)):
			feat[ord(sentence[x])]=1

	#print(feat,'\tafter forloop\n')
	return feat

#sentence=re.findall(r'(?<=<s>).*(?=<s/>)',file)
lines=re.findall(r'<s>(.*?)<\/s>',file)
#print(sentence[0])
input_array=[]
output_array=[]
for sentence in lines:
	c=0
	for letter in sentence:
		c=c+1
		output=0
		#print(letter,'\t',sentence[-1])
		#print(vector_generator(sentence,c),'\n')
		#print(letter,'\n',feature_vector,'\n')
		if(letter == sentence[-1]):
			output=1
		input_array.append(vector_generator(sentence,c))
		output_array.append(output)
		#break
	#break


#print(np.shape(input_array),'\n')
#print(np.shape(output_array))
limit = 25000
X_train = input_array[:limit]
Y_train = output_array[:limit]
X_test = input_array[limit:]
Y_test = output_array[limit:]
mlp=MLPClassifier(hidden_layer_sizes=(80,20))
mlp.fit(X_train, Y_train)
predictions = mlp.predict(X_test)
# print(predictions)
print(mlp.score(X_test, Y_test))	