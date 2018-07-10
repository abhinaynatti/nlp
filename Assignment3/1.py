import re
f = open( './en_ewt-ud-train.conllu', 'r')
f= f.read()
a = f.split('\n')
b = ''
for line in a:
	if(len(line)==0):
		b=b+'\n'
	elif(isdigit(line[0])):
		b=b+line+'\n'
	elif(line[0] in ['#']):
		b=b+'\n'
b = re.sub(r'\n\n+',r'\n\n',b)
print(b)