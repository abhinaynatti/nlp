import re
f=open("A2.txt",'r')
line= f.read()
line=re.sub(r'\n\n[\n]*',r' [%]\n\n',line)
line=re.sub(r'\n',r' ',line)
#print(line)
#line=re.sub(r'\n\n',r'',line)
'''sentences=re.findall(r'((([^\n]*?((?<!Mr)\.|\?|\!|\'|\;))(?=\s[\']?[A-Z]|\[%\])))',line)
for i in sentences:
	print(i)
	'''
line=re.sub(r'((([^\n]*?((?<!Mr)(?<![A-Z])\.|\?|\!|\'|\;))(?=\s([\']?([A-Z])|\[%\]))))',r'<s>\3</s>',line)
line=re.sub	(r'\[%\]',r'',line)
print(line)