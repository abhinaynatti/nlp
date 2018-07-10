import re
f=open("A1.txt",'r')
line= f.read()
line=re.sub(r'(?<=[a-z])(\')(?=[a-z])',r'$',line)
# quote=re.findall(r'\'(([^\']|(\\.))*)\'(?=\s)',line)
#line=re.sub(r'\'([.\n]*?)\'',r'"\1"',line)
line = re.sub(r'\'(([^\']|(\\.))*)\'',r'"\1"',line)
# for i in quote:
#   	print(i[0])
line = re.sub(r'\$',r'\'',line)
line = re.sub(r'\\',r'',line)
print(line)