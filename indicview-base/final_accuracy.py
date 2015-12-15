import numpy as np

f = open('final_results.txt').readlines()
f = [f[i].strip().split('%')[0] for i in range(len(f))]


for i in range(len(f)):
	if '-' in f[i]:
		f[i] = 0
	else:
		f[i] = float(f[i])

f = np.asarray(f)
print f.mean(), np.sqrt(f.var()), max(f), min(f)