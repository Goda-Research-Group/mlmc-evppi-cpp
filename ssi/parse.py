
import numpy as np

d = [[] for _ in range(3)]
with open('./NMA_all.txt', 'r') as f:
    line = f.readline()
    while line:
        x = line.split()
        d[0].append(-float(x[1]))
        d[1].append(float(x[2]) - float(x[1]))
        d[2].append(float(x[3]) - float(x[1]))
        line = f.readline()

print([np.mean(d[x]) for x in range(3)])
print(np.cov(d))
