import sys
import numpy as np
import matplotlib.pyplot as plt

test_level = 0
aveP = np.empty(30)
aveZ = np.empty(30)
varP = np.empty(30)
varZ = np.empty(30)
kurt = np.empty(30)

ex_level = 0
eps = np.empty(30)
mlmc = np.empty(30)
std = np.empty(30)
N_level = np.empty(10)
N = np.empty((10, 30))

dir = sys.argv[1]
with open('./' + dir + '/_output.txt', 'r') as f:
    line = f.readline()
    test = True

    while line:
        line = line.strip()
        for _ in range(100):
            line = line.replace('  ', ' ')
        line = line.split(' ')

        if line[0] == 'l':
            f.readline()
            f.readline()
        elif line[0] == 'eps':
            f.readline()
        elif line[0] == '' or line[0] == 'alpha' or line[0] == 'beta' or line[0] == 'gamma' or line[0] == 'EVPI':
            test = False
        else:
            if test:
                aveZ[test_level] = float(line[1])
                aveP[test_level] = float(line[2])
                varZ[test_level] = float(line[3])
                varP[test_level] = float(line[4])
                kurt[test_level] = float(line[5])
                test_level += 1
            else:
                eps[ex_level] = float(line[0])
                mlmc[ex_level] = float(line[2])
                std[ex_level] = float(line[3])
                N_level[ex_level] = len(line) - 6
                for i in range(len(line) - 6):
                    N[ex_level][i] = int(line[i + 6])

                ex_level += 1

        line = f.readline()

fig = plt.figure(figsize=(14, 9))

level = np.array([i + 1 for i in range(test_level)])

ax = fig.add_subplot(231)
ax.plot(level, varP[:test_level], marker='s', label='Pl')
ax.plot(level, varZ[:test_level], marker='D', label='Zl')
ax.set_yscale('log', basey=2)
ax.set_title('log2 variance')
ax.legend()

ax = fig.add_subplot(232)
ax.plot(level, aveP[:test_level], marker='s', label='Pl')
ax.plot(level, aveZ[:test_level], marker='D', label='Zl')
ax.set_title('log2 mean')
ax.set_yscale('log', basey=2)
ax.legend()

ax = fig.add_subplot(233)
ax.plot(level, kurt[:test_level], marker='s')
ax.set_title('kurtosis')

ax = fig.add_subplot(234)
markers = ['o', 's', 'X', 'D', '^', 'v']
for i in range(ex_level):
    level = np.array([i + 1 for i in range(int(N_level[i]))])
    ax.plot(level, N[i][:int(N_level[i])], marker=markers[i], label=str(eps[i]))
ax.set_title('N')
ax.set_yscale('log', basey=10)
ax.legend()

ax = fig.add_subplot(235)
mlmc_cost = np.array([eps[i] * eps[i] * mlmc[i] for i in range(ex_level)])
std_cost = np.array([eps[i] * eps[i] * std[i] for i in range(ex_level)])
ax.plot(eps[:ex_level], mlmc_cost, marker='s', label='MLMC')
ax.plot(eps[:ex_level], std_cost, marker='D', label='Std MC')
ax.set_title('e^2 Cost')
ax.set_yscale('log', basey=10)
ax.set_xscale('log', basex=10)
ax.set_xlim(eps[ex_level - 1], eps[0])
ax.legend()

plt.show()