import numpy as np

def regression(x, y):
    sum_x2 = 0
    sum_xy = 0
    for i in range(len(x)):
        sum_x2 += x[i] ** 2
        sum_xy += x[i] * y[i]
    return (len(x) * sum_xy - np.sum(x) * np.sum(y)) / (len(x) * sum_x2 - np.sum(x) * np.sum(x))

def log2_regression(y, level):
    x = np.zeros(level)
    log2_y = np.zeros(level)
    for i in range(level):
        x[i] = i + 1
        log2_y[i] = - np.log2(y[i + 1])
    return regression(x, log2_y)

def pre_sampling(params):
    params[0] = np.random.normal(0, 1, 1)
    return params

def post_sampling(params, m):
    params[1] = np.random.normal(0, 1, m)
    return params

def calc(params, scr):
    return scr * (params[0] + params[1])

n_samples = np.zeros(30)
n_samples_done = np.zeros(30)
aveZ = np.zeros(30)
aveP = np.zeros(30)
varZ = np.zeros(30)
varP = np.zeros(30)
kurt = np.zeros(30)
result = np.zeros((30, 6))
alpha = 0.96
beta = 1.43

def evppi_calc(level):
    global result

    m = pow(2, level)
    params = np.zeros((2, m))
    params = pre_sampling(params)
    params = post_sampling(params, m)

    if level == 0:
        sum_of_max = 0
        sum_of_no_scr = 0
        sum_of_scr = 0

        for i in range(m):
            no_scr = calc(params[:, i], scr = 0)
            scr = calc(params[:, i], scr = 1)

            sum_of_max += max(no_scr, scr)
            sum_of_no_scr += no_scr
            sum_of_scr += scr

        sum_of_max /= m
        max_of_sum = max(sum_of_no_scr, sum_of_scr) / m
        p = sum_of_max - max_of_sum

        result[level][0] += p
        result[level][1] += p * p
        result[level][2] += p
        result[level][3] += p * p
        result[level][4] += p * p * p
        result[level][5] += p * p * p * p

    else:
        sum_of_max = 0
        sum_of_no_scr_first = 0
        sum_of_scr_first = 0
        sum_of_no_scr_second = 0
        sum_of_scr_second = 0

        for i in range(int(m / 2)):
            no_scr = calc(params[:, i], scr = 0)
            scr = calc(params[:, i], scr = 1)

            sum_of_max += max(no_scr, scr)
            sum_of_no_scr_first += no_scr
            sum_of_scr_first += scr

        for i in range(int(m / 2), m):
            no_scr = calc(params[:, i], scr = 0)
            scr = calc(params[:, i], scr = 1)

            sum_of_max += max(no_scr, scr)
            sum_of_no_scr_second += no_scr
            sum_of_scr_second += scr

        sum_of_max /= m
        max_of_sum = max(sum_of_no_scr_first + sum_of_no_scr_second, sum_of_scr_first + sum_of_scr_second) / m
        p = sum_of_max - max_of_sum
        result[level][0] += p
        result[level][1] += p * p

        max_of_sum_first = max(sum_of_no_scr_first, sum_of_scr_first) / int(m / 2)
        max_of_sum_second = max(sum_of_no_scr_second, sum_of_scr_second) / (m - int(m / 2))
        z = (max_of_sum_first + max_of_sum_second) / 2 - max_of_sum
        result[level][2] += z
        result[level][3] += z * z
        result[level][4] += z * z * z
        result[level][5] += z * z * z * z

def mlmc_calc(level):
    global n_samples
    global n_samples_done
    global aveZ
    global aveP
    global varZ
    global varP
    global result
    global kurt

    for l in range(level + 1):
        for _ in range(int(n_samples[l])):
            evppi_calc(l)

        n_samples_done[l] += n_samples[l]
        n_samples[l] = 0
        n = n_samples_done[l]
        n_samples_done[l] = n
        aveZ[l] = result[l][2] / n
        aveP[l] = result[l][0] / n
        varZ[l] = result[l][3] / n - aveZ[l] * aveZ[l]
        varP[l] = result[l][1] / n - aveP[l] * aveP[l]
        if l > 0:
            kurt[l] = (result[l][5] / n - 4 * result[l][4] / n * aveZ[l] + 6 * result[l][3] / n * aveZ[l] ** 2 - 3 * aveZ[l] ** 4) / varZ[l] ** 2


def mlmc_test():
    global n_samples
    global n_samples_done
    global aveZ
    global aveP
    global varZ
    global varP
    global kurt
    global alpha
    global beta

    print('l aveZ aveP varZ varP kurt check')
    print('-----------------------------')

    with open('./output.txt', 'w') as f:
        f.write('l aveZ aveP varZ varP kurt check\n')
        f.write('-----------------------------\n')

    aveZ = np.zeros(30)
    aveP = np.zeros(30)
    varZ = np.zeros(30)
    varP = np.zeros(30)
    kurt = np.zeros(30)
    n_samples = np.ones(30) * 2000
    n_samples_done = np.zeros(30)

    for l in range(11):
        result = np.zeros((30, 6))
        mlmc_calc(l)
        print(str(l) + ' ' + str(aveZ[l]) + ' ' + str(aveP[l]) + ' ' + str(varZ[l]) + ' ' + str(varP[l]) + ' ' + str(kurt[l]))
        with open('./output.txt', 'a') as f:
            f.write(str(l) + ' ' + str(aveZ[l]) + ' ' + str(aveP[l]) + ' ' + str(varZ[l]) + ' ' + str(varP[l]) + ' ' + str(kurt[l]) + '\n')

    alpha = log2_regression(aveZ, 10)
    beta = log2_regression(varZ, 10)
    print('alpha = ' + str(alpha))
    print('beta = ' + str(beta))

    with open('./output.txt', 'a') as f:
        f.write('\n')
        f.write('alpha = ' + str(alpha) + '\n')
        f.write('beta  = ' + str(beta) + '\n\n')

def eval_eps(e, level):
    global n_samples
    global n_samples_done
    global aveZ
    global varZ
    global alpha
    global beta

    cost = np.array([pow(2, l) for l in range(30)])
    converged = False
    while not converged:
        mlmc_calc(level)

        for l in range(2, level + 1):
            aveZ_min = 0.5 * aveZ[l - 1] / pow(2, alpha);
            if aveZ_min > aveZ[l]:
                aveZ[l] = aveZ_min

            varZ_min = 0.5 * varZ[l - 1] / pow(2, beta);
            if varZ_min > varZ[l]:
                varZ[l] = varZ_min

        sm = 0
        for l in range(level + 1):
            sm += np.sqrt(varZ[l] * cost[l])

        converged = True
        for l in range(level + 1):
            n_samples[l] = np.ceil(max(0, np.sqrt(varZ[l] / cost[l]) * sm / (0.75 * e * e) - n_samples_done[l]))
            if n_samples[l] > 0.01 * n_samples_done[l]:
                converged = False

        if converged:
            rem = aveZ[level] / (pow(2, alpha) - 1)
            if rem > e * 0.5:
                if level == 29:
                    print('level over')
                    exit()
                else:
                    converged = False
                    level += 1
                    varZ[level] = varZ[level - 1] / pow(2, beta)
                    sm += np.sqrt(varZ[level] * cost[level])
                    for l in range(level + 1):
                        n_samples[l] = np.ceil(max(0, np.sqrt(varZ[l] / cost[l]) * sm / (0.75 * e * e) - n_samples_done[l]))

    return level

def eps_test(eps):
    global n_samples
    global n_samples_done
    global aveZ
    global aveP
    global varZ
    global varP

    print("eps value mlmc std save N...")
    print("------------------------------------------------------------------")

    with open('./output.txt', 'a') as f:
        f.write('eps value mlmc std save N...\n')
        f.write('-----------------------------\n')

    n_samples = np.array([1000 if level < 3 else 0 for level in range(30)])
    n_samples_done = np.zeros(30)
    result = np.zeros((30, 6))
    aveZ = np.zeros(30)
    aveP = np.zeros(30)
    varZ = np.zeros(30)
    varP = np.zeros(30)
    cost = np.array([pow(2, l) for l in range(30)])

    level = 2
    for e in eps:
        level = eval_eps(e, level)

        value = 0
        mlmc_cost = 0
        for l in range(level + 1):
            value += aveZ[l]
            mlmc_cost += n_samples_done[l] * cost[l]
        std_cost = varP[level] * cost[level] / (0.75 * e * e)

        print(str(e) + ' ' + str(value) + ' ' + str(mlmc_cost) + ' '+ str(std_cost) + ' ' + str(std_cost / mlmc_cost), end = ' ')
        for l in range(level + 1):
            print(int(n_samples_done[l]), end = ' ')
        print('')

        with open('./output.txt', 'a') as f:
            f.write(str(e) + ' ' + str(value) + ' ' + str(mlmc_cost) + ' '+ str(std_cost) + ' ' + str(std_cost / mlmc_cost) + ' ')
            for l in range(level + 1):
                f.write(str(int(n_samples_done[l])) + ' ')
            f.write('\n')

if __name__ == '__main__':
    mlmc_test()
    eps_test(np.array([0.002, 0.001, 0.0005, 0.0002, 0.0001]))