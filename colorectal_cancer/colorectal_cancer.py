import numpy as np
from scipy.linalg import expm

lambda_asr = np.array([
    0.003269711, 0.003530327, 0.003792993, 0.004052078, 0.004317881,
    0.004599621, 0.004922169, 0.005309455, 0.005781141, 0.006331949,
    0.006955423, 0.007629364, 0.008340312, 0.009080023, 0.009872359,
    0.010786443, 0.011838922, 0.012979695, 0.014178268, 0.015473447,
    0.016880352, 0.018547232, 0.020472230, 0.022630014, 0.024950558,
    0.027424561, 0.030211367, 0.033370463, 0.037086502, 0.041297081,
    0.045886804, 0.050882598, 0.056705728, 0.063777195, 0.071586555,
    0.080427769, 0.090940935, 0.102714297, 0.115824094, 0.130373073,
    0.146459453, 0.164173422, 0.183592933, 0.204779190, 0.227771802,
    0.252584036, 0.279198097, 0.307561326, 0.337583002, 0.369133107
])

ages = np.arange(50) + 50
lambda_g_mean = np.array([0.4323036, 1.0694100])
lambda_g_cov = np.array([
    [0.005081143, -0.0010271285],
    [-0.001027128, 0.0002115364]
])
discount = [1 / (1.03 ** x) for x in range(51)]
p_screen          = np.array([1 if x % 10 == 0       and x < 90 else 0 for x in range(50, 100)])
p_surveillance_lr = np.array([1 if x % 5 == 0        and x < 90 else 0 for x in range(50, 100)])
p_surveillance_hr = np.array([1 if (x - 50) % 3 == 0 and x < 86 else 0 for x in range(50, 100)])

def pre_sampling(params):
    params[0], params[1] = np.random.multivariate_normal(lambda_g_mean, lambda_g_cov, 1).T ** 15 # lambda1, g
    params[7] = 1 - np.exp(- params[0] * 50 ** params[1]) # prev adenoma
    return params

def post_sampling(params, m):
    params[2] = np.exp(np.random.normal(-3.4538776, 0.58739416, m))         # lambda2
    params[3] = np.exp(np.random.normal(-3.9120230, 0.35364652, m))         # lambda3
    params[4] = np.exp(np.random.normal(-1.1512925, 0.23374764, m))         # lambda4
    params[5] = np.exp(np.random.normal(-1.4067054, 0.10343498, m))         # lambda5
    params[6] = np.exp(np.random.normal(-0.7803239, 0.21614741, m))         # lambda6
    params[8] = np.random.beta(6.132642, 2.504882, m)                       # prop adenoma sm
    params[9] = np.random.beta(373.5, 110.2, m)                             # sens of small adenoma
    params[10] = np.random.beta(2475, 378, m)                               # spec of small adenoma
    params[11] = np.random.beta(59.3, 1.2, m)                               # sens of large adenoma
    params[12] = np.random.beta(2475, 378, m)                               # spec of large adenoma
    params[13] = np.exp(np.random.normal(0.6931472, 0.2802582, m))          # hazard ratio for low risk
    params[14] = np.exp(np.random.normal(1.098612, 0.1768233, m))           # hazard ratio for high risk
    params[15] = 1 - np.exp(np.random.normal(-4.60517, 1.351612, m))        # utility of normal
    params[16] = 1 - np.exp(np.random.normal(-1.931022, 0.2802582, m))      # utility of clinical crc early
    params[17] = 1 - np.exp(np.random.normal(-0.3566749, 0.103435, m))      # utility of clinical crc late
    params[18] = np.exp(np.random.normal(9.21034, 0.0511915, m))            # cost of colonoscopy
    params[19] = np.exp(np.random.normal(9.976924, 0.03565356, m)) * 1.2356 # cost of cancer early
    params[20] = np.exp(np.random.normal(10.51867, 0.02760551, m)) * 1.2356 # cost of cancer late
    return params

def crc_nhm_tp(params):
    # transition rate matrix (intensity matrix)
    # 0: "Normal",
    # 1: "SmallAdeno",
    # 2: "LargeAdeno",
    # 3: "PreClinCRC_Early",
    # 4: "PreClinCRC_Late",
    # 5: "ClinCRC_Early",
    # 6: "ClinCRC_Late",
    # 7: "CRC_Death",
    # 8: "OC_Death"

    transition = np.zeros((9, 9, 50))
    lambda_1_t = params[0] * params[1] * ages ** (params[1] - 1)
    transition[0][0] = - lambda_1_t - lambda_asr
    transition[0][1] = lambda_1_t
    transition[0][8] = lambda_asr
    transition[1][1] = - params[2] - lambda_asr
    transition[1][2] = params[2]
    transition[1][8] = lambda_asr
    transition[2][2] = - params[3] - lambda_asr
    transition[2][3] = params[3]
    transition[2][8] = lambda_asr
    transition[3][3] = - params[4] - params[5] - lambda_asr
    transition[3][4] = params[4]
    transition[3][5] = params[5]
    transition[3][8] = lambda_asr
    transition[4][4] = - params[6] - lambda_asr
    transition[4][6] = params[6]
    transition[4][8] = lambda_asr
    transition[5][5] = - 0.0302 - lambda_asr
    transition[5][7] = 0.0302
    transition[5][8] = lambda_asr
    transition[6][6] = - 0.2099 - lambda_asr
    transition[6][7] = 0.2099
    transition[6][8] = lambda_asr

    # intensity matrix -> probability matrix
    for i in range(50):
        transition[:, :, i] = expm(transition[:, :, i])

    # check
    # for i in range(50):
    #     for j in range(9):
    #         if abs(1 - np.sum(transition[j, :, i])) > 1e-8:
    #             print('crc_nhm_tp matrix is not valid')
    #             exit()

    return transition

def crc_nhm_hist_tp(params, hist):
    # hist: Indicator variable selecting history for low risk (=1) or high risk (=0)

    lambda_1_t = params[0] * params[1] * ages ** (params[1] - 1)
    hist = params[13] * hist + params[14] * (1 - hist)

    # transition rate matrix (intensity matrix)
    transition = np.zeros((10, 10, 50))
    transition[0][0] = - lambda_1_t - lambda_asr
    transition[0][2] = lambda_1_t
    transition[0][9] = lambda_asr
    transition[1][1] = - lambda_1_t * hist - lambda_asr
    transition[1][2] = lambda_1_t * hist
    transition[1][9] = lambda_asr
    transition[2][2] = - params[2] - lambda_asr
    transition[2][3] = params[2]
    transition[2][9] = lambda_asr
    transition[3][3] = - params[3] - lambda_asr
    transition[3][4] = params[3]
    transition[3][9] = lambda_asr
    transition[4][4] = - params[4] - params[5] - lambda_asr
    transition[4][5] = params[4]
    transition[4][6] = params[5]
    transition[4][9] = lambda_asr
    transition[5][5] = - params[6] - lambda_asr
    transition[5][7] = params[6]
    transition[5][9] = lambda_asr
    transition[6][6] = - 0.0302 - lambda_asr
    transition[6][8] = 0.0302
    transition[6][9] = lambda_asr
    transition[7][7] = - 0.2099 - lambda_asr
    transition[7][8] = 0.2099
    transition[7][9] = lambda_asr

    # intensity matrix -> probability matrix
    for i in range(50):
        transition[:, :, i] = expm(transition[:, :, i])

    # check
    # for i in range(50):
    #     for j in range(9):
    #         if abs(1 - np.sum(transition[j, :, i])) > 1e-8:
    #             print('crc_nhm_hist_tp matrix is not valid')
    #             exit()

    return transition

def crc_screening_tp(params, scr):
    screen = p_screen * scr
    surveillance_lr = p_surveillance_lr * scr
    surveillance_hr = p_surveillance_hr * scr

    true_pos = params[9]           # sens COL
    false_neg = 1 - params[9]      # sens COL
    true_neg = params[10]          # spec COL
    false_pos = 1 - params[10]     # spec COL
    true_pos_crc = params[11]      # sens col crc
    false_neg_crc = 1 - params[11] # sens col crc

    transition_nhm = crc_nhm_tp(params)
    transition_hist_lr = crc_nhm_hist_tp(params, hist = 1)
    transition_hist_hr = crc_nhm_hist_tp(params, hist = 0)
    transition_scr = np.zeros((19, 19, 50))
    # "Normal",
    # "Normal_Hist_LR",
    # "Normal_Hist_HR",
    # "SmallAdeno",
    # "SmallAdeno_LR",
    # "SmallAdeno_HR",
    # "LargeAdeno",
    # "LargeAdeno_LR",
    # "LargeAdeno_HR",
    # "PreClinCRC_Early",
    # "PreClinCRC_Early_LR",
    # "PreClinCRC_Early_HR",
    # "PreClinCRC_Late",
    # "PreClinCRC_Late_LR",
    # "PreClinCRC_Late_HR",
    # "ClinCRC_Early",
    # "ClinCRC_Late",
    # "CRC_Death",
    # "OC_Death"

    no_hist = np.array([0, 3, 6, 9, 12, 15, 16, 17, 18])
    lr_hist = np.array([1, 4, 7, 10, 13, 15, 16, 17, 18])
    hr_hist = np.array([2, 5, 8, 11, 14, 15, 16, 17, 18])

    # from normal
    for i in range(len(no_hist)):
        transition_scr[0][no_hist[i]] = \
            (1 - screen) * transition_nhm[0][i] + \
            screen * true_neg * transition_nhm[0][i] + \
            screen * false_pos * transition_nhm[0][i]

    # from normal hist lr
    for i in range(len(lr_hist)):
        transition_scr[1][lr_hist[i]] = \
            (1 - surveillance_lr) * transition_hist_lr[1][i + 1] + \
            surveillance_lr * true_neg * transition_hist_lr[1][i + 1] + \
            surveillance_lr * false_pos * transition_hist_lr[1][i + 1]

    # from normal hist hr
    for i in range(len(hr_hist)):
        transition_scr[2][hr_hist[i]] = \
            (1 - surveillance_hr) * transition_hist_hr[1][i + 1] + \
            surveillance_hr * true_neg * transition_hist_hr[1][i + 1] + \
            surveillance_hr * false_pos * transition_hist_hr[1][i + 1]

    # from small adenoma
    for i in range(len(no_hist)):
        transition_scr[3][no_hist[i]] = \
            (1 - screen) * transition_nhm[1][i] + screen * false_neg *  transition_nhm[1][i]
    transition_scr[3][1] = screen * true_pos

    # from small adenoma lr
    for i in range(len(lr_hist)):
        transition_scr[4][lr_hist[i]] = \
            (1 - surveillance_lr) * transition_hist_lr[2][i + 1] + surveillance_lr * false_neg * transition_hist_lr[2][i + 1]
    transition_scr[4][1] = surveillance_lr * true_pos

    # from small adenoma hr
    for i in range(len(hr_hist)):
        transition_scr[5][hr_hist[i]] = \
            (1 - surveillance_hr) * transition_hist_hr[2][i + 1] + surveillance_hr * false_neg * transition_hist_hr[2][i + 1]
    transition_scr[5][1] = surveillance_hr * true_pos

    # from large adenoma
    for i in range(len(no_hist)):
        transition_scr[6][no_hist[i]] = \
            (1 - screen) * transition_nhm[2][i] + screen * false_neg_crc * transition_nhm[2][i]
    transition_scr[6][2] = screen * true_pos_crc

    # from large adenoma lr
    for i in range(len(hr_hist)):
        transition_scr[7][hr_hist[i]] = \
            (1 - surveillance_lr) * transition_hist_lr[3][i + 1] + surveillance_lr * false_neg_crc * transition_hist_lr[3][i + 1]
    transition_scr[7][2] = surveillance_lr * true_pos_crc

    # from large adenoma hr
    for i in range(len(hr_hist)):
        transition_scr[8][hr_hist[i]] = \
            (1 - surveillance_hr) * transition_hist_hr[3][i + 1] + surveillance_hr * false_neg_crc * transition_hist_hr[3][i + 1]
    transition_scr[8][2] = surveillance_hr * true_pos_crc

    # from pre clinic crc early
    for i in range(len(no_hist)):
        transition_scr[9][no_hist[i]] = \
            (1 - screen) * transition_nhm[3][i] + screen * false_neg_crc * transition_nhm[3][i]
    transition_scr[9][15] += screen * true_pos_crc

    # from pre clinic crc early lr
    for i in range(len(lr_hist)):
        transition_scr[10][lr_hist[i]] = \
            (1 - surveillance_lr) * transition_hist_lr[4][i + 1] + surveillance_lr * false_neg_crc * transition_hist_lr[4][i + 1]
    transition_scr[10][15] += surveillance_lr * true_pos_crc

    # from pre clinic crc early hr
    for i in range(len(hr_hist)):
        transition_scr[11][hr_hist[i]] = \
            (1 - surveillance_hr) * transition_hist_hr[4][i + 1] + surveillance_hr * false_neg_crc * transition_hist_hr[4][i + 1]
    transition_scr[11][15] += surveillance_hr * true_pos_crc

    # from pre clinic crc late
    for i in range(len(no_hist)):
        transition_scr[12][no_hist[i]] = \
            (1 - screen) * transition_nhm[4][i] + screen * false_neg_crc * transition_nhm[4][i]
    transition_scr[12][16] += screen * true_pos_crc

    # from pre clinic crc late lr
    for i in range(len(lr_hist)):
        transition_scr[13][lr_hist[i]] = \
            (1 - surveillance_lr) * transition_hist_lr[5][i + 1] + surveillance_lr * false_neg_crc * transition_hist_lr[5][i + 1]
    transition_scr[13][16] += surveillance_lr * true_pos_crc

    # from pre clinic crc late hr
    for i in range(len(hr_hist)):
        transition_scr[14][hr_hist[i]] = \
            (1 - surveillance_hr) * transition_hist_hr[5][i + 1] + surveillance_hr * false_neg_crc * transition_hist_hr[5][i + 1]
    transition_scr[14][16] += surveillance_hr * true_pos_crc

    # from clinic crc early
    for i in range(len(no_hist)):
        transition_scr[15][no_hist[i]] = transition_nhm[5][i]

    # from clinic crc late
    for i in range(len(no_hist)):
        transition_scr[16][no_hist[i]] = transition_nhm[6][i]

    # from crc death
    transition_scr[17][17] = 1

    # from other cause of death
    transition_scr[18][18] = 1

    # cheeck
    # for i in range(50):
    #     for j in range(19):
    #         if abs(1 - np.sum(transition_scr[j, :, i])) > 1e-8:
    #             print('crc_screening_tp matrix is not valid')
    #             exit()

    return transition_scr

def crc_screening(params, scr):
    init = np.array([
        0.998 - params[7],
        0, 0,
        params[7] * params[8],
        0, 0,
        params[7] * (1 - params[8]),
        0, 0,
        0.0012,
        0, 0,
        0.0008,
        0, 0, 0, 0, 0, 0
    ])

    transition = crc_screening_tp(params, scr)

    trace = np.zeros((51, 19))
    trace[0] = init
    for i in range(50):
        trace[i + 1] = np.dot(trace[i], transition[:, :, i])

    utilities = np.array([
        params[15], params[15], params[15],
        params[15], params[15], params[15],
        params[15], params[15], params[15],
        params[15], params[15], params[15],
        params[15], params[15], params[15], # utility of normal
        params[16], # utility of clinical crc early
        params[17], # utility of clinical crc late
        0, 0
    ])

    costs = np.array([
        np.append(p_screen, 0)          * scr * params[18],
        np.append(p_surveillance_lr, 0) * scr * params[18],
        np.append(p_surveillance_hr, 0) * scr * params[18],
        np.append(p_screen, 0)          * scr * params[18],
        np.append(p_surveillance_lr, 0) * scr * params[18],
        np.append(p_surveillance_hr, 0) * scr * params[18],
        np.append(p_screen, 0)          * scr * params[18],
        np.append(p_surveillance_lr, 0) * scr * params[18],
        np.append(p_surveillance_hr, 0) * scr * params[18],
        np.append(p_screen, 0)          * scr * params[18],
        np.append(p_surveillance_lr, 0) * scr * params[18],
        np.append(p_surveillance_hr, 0) * scr * params[18],
        np.append(p_screen, 0)          * scr * params[18],
        np.append(p_surveillance_lr, 0) * scr * params[18],
        np.append(p_surveillance_hr, 0) * scr * params[18],
        np.array([params[19] for _ in range(51)]),
        np.array([params[20] for _ in range(51)]),
        np.zeros(51),
        np.zeros(51)
    ]).T

    utility = np.sum(np.dot(trace, utilities) * discount)
    cost = np.sum(np.sum(trace * costs, axis = 1) * discount)

    return np.array([utility, cost])

def calc(params, scr):
    ret = crc_screening(params, scr = scr)
    return ret[0] * 75000 - ret[1]

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

n_samples = np.zeros(30)
n_samples_done = np.zeros(30)
aveZ = np.zeros(30)
aveP = np.zeros(30)
varZ = np.zeros(30)
varP = np.zeros(30)
kurt = np.zeros(30)
result = np.zeros((30, 6))
alpha = 0.918167
beta  = 1.36201

def evppi_calc(level):
    global result

    m = pow(2, level)
    params = np.zeros((21, m))
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

    for l in range(8):
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

def sml_calc(n):
    sum_of_max = 0.0
    sum_of_scr = 0.0
    sum_of_no_scr = 0.0

    for _ in range(n):
        params = np.zeros((21, 1))
        params = pre_sampling(params)
        params = post_sampling(params, 1)

        no_scr = calc(params[:, 0], scr = 0)
        scr = calc(params[:, 0], scr = 1)

        sum_of_max += max(no_scr, scr)
        sum_of_no_scr += no_scr
        sum_of_scr += scr

    max_of_sum = max(sum_of_scr, sum_of_no_scr)
    print("evpi = " + str((sum_of_max - max_of_sum) / n))

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

    n_samples = np.array([100 if level < 3 else 0 for level in range(30)])
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
    sml_calc(10000)
    mlmc_test()
    eps_test(np.array([100, 50, 20, 10, 5]))