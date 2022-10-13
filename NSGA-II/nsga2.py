import numpy as np
import copy

def init_pop(config):
    '''
    随机生成初始种群
    输入：
        size: 初始种群规模
        l   : 单条染色体的长度
        lb  : 基因的下界
        ub  : 基因的上界
    '''
    size, l, lb, ub = config['size'], config['l'], config['lb'], config['ub']
    population = np.random.uniform(low=lb, high=ub+1e-7, size=(size, l))
    return population

def simulated_binary_crossover(config, population):
    '''
    模拟二进制交叉
    输入：
        size : 种群规模
        l    : 单条染色体的长度
        pc   : 交叉概率
        eta_c: 模拟二进制交叉的分布指数
        lb:     基因的下界
        ub:     基因的上界
        population: 种群
    ''' 
    size, l, pc, etac, lb, ub = config['size'], config['l'], config['pc'], config['etac'], config['lb'], config['ub']
    random_sequence = list(range(size))
    np.random.shuffle(random_sequence) # 将染色体序号随机排列

    child_list = copy.deepcopy(population)
    for idx in range(round(size/2)):
        crossover_prob = np.random.rand()
        if crossover_prob < pc:
            parent_1 = population[random_sequence[2*idx], :]
            parent_2 = population[random_sequence[2*idx+1], :]
            child_1 = copy.deepcopy(parent_1)
            child_2 = copy.deepcopy(parent_2)

            for idx_variable in range(l):
                var_random = np.random.rand()
                if var_random < 0.5:
                    u = np.random.rand()
                    y1 = min(parent_1[idx_variable], parent_2[idx_variable])
                    y2 = max(parent_1[idx_variable], parent_2[idx_variable])

                    beta = 1 + 2 * min(y1-lb, ub-y2) / (y2 - y1)
                    alpha = 2 - np.power(beta, -(etac+1))
                    if(u <= 1.0/alpha):
                        beta_q = np.power(u * alpha, 1.0 / (etac + 1))
                    else:
                        beta_q = np.power(1.0 / (2 - u*alpha), 1.0 / (etac + 1))

                    c1 = 0.5 * ((y1 + y2) - beta_q * np.abs(y2 - y1))
                    c2 = 0.5 * ((y1 + y2) + beta_q * np.abs(y2 - y1))

                    if (parent_1[idx_variable] < parent_2[idx_variable]):
                        child_1[idx_variable] = c1
                        child_2[idx_variable] = c2
                    else:
                        child_1[idx_variable] = c2
                        child_2[idx_variable] = c1

            child_list[random_sequence[2*idx], :] = child_1
            child_list[random_sequence[2*idx+1], :] = child_2

    return child_list

def polynomial_mutation(config, population, current_itr=0):
    '''
    多项式变异
    输入：
        size        : 种群规模
        l           : 单条染色体的长度
        pm          : 变异概率
        lb          : 基因的下界
        ub          : 基因的上界
        etam        : 多项式变异的分布指数
        iteration   : 总迭代次数
        population  : 种群  
        current_itr : 当前迭代次数  
    '''
    size, l, pm, lb, ub, etam = config['size'], config['l'], config['pm'], config['lb'], config['ub'], config['etam']
    iteration = config['iteration']
    random_seq = np.random.randint(0, l, size=size)

    m_population = copy.deepcopy(population)
    for idx in range(size):
        mutation_prob = np.random.rand()
        if mutation_prob < pm:
            u = np.random.rand()           
            y_k = m_population[idx, random_seq[idx]]
            delta = min(y_k - lb, ub - y_k) / (ub - lb)
            if (u < 0.5):
                delta_q = np.power(2*u+(1-2*u)*np.power(1-delta, etam+1), 1.0/(etam+1)) - 1
            else:
                delta_q = 1 - np.power(2*(1-u)+2*(u-0.5)*np.power(1-delta, etam+1), 1.0/(etam+1))
            
            m_population[idx, random_seq[idx]] = y_k + delta_q * (ub - lb)

    return m_population

def is_dominate(p, q, objs):
    '''
    判断p是否支配q,也就是判断以p作为输入的所有目标函数值是不是都比以q作为输入的对应目标函数值优(小),是则返回True,否则返回False
    输入：
        p:      染色体p
        q:      染色体q
        objs:   目标函数组成的数组
    '''
    less_than = True
    strictly_less_than = False
    for func in objs:
        f_val_1 = func(p)
        f_val_2 = func(q)
        if(f_val_1 > f_val_2):
            return False
        if(f_val_1 < f_val_2):
            strictly_less_than = True
        
    return less_than & strictly_less_than

def fast_non_dominated_sort(config, population, objs):
    '''
    确定种群中的支配关系
    输入:
        config:     参数与配置
        population: 当前种群
        objs:       目标函数组成的数组
    输出:
        sort_result:排序结果，由集合组成的数组，集合在数组中的索引+1就是它的rank，集合里的元素是染色体在population中的索引
    '''
    S = []
    n = np.ones(population.shape[0]) * (-1)
    F = []
    rank = np.zeros(population.shape[0])
    F1 = [] # F的第一个元素
    for idx_p in range(population.shape[0]):
        S_p = []
        n_p = 0
        for idx_q in range(population.shape[0]):
            if(is_dominate(population[idx_p, :][None, :], population[idx_q, :][None, :], objs)):
                S_p.append(idx_q)
            elif(is_dominate(population[idx_q, :][None, :], population[idx_p, :][None, :], objs)):
                n_p += 1
        if(n_p == 0):
            rank[idx_p] = 1
            F1.append(idx_p)
        S.append(S_p)
        n[idx_p] = n_p
    F.append(F1)

    i = 1
    F_i = copy.deepcopy(F1)
    while len(F_i) != 0:
        Q = []
        for idx_p in F_i:
            for idx_q in S[idx_p]:
                n[idx_q] -= 1
                if(n[idx_q]==0):
                    rank[idx_q] = i + 1
                    Q.append(idx_q)
        i += 1
        F_i = copy.deepcopy(Q)
        if len(F_i) != 0:
            F.append(F_i)

    return F

def crowding_distance_assignment(config, nondominated_set, population, objs):
    '''
    计算集合中每个元素的拥挤距离
    '''
    l = len(nondominated_set)
    distance = np.zeros(l)
    for obj in objs:
        f_vals = obj(population[nondominated_set, :])
        sorted_idx = np.argsort(f_vals)
        distance[sorted_idx[0]] += np.inf
        distance[sorted_idx[-1]] += np.inf
        for idx in range(1, l-1):
            distance[sorted_idx[idx]] += (distance[sorted_idx[idx+1]] - distance[sorted_idx[idx-1]]) / (f_vals.max() - f_vals.min())
    return distance

def tournament_selection(config, population, F, objs):
    '''
    锦标赛选择
    输入:
        size:       种群规模
        population: 现有种群
        F:          非支配排序结果
    '''
    size = config['size']
    next_population_idx = []
    front_idx = 0
    for i, F_i in enumerate(F):
        front_idx = i
        if(len(next_population_idx) + len(F_i) <= size):
            for idx in F_i:
                next_population_idx.append(idx)
        else:
            break
    
    if(len(next_population_idx) < size):
        crowd_distance = crowding_distance_assignment(config, F[front_idx], population, objs)
        sorted_idx = np.argsort(crowd_distance)
        for idx in range(size-len(next_population_idx)):
            next_population_idx.append(F[front_idx][sorted_idx[idx].tolist()])

    return copy.deepcopy(population[next_population_idx, :])
