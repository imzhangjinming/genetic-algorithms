'''
简单遗传算法
'''
import numpy as np
import copy
from test_func import f1

def generate_population(config):
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


def crossover(config, population):
    '''
    算术交叉算子
    输入：
        size : 种群规模
        l    : 单条染色体的长度
        pc   : 交叉概率
        alpha: 算术交叉的参数
        population: 种群
    '''  
    size, l, pc, alpha = config['size'], config['l'], config['pc'], config['alpha']
    random_sequence = list(range(size))
    np.random.shuffle(random_sequence) # 将染色体序号随机排列

    child_list = copy.deepcopy(population)
    for idx in range(round(size/2)):
        crossover_prob = np.random.rand()
        if crossover_prob < pc:
            parent_1 = population[random_sequence[2*idx], :]
            parent_2 = population[random_sequence[2*idx+1], :]
            child_1 = alpha * parent_1 + (1.0 - alpha) * parent_2
            child_2 = (1.0 - alpha) * parent_1  + alpha * parent_2
            child_list[random_sequence[2*idx], :] = child_1
            child_list[random_sequence[2*idx+1], :] = child_2
    return child_list

def mutation(config, population, current_itr):
    '''
    变异算子
    参考：管小艳. 实数编码下遗传算法的改进及其应用[D/OL]. 重庆大学, 2012.

    输入：
        size        : 种群规模
        l           : 单条染色体的长度
        pm          : 变异概率
        lb          : 基因的下界
        ub          : 基因的上界
        r           : 变异算子的参数
        b           : 变异算子的参数
        iteration   : 总迭代次数
        population  : 种群  
        current_itr : 当前迭代次数  
    '''
    size, l, pm, lb, ub, r, b = config['size'], config['l'], config['pm'], config['lb'], config['ub'], config['r'], config['b']
    iteration = config['iteration']
    random_seq = np.random.randint(0, l, size=size)

    m_population = copy.deepcopy(population)
    for idx in range(size):
        mutation_prob = np.random.rand()
        if mutation_prob < pm:
            s_t = 1 - np.power(r, np.power(1 - current_itr/iteration, b))
            x_k = m_population[idx, random_seq[idx]]
            mutation_lower = x_k - s_t * (x_k - lb)
            mutation_upper = x_k + s_t * (ub - x_k)
            m_population[idx, random_seq[idx]] = np.random.uniform(mutation_lower, mutation_upper)

    return m_population

def fitness(population, f):
    return f(population)

def selection(config, population, fitness_vals):
    '''
    选择
    输入：
        size        : 种群规模
        S           : parameter in elitism preservation
        population  : 现有种群
        fitness_vals: 现有种群对应的适应度值
    '''
    size, S = config['size'], config['S']
    # fitness_vals = fitness(population, f1)

    sort_idx = np.argsort(fitness_vals)
    selected = copy.deepcopy(population[sort_idx[:size-S], :])

    return selected
