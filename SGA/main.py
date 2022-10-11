import argparse
import numpy as np
from sga import generate_population, crossover, mutation, fitness, selection
from test_func import f1
import copy
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def get_args():
    '''
    获取参数
    '''
    argparser = argparse.ArgumentParser(description='Simple Genetic Algorithm')
    argparser.add_argument('--size', default=150, type=int, help='size of population')
    argparser.add_argument('--iteration', default=100, type=int, help='number of iterations')
    argparser.add_argument('-pc', default=0.9, type=float, help='probability of crossover')
    argparser.add_argument('-pm', default=0.02, type=float, help='probability of mutation')
    argparser.add_argument('-l', default=2, type=int, help='length of chromosome')
    argparser.add_argument('-lb', default=-10, type=float, help='lower bound of gene')
    argparser.add_argument('-ub', default=10, type=float, help='upper bound of gene')
    argparser.add_argument('--alpha', default=0.2, type=float, help='alpha parameter in crossover operator')
    argparser.add_argument('-r', default=0.5, type=float, help='r parameter in mutation operator')
    argparser.add_argument('-b', default=3, type=float, help='b parameter in mutation operator')
    argparser.add_argument('-S', default=2, type=float, help='parameter in elitism preservation')

    config = argparser.parse_args()
    config = vars(config)
    return config

if __name__ == '__main__':
    config = get_args()
    iteration = config['iteration']
    S = config['S']

    history = [] # 迭代历史
    population = generate_population(config)
    for itr in range(iteration):
        # 精英保存策略
        fitness_vals = fitness(population, f1)
        sort_idx = np.argsort(fitness_vals)
        elitism = copy.deepcopy(population[sort_idx[:S], :])

        # 将最好的适应值和染色体保存到历史记录中
        history.append((fitness_vals[sort_idx[0]], population[sort_idx[0], :]))
        
        # 交叉
        children = crossover(config, population)

        # 变异
        m_children = mutation(config, children, itr)

        # 将子代加入种群
        population = np.concatenate((copy.deepcopy(population), copy.deepcopy(m_children)), axis=0)

        # 选择
        fitness_vals = fitness(population, f1)
        population = selection(config, population, fitness_vals)

        # 将保存的精英加入新种群
        population = np.concatenate((copy.deepcopy(population), copy.deepcopy(elitism)), axis=0)

    best_fitness_vals = []
    best_chromosome = np.zeros((len(history), config['l']))
    for idx, item in enumerate(history):
        best_fitness_vals.append(item[0])
        best_chromosome[idx, :] = item[1]
    
    plt.figure()
    x = np.arange(iteration) + 1
    plt.plot(x, best_fitness_vals)
    plt.xlabel('iteration')
    plt.ylabel('fitness value')
    plt.show()
    input()