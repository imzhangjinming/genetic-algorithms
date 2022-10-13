import argparse
import numpy as np
from nsga2 import init_pop, fast_non_dominated_sort, crowding_distance_assignment, simulated_binary_crossover, tournament_selection, polynomial_mutation
from test_func import f1, f2, f3, f4
import copy
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def get_args():
    '''
    获取参数
    '''
    argparser = argparse.ArgumentParser(description='Simple Genetic Algorithm')
    argparser.add_argument('--size', default=150, type=int, help='size of population')
    argparser.add_argument('--iteration', default=250, type=int, help='number of iterations')
    argparser.add_argument('-pc', default=0.5, type=float, help='probability of crossover')
    argparser.add_argument('-pm', default=1/3, type=float, help='probability of mutation')
    argparser.add_argument('-l', default=3, type=int, help='length of chromosome')
    argparser.add_argument('-lb', default=-5, type=float, help='lower bound of gene')
    argparser.add_argument('-ub', default=5, type=float, help='upper bound of gene')
    argparser.add_argument('-etac', default=20, type=float, help='distribution index of simulated binary crossover')
    argparser.add_argument('-etam', default=20, type=float, help='distribution index of polynomial mutation')

    config = argparser.parse_args()
    config = vars(config)
    return config

if __name__ == '__main__':
    config = get_args()
    iteration = config['iteration']
    objs = [f3, f4]

    history = [] # 迭代历史
    population = init_pop(config)
    for itr in range(iteration):
        # 交叉
        children = simulated_binary_crossover(config, population)

        # 变异
        m_children = polynomial_mutation(config, children)

        # 将子代加入种群
        population = np.concatenate((copy.deepcopy(population), copy.deepcopy(m_children)), axis=0)

        # 非支配排序
        F = fast_non_dominated_sort(config, population, objs)

        # 选择
        population = tournament_selection(config, population, F, objs)

        history.append(population)

    last_population = history[-1]
    f1_vals = f1(last_population)
    f2_vals = f2(last_population)
    plt.figure()
    plt.scatter(f1_vals, f2_vals)
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.show()
    input()

