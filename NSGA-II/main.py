import argparse
import numpy as np
from nsga2 import init_pop, fast_non_dominated_sort, crowding_distance_assignment, simulated_binary_crossover, tournament_selection, polynomial_mutation
from test_func import test_function
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
    argparser.add_argument('--iteration', default=100, type=int, help='number of iterations')
    argparser.add_argument('-pc', default=0.9, type=float, help='probability of crossover')
    argparser.add_argument('-pm', default=0.02, type=float, help='probability of mutation')
    argparser.add_argument('-l', default=2, type=int, help='length of chromosome')
    argparser.add_argument('-lb', default=-10, type=float, help='lower bound of gene')
    argparser.add_argument('-ub', default=10, type=float, help='upper bound of gene')

    config = argparser.parse_args()
    config = vars(config)
    return config

if __name__ == '__main__':
    pass