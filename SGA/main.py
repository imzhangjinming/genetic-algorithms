import argparse
import numpy as np
from sga import generate_population, crossover, mutation
from test_func import f

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

    config = argparser.parse_args()
    config = vars(config)
    return config

if __name__ == '__main__':
    config = get_args()
    init_population = generate_population(config)
    children = crossover(config, init_population)
    m_children = mutation(config, children, 1)
    f_vals = f(m_children)
    print(init_population)
    print(children)
    print(m_children==children)
    print(f_vals)