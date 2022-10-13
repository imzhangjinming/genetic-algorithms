import numpy as np

'''
测试函数的具体定义参见
[1] DEB K, PRATAP A, AGARWAL S, 等. A fast and elitist multiobjective genetic algorithm: NSGA-II[J/OL]. IEEE Transactions on Evolutionary Computation, 2002, 6(2): 182-197. DOI:10.1109/4235.996017.

'''

'''
f1 + f2 是SCH测试函数
'''
def f1(input):
    '''
    测试函数 f1 = x^2
    '''
    return np.square(input)

def f2(input):
    '''
    测试函数 f2 = (x-2)^2
    '''    
    return np.square(input-2)

'''
f3 + f4 是KUR测试函数
'''
def f3(input):
    '''
    测试函数 f3 = \sum_{i=1}^{n-1}\left(-10 \exp \left(-0.2 \sqrt{x_{i}^{2}+x_{i+1}^{2}}\right)\right)
    '''
    output = -10*np.exp(-0.2*np.sqrt(np.square(input[:,0])+np.square(input[:,1])))  - 10*np.exp(-0.2*np.sqrt(np.square(input[:,1])+np.square(input[:,2])))
    
    return output

def f4(input):
    '''
    测试函数 f4 = \sum_{i=1}^{n}\left(\left|x_{i}\right|^{0.8}+5 \sin x_{i}^{3}\right)
    '''
    output = np.power(np.abs(input[:, 0]), 0.8) + 5*np.sin(np.power(input[:, 0], 3)) + np.power(np.abs(input[:, 1]), 0.8) + 5*np.sin(np.power(input[:, 1], 3)) + np.power(np.abs(input[:, 2]), 0.8) + 5*np.sin(np.power(input[:, 2], 3))

    return output