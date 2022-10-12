import numpy as np


def f(input):
    '''
    六峰驼背函数
        f(x, y) = (4 - 2.1 * x^2 + 1/3 * x^4) * x^2 + (-4 + 4 * y^2) * y^2 + x * y, -10 <= x, y <= 10
    输入：
        input: n x 
    '''
    x = input[:, 0]
    y = input[:, 1]
    f_val = (4 - 2.1 * np.square(x) + 1.0/3.0 * np.power(x, 4)) * np.square(x) + (-4 + 4 * np.square(y)) * np.square(y) + x * y

    return f_val

def f2(input):
    '''
    Rastrigin 2D
    '''
    x = input[:, 0]
    y = input[:, 1]
    f_val = np.square(x) + np.square(y) - 10 * (np.cos(2*np.pi*x)+np.cos(2*np.pi*y)) + 20

    return f_val

def f1(input):
    '''
    Schaffer
    '''
    x = input[:, 0]
    y = input[:, 1]
    f_val = (np.square(np.sin(np.sqrt(np.square(x)+np.square(y)))) - 0.5) / np.square(1+0.001*(np.square(x)+np.square(y))) - 0.5

    return f_val  