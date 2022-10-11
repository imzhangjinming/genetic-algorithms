import numpy as np


def f1(input):
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