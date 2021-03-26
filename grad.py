import numpy as np

def f(x):
    """
    :param x: np.array(np.float) размерности 2
    :return: float
    """
    return np.sum(np.sin(x)**2, axis=0)


def grad_f(x):
    """
    Градиент функциии из учебного ноутбука.
    :param x: np.array размерности 2
    :return: np.array(np.float) размерности 2
    """
    return 2 * np.sin(x) * np.cos(x)


def grad_descent_2d(f, grad_f, lr, num_iter=100, x0=None):
    """
    функция, которая реализует градиентный спуск в минимум для функции f от двух переменных. 
        :param f: скалярная функция двух переменных
        :param grad_f: градиент функции f (вектор размерности 2)
        :param lr: learning rate алгоритма
        :param num_iter: количество итераций градиентного спуска
        :return: np.array пар вида (x, f(x))
    """
    if x0 is None:
        x0 = np.random.random(2)
    history = []
    curr_x = x0.copy()
    for iter_num in range(num_iter):
        entry = np.hstack((curr_x, f(curr_x)))
        history.append(entry)
        curr_x -= curr_x * lr

    return np.vstack(history)
