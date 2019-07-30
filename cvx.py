# -*- coding: UTF-8 -*
import numpy as np
import pandas as pd
import math

from cvxopt.blas import dot  # package for optimization
from cvxopt import matrix, solvers

lambda1 = 0.5;
lambda2 = 0.5;


# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
def init(context):
    context.stks = ["000056.XSHE", "601088.XSHG", "600556.XSHG", "000049.XSHE", "002104.XSHE", "600804.XSHG",
                    "000757.XSHE", "000858.XSHE", "600578.XSHG", "002470.XSHE"]


# 你选择的证券的数据更新将会触发此段逻辑，例如日或分钟历史数据切片或者是实时数据切片更新
def handle_bar(context, bar_dict):
    Sigma_tau = EstimateCovarianceMatrix(context.stks)

    W_old = []
    for stk in context.stks:
        W_old.append(context.portfolio.positions[stk].quantity * bar_dict[stk].last)
    W_old = np.divide(W_old, context.portfolio.portfolio_value)
    [W_new, status] = DRPRM(Sigma_tau, W_old)
    k = 0
    if status == 'optimal':
        for stk in context.stks:
            if bar_dict[stk].is_trading:
                order_target_percent(stk, W_new[k])
            k = k + 1
    else:
        logger.info("Optimization Failed.")


def EstimateCovarianceMatrix(stks):
    tau = 100
    OpenPrice = []
    ClosePrice = []
    for stk in stks:
        op = history(tau, '1d', 'open')[stk].values
        cp = history(tau, '1d', 'close')[stk].values
        OpenPrice.append(op)
        ClosePrice.append(cp)

    ratio = np.divide(ClosePrice, OpenPrice)

    CovarianceMatrix = np.cov(ratio)

    return CovarianceMatrix


def DRPRM(Sigma, W_old):
    N = len(Sigma)
    Q = 2 * matrix(Sigma) + 2 * lambda2 * matrix(np.eye(N))

    P = lambda1 * matrix(np.ones(N), (N, 1)) - 2 * lambda2 * matrix(W_old, (N, 1))

    G = -1 * matrix(np.eye(N))
    h = matrix(np.zeros(N), (N, 1))

    A = matrix(np.ones(N), (1, N))
    b = matrix(1.0)

    sol = solvers.qp(Q, P, G, h, A, b)
    '''
    According to [1], we obtain the new portfolio weight by solving
    min w^T*Sigma*w + lambda1*norm(w,1)+lambda2*norm(w-w_old,2)
    s.t w >= 0
        sum(w) = 1

    Using the fact that w >=0, the original problem can be reformulated as a qudratic progamming
    '''

    return sol['x'], sol['status']