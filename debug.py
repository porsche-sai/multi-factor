# coding: utf-8


import threading
from copy import deepcopy

import pandas as pd
import statsmodels.api as sm
from CAL.PyCAL import *
from lib.self_factors import get_factor_data
from quartz.universe import set_universe


class Cache(object):
    """
    用于对象存取
    """

    @classmethod
    def from_cache(cls, path):
        """
        读取硬盘中的对象

        Parameters
        ----------
        path: str, 对象存放路径

        Returns
        -------
        object: 对象
        """
        return pd.read_pickle(path)

    def save_obj(self, path=None):
        """
        存放对象

        Parameters
        ----------
        path: str or None, 存放路径。若为None, 则以类名+当前时间, 存放于当前路径

        Returns
        -------

        """
        if path is None:
            path = '%s instance %s' % (self.__class__, pd.to_datetime('now'))
        pd.to_pickle(self, path)
        print '%s instance saved' % self.__class__

    def copy(self):
        return deepcopy(self)


class Recoder(Cache):
    def __init__(self, w0, expo, xrm, R, V, crm, w_last, w_halt, threshold, bound, result):
        self.w0 = w0
        self.expo = expo
        self.xrm = xrm
        self.R = R
        self.V = V
        self.crm = crm
        self.w_last = w_last
        self.w_halt = w_halt
        self.threshold = threshold
        self.bound = bound
        self.result = result
