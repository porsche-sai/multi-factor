# -*- coding: UTF-8 -*

import pandas as pd
import numpy as np
import statsmodels.api as sm
from CAL.PyCAL import *
import datetime as dt
from numpy import abs
from numpy import log
from numpy import sign
from scipy.stats import rankdata
from quartz_extensions.MFHandler.SignalProcess import standardize, neutralize, winsorize

cal = Calendar('China.SSE')


def REVERSE(secID, tradeDate):
    data = DataAPI.MktStockFactorsOneDayGet(tradeDate=tradeDate,
                                            secID=secID,
                                            ticker=u"",
                                            field=u"secID,REVS20",
                                            pandas="1").set_index('secID')
    ret = data.iloc[:, -1].rename('REVERSE')
    return ret


def MOM_Est_EPS(secID, tradeDate):
    foward_20 = cal.advanceDate(tradeDate, '-20B', BizDayConvention.Preceding).strftime("%Y-%m-%d")
    foward_20_df = DataAPI.MktStockFactorsOneDayGet(tradeDate=foward_20,
                                                    secID=secID,
                                                    ticker=u"",
                                                    field=u"secID,FY12P",
                                                    pandas="1").set_index('secID')

    today_df = DataAPI.MktStockFactorsOneDayGet(tradeDate=tradeDate,
                                                secID=secID,
                                                ticker=u"",
                                                field=u"secID,FY12P",
                                                pandas="1").set_index('secID')

    return ((today_df - foward_20_df) / foward_20_df.abs()).rename(columns={'FY12P': 'MOM_Est_EPS'})


def YOY_EPS(secID, tradeDate):
    lastday = cal.advanceDate(tradeDate, '-1B', BizDayConvention.Preceding).strftime("%Y-%m-%d")
    last_df = DataAPI.MktStockFactorsOneDayGet(tradeDate=lastday,
                                               secID=secID,
                                               ticker=u"",
                                               field=u"secID,EPS",
                                               pandas="1").set_index('secID')
    today_df = DataAPI.MktStockFactorsOneDayGet(tradeDate=tradeDate,
                                                secID=secID,
                                                ticker=u"",
                                                field=u"secID,EPS",
                                                pandas="1").set_index('secID')
    return ((today_df - last_df) / last_df.abs()).rename(columns={'EPS': 'YOY_EPS'})


def MOM_ROE(secID, tradeDate):
    lastday = cal.advanceDate(tradeDate, '-1B', BizDayConvention.Preceding).strftime("%Y-%m-%d")
    last_df = DataAPI.MktStockFactorsOneDayGet(tradeDate=lastday,
                                               secID=secID,
                                               ticker=u"",
                                               field=u"secID,ROE",
                                               pandas="1").set_index('secID')
    today_df = DataAPI.MktStockFactorsOneDayGet(tradeDate=tradeDate,
                                                secID=secID,
                                                ticker=u"",
                                                field=u"secID,ROE",
                                                pandas="1").set_index('secID')
    return ((today_df - last_df) / last_df.abs()).rename(columns={'ROE': 'MOM_ROE'})


def RESISTANCE(secID, tradeDate, N=120):
    endDate = tradeDate
    beginDate = cal.advanceDate(endDate, '-%dB' % N, BizDayConvention.Preceding).strftime("%Y-%m-%d")

    data = DataAPI.MktEqudAdjGet(tradeDate=u"",
                                 secID=secID,
                                 ticker=u"",
                                 isOpen="",
                                 beginDate=beginDate,
                                 endDate=endDate,
                                 field=u"secID,tradeDate,closePrice,turnoverVol",
                                 pandas="1")
    Close = data.pivot('tradeDate', 'secID', 'closePrice')
    Volume = data.pivot('tradeDate', 'secID', 'turnoverVol')
    trade_len, num = Close.shape  # 时间长度和股票长度
    current_price = Close.iloc[-1]  # 当前价格
    pos = Close > current_price  # 价格在当前价格以上的索引
    w1 = np.log(current_price / (Close - current_price).abs())  # 距离权重
    # 当价格等于当前价时，会出现inf值，将其替换为最大值
    w1 = w1.replace(np.inf, w1.replace(np.inf, np.nan).max())
    w2 = (np.log(np.tile(np.arange(1, trade_len + 1), (num, 1))) / np.log(trade_len + 1)).T  # 时间权重
    res_sum = Volume * w1 * w2
    res = (res_sum * pos).sum() / res_sum.iloc[:-1].sum()  # True为1,False为0
    return res.rename('RESISTANCE')


def HTOS(secID, tradeDate):
    endDate = np.datetime64(tradeDate)
    beginDate = endDate - 100
    beginDate = ''.join(beginDate.astype(str).split('-'))
    endDate = ''.join(endDate.astype(str).split('-'))

    # 取股票池
    data = DataAPI.EquFloatShTenGet(secID=secID,
                                    ticker=u"",
                                    beginDate=beginDate,
                                    endDate=endDate,
                                    field=u"secID,holdPct,endDate",
                                    pandas="1")

    max_end = data.groupby(['secID'])['endDate'].max().reset_index()
    return data.merge(max_end, on=['secID', 'endDate'], how='right').groupby('secID')['holdPct'].sum().rename('HTOS')


def Dispersion(secID, tradeDate):
    endDate = ''.join(tradeDate.split('-'))
    beginDate = cal.advanceDate(endDate, '-%dB' % 20, BizDayConvention.Preceding).strftime("%Y%m%d")
    index_ret = DataAPI.MktIdxFactorDateRangeGet(secID=u"",
                                                 ticker=u"000300",
                                                 beginDate=beginDate,
                                                 endDate=endDate,
                                                 field=u"ChgPct",
                                                 pandas="1").T.values[0] / 100 + 1
    stk_data = DataAPI.MktEqudGet(tradeDate=u"",
                                  secID=secID,
                                  ticker="",
                                  beginDate=beginDate,
                                  endDate=endDate,
                                  isOpen="",
                                  field=u"tradeDate,secID,chgPct",
                                  pandas="1").pivot(index='tradeDate',
                                                    columns='secID',
                                                    values='chgPct').reindex(columns=secID)
    stk_ret = stk_data.T.values / 100 + 1
    PE_arr = []
    LFLO_arr = []
    for stk in secID:
        tmp = DataAPI.MktStockFactorsDateRangeGet(secID=stk,
                                                  ticker=u"",
                                                  beginDate=beginDate,
                                                  endDate=endDate,
                                                  field=u"secID,tradeDate,PE,LFLO",
                                                  pandas="1")
        PE_arr.append(tmp.pivot('tradeDate', 'secID', 'PE'))
        LFLO_arr.append(tmp.pivot('tradeDate', 'secID', 'LFLO'))
    PE = pd.concat(PE_arr, axis=1)
    LFLO = pd.concat(LFLO_arr, axis=1)
    PE = PE.apply(lambda x: standardize(neutralize(winsorize(x), endDate)), axis=1)
    LFLO = PE.apply(lambda x: standardize(neutralize(winsorize(x), endDate)), axis=1)
    stklist = PE.columns.tolist()

    # 因子合并，重排列。每个二维数组为单支股票的因子时间序列
    PE_LFLO = np.dstack([PE.values.T, LFLO.values.T])

    dis_arr = []
    for i, array in enumerate(PE_LFLO):
        X = np.column_stack([index_ret, array])  # 将指数收益向量与因子时间序列按列合并
        Y = stk_ret[i]  # 对应的个股收益率
        res = sm.OLS(Y, X).fit()  # 回归分析
        rsquared = res.rsquared  # R方
        resid = res.resid  # 残差数组
        dispersion = np.sqrt(1 - rsquared) * resid.std()  # 计算离散度
        dis_arr.append(dispersion)  # 添加至列表

    return pd.Series(dis_arr, index=stklist, name='Dispersion')


# region Auxiliary functions
def ts_sum(df, window=10):
    """
    Wrapper function to estimate rolling sum.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """

    return df.rolling(window).sum()


def sma(df, window=10):
    """
    Wrapper function to estimate SMA.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).mean()


def stddev(df, window=10):
    """
    Wrapper function to estimate rolling standard deviation.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).std()


def correlation(x, y, window=10):
    """
    Wrapper function to estimate rolling corelations.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return x.rolling(window).corr(y)


def covariance(x, y, window=10):
    """
    Wrapper function to estimate rolling covariance.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return x.rolling(window).cov(y)


def rolling_rank(na):
    """
    Auxiliary function to be used in pd.rolling_apply
    :param na: numpy array.
    :return: The rank of the last value in the array.
    """
    return rankdata(na)[-1]


def ts_rank(df, window=10):
    """
    Wrapper function to estimate rolling rank.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series rank over the past window days.
    """
    return df.rolling(window).apply(rolling_rank)


def rolling_prod(na):
    """
    Auxiliary function to be used in pd.rolling_apply
    :param na: numpy array.
    :return: The product of the values in the array.
    """
    return np.prod(na)


def product(df, window=10):
    """
    Wrapper function to estimate rolling product.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series product over the past 'window' days.
    """
    return df.rolling(window).apply(rolling_prod)


def ts_min(df, window=10):
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).min()


def ts_max(df, window=10):
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series max over the past 'window' days.
    """
    return df.rolling(window).max()


def delta(df, period=1):
    """
    Wrapper function to estimate difference.
    :param df: a pandas DataFrame.
    :param period: the difference grade.
    :return: a pandas DataFrame with today’s value minus the value 'period' days ago.
    """
    return df.diff(period)


def delay(df, period=1):
    """
    Wrapper function to estimate lag.
    :param df: a pandas DataFrame.
    :param period: the lag grade.
    :return: a pandas DataFrame with lagged time series
    """
    return df.shift(period)


def rank(df):
    """
    Cross sectional rank
    :param df: a pandas DataFrame.
    :return: a pandas DataFrame with rank along columns.
    """
    return df.rank(axis=1, pct=True)


def scale(df, k=1):
    """
    Scaling time serie.
    :param df: a pandas DataFrame.
    :param k: scaling factor.
    :return: a pandas DataFrame rescaled df such that sum(abs(df)) = k
    """
    return df.mul(k).div(np.abs(df).sum())


def ts_argmax(df, window=10):
    """
    Wrapper function to estimate which day ts_max(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return df.rolling(window).apply(np.argmax) + 1


def ts_argmin(df, window=10):
    """
    Wrapper function to estimate which day ts_min(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return df.rolling(window).apply(np.argmin) + 1


def decay_linear(df, period=10):
    """
    Linear weighted moving average implementation.
    :param df: a pandas DataFrame.
    :param period: the LWMA period
    :return: a pandas DataFrame with the LWMA.
    """
    # Clean data
    if df.isnull().values.any():
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(value=0, inplace=True)

    na_lwma = np.zeros_like(df)
    na_lwma[:period, :] = df.ix[:period, :]
    na_series = df.as_matrix()

    divisor = period * (period + 1) / 2
    y = (np.arange(period) + 1) * 1.0 / divisor
    # Estimate the actual lwma with the actual close.
    # The backtest engine should assure to be snooping bias free.
    for row in range(period - 1, df.shape[0]):
        x = na_series[row - period + 1: row + 1, :]
        na_lwma[row, :] = (np.dot(x.T, y))
    return pd.DataFrame(na_lwma, index=df.index, columns=df.columns)


# endregion

"""
t=[a for a in dir(aa) if a.startswith('alpha')] crea una lista con los nombres de los metodos
getattr(aa, t[2])() llama el metodo.
"""


class Alphas(object):
    def __init__(self, pn_data):
        """
        :type pn_data: pandas.Panel
        """
        data = pn_data
        self.secID = data.items.tolist()
        self.dates = data.major_axis
        self.beginDate = self.dates[0]
        self.tradeDate = self.dates[-1]
        self.open = data.minor_xs('openPrice')
        self.high = data.minor_xs('highestPrice')
        self.low = data.minor_xs('lowestPrice')
        self.close = data.minor_xs('closePrice')
        self.volume = data.minor_xs('turnoverVol')
        self.turnoverValue = data.minor_xs('turnoverValue')
        self.returns = self.close.pct_change()

    def alpha001(self):
        inner = self.close
        inner[self.returns < 0] = stddev(self.returns, 20)
        return rank(ts_argmax(inner ** 2, 5))

    def alpha002(self):
        df = -1 * correlation(rank(delta(log(self.volume), 2)), rank((self.close - self.open) / self.open), 6)
        return df.replace([-np.inf, np.inf], 0).fillna(value=0)

    def alpha003(self):
        df = -1 * correlation(rank(self.open), rank(self.volume), 10)
        return df.replace([-np.inf, np.inf], 0).fillna(value=0)

    def alpha004(self):
        return -1 * ts_rank(rank(self.low), 9)

    def alpha006(self):
        df = -1 * correlation(self.open, self.volume, 10)
        return df.replace([-np.inf, np.inf], 0).fillna(value=0)

    def alpha007(self):
        adv20 = sma(self.volume, 20)
        alpha = -1 * ts_rank(abs(delta(self.close, 7)), 60) * sign(delta(self.close, 7))
        alpha[adv20 >= self.volume] = -1
        return alpha

    def alpha008(self):
        return -1 * (rank(((ts_sum(self.open, 5) * ts_sum(self.returns, 5)) -
                           delay((ts_sum(self.open, 5) * ts_sum(self.returns, 5)), 10))))

    def alpha009(self):
        delta_close = delta(self.close, 1)
        cond_1 = ts_min(delta_close, 5) > 0
        cond_2 = ts_max(delta_close, 5) < 0
        alpha = -1 * delta_close
        alpha[cond_1 | cond_2] = delta_close
        return alpha

    def alpha010(self):
        delta_close = delta(self.close, 1)
        cond_1 = ts_min(delta_close, 4) > 0
        cond_2 = ts_max(delta_close, 4) < 0
        alpha = -1 * delta_close
        alpha[cond_1 | cond_2] = delta_close
        return alpha

    def alpha012(self):
        return sign(delta(self.volume, 1)) * (-1 * delta(self.close, 1))

    def alpha013(self):
        return -1 * rank(covariance(rank(self.close), rank(self.volume), 5))

    def alpha014(self):
        df = correlation(self.open, self.volume, 10)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * rank(delta(self.returns, 3)) * df

    def alpha015(self):
        df = correlation(rank(self.high), rank(self.volume), 3)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * ts_sum(rank(df), 3)

    def alpha016(self):
        return -1 * rank(covariance(rank(self.high), rank(self.volume), 5))

    def alpha017(self):
        adv20 = sma(self.volume, 20)
        return -1 * (rank(ts_rank(self.close, 10)) *
                     rank(delta(delta(self.close, 1), 1)) *
                     rank(ts_rank((self.volume / adv20), 5)))

    def alpha018(self):
        df = correlation(self.close, self.open, 10)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * (rank((stddev(abs((self.close - self.open)), 5) + (self.close - self.open)) +
                          df))

    def alpha019(self):
        return ((-1 * sign((self.close - delay(self.close, 7)) + delta(self.close, 7))) *
                (1 + rank(1 + ts_sum(self.returns, 250))))

    def alpha020(self):
        return -1 * (rank(self.open - delay(self.high, 1)) *
                     rank(self.open - delay(self.close, 1)) *
                     rank(self.open - delay(self.low, 1)))

    def alpha021(self):
        cond_1 = sma(self.close, 8) + stddev(self.close, 8) < sma(self.close, 2)
        cond_2 = sma(self.volume, 20) / self.volume < 1
        alpha = pd.DataFrame(np.ones_like(self.close), index=self.close.index,
                             columns=self.close.columns)
        alpha[cond_1 | cond_2] = -1
        return alpha

    def alpha022(self):
        df = correlation(self.high, self.volume, 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * delta(df, 5) * rank(stddev(self.close, 20))

    def alpha023(self):
        cond = sma(self.high, 20) < self.high
        alpha = pd.DataFrame(np.zeros_like(self.close), index=self.close.index,
                             columns=self.close.columns)
        alpha[cond] = -1 * delta(self.high, 2)
        return alpha

    def alpha024(self):
        cond = delta(sma(self.close, 100), 100) / delay(self.close, 100) <= 0.05
        alpha = -1 * delta(self.close, 3)
        alpha[cond] = -1 * (self.close - ts_min(self.close, 100))
        return alpha

    def alpha026(self):
        df = correlation(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * ts_max(df, 3)

    def alpha028(self):
        adv20 = sma(self.volume, 20)
        df = correlation(adv20, self.low, 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return scale(((df + ((self.high + self.low) / 2)) - self.close))

    def alpha029(self):
        return (ts_min(rank(rank(scale(log(ts_sum(rank(rank(-1 * rank(delta((self.close - 1), 5)))), 2))))), 5) +
                ts_rank(delay((-1 * self.returns), 6), 5))

    def alpha030(self):
        delta_close = delta(self.close, 1)
        inner = sign(delta_close) + sign(delay(delta_close, 1)) + sign(delay(delta_close, 2))
        return ((1.0 - rank(inner)) * ts_sum(self.volume, 5)) / ts_sum(self.volume, 20)

    def alpha031(self):
        adv20 = sma(self.volume, 20)
        df = correlation(adv20, self.low, 12)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(self.close, 10)))), 10)))) +
                 rank((-1 * delta(self.close, 3)))) + sign(scale(df)))

    def alpha033(self):
        return rank(-1 + (self.open / self.close))

    def alpha034(self):
        inner = stddev(self.returns, 2) / stddev(self.returns, 5)
        inner = inner.replace([-np.inf, np.inf], 1).fillna(value=1)
        return rank(2 - rank(inner) - rank(delta(self.close, 1)))

    def alpha035(self):
        return ((ts_rank(self.volume, 32) *
                 (1 - ts_rank(self.close + self.high - self.low, 16))) *
                (1 - ts_rank(self.returns, 32)))

    def alpha037(self):
        return rank(correlation(delay(self.open - self.close, 1), self.close, 200)) + rank(self.open - self.close)

    def alpha038(self):
        inner = self.close / self.open
        inner = inner.replace([-np.inf, np.inf], 1).fillna(value=1)
        return -1 * rank(ts_rank(self.open, 10)) * rank(inner)

    def alpha039(self):
        adv20 = sma(self.volume, 20)
        return ((-1 * rank(delta(self.close, 7) * (1 - rank(decay_linear(self.volume / adv20, 9))))) *
                (1 + rank(ts_sum(self.returns, 250))))

    def alpha040(self):
        return -1 * rank(stddev(self.high, 10)) * correlation(self.high, self.volume, 10)

    def alpha043(self):
        adv20 = sma(self.volume, 20)
        return ts_rank(self.volume / adv20, 20) * ts_rank((-1 * delta(self.close, 7)), 8)

    def alpha044(self):
        df = correlation(self.high, rank(self.volume), 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * df

    def alpha045(self):
        df = correlation(self.close, self.volume, 2)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * (rank(sma(delay(self.close, 5), 20)) * df *
                     rank(correlation(ts_sum(self.close, 5), ts_sum(self.close, 20), 2)))

    def alpha046(self):
        inner = ((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10)
        alpha = (-1 * delta(self.close))
        alpha[inner < 0] = 1
        alpha[inner > 0.25] = -1
        return alpha

    def alpha049(self):
        inner = (((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10))
        alpha = (-1 * delta(self.close))
        alpha[inner < -0.1] = 1
        return alpha

    def alpha051(self):
        inner = (((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10))
        alpha = (-1 * delta(self.close))
        alpha[inner < -0.05] = 1
        return alpha

    def alpha052(self):
        return (((-1 * delta(ts_min(self.low, 5), 5)) *
                 rank(((ts_sum(self.returns, 240) - ts_sum(self.returns, 20)) / 220))) * ts_rank(self.volume, 5))

    def alpha053(self):
        inner = (self.close - self.low).replace(0, 0.0001)
        return -1 * delta((((self.close - self.low) - (self.high - self.close)) / inner), 9)

    def alpha054(self):
        inner = (self.low - self.high).replace(0, -0.0001)
        return -1 * (self.low - self.close) * (self.open ** 5) / (inner * (self.close ** 5))

    def alpha055(self):
        divisor = (ts_max(self.high, 12) - ts_min(self.low, 12)).replace(0, 0.0001)
        inner = (self.close - ts_min(self.low, 12)) / (divisor)
        df = correlation(rank(inner), rank(self.volume), 6)
        return -1 * df.replace([-np.inf, np.inf], 0).fillna(value=0)

    def alpha060(self):
        divisor = (self.high - self.low).replace(0, 0.0001)
        inner = ((self.close - self.low) - (self.high - self.close)) * self.volume / divisor
        return - ((2 * scale(rank(inner))) - scale(rank(ts_argmax(self.close, 10))))

    def alpha059(self):
        return ts_sum((self.high - self.open) / self.open - (self.open - self.low) / self.open, 50)

    def alpha102(self):
        #         W_data = DataAPI.MktEqudAdjGet(secID=self.secID,
        #                               beginDate=self.beginDate,
        #                               endDate=self.tradeDate,
        #                               field=u"secID,tradeDate,turnoverValue",
        #                               pandas="1")

        #         W = W_data.pivot('tradeDate', 'secID', 'turnoverValue').reindex(columns=self.secID)  # 日成交额
        W = self.turnoverValue
        Q = self.volume  # 成交量
        B = (self.high + self.close + self.low + self.open).rolling(window=5).mean() / 4  # 每日均价
        VRBP = B.diff() / B.shift(1)

        # 股本数据
        share_data = DataAPI.EquShareGet(secID=self.secID,
                                         ticker=u"",
                                         beginDate=self.beginDate,
                                         endDate=self.tradeDate,
                                         partyID=u"",
                                         field=u"secID,changeDate,floatA",
                                         pandas="1")
        share_df = share_data.pivot('changeDate', 'secID', 'floatA').reindex(columns=self.secID)
        QF = share_df.ffill().reindex(self.close.index).ffill().bfill()  # 自由流通股数
        SPPI = (W - B * Q) / (QF * B.shift())

        sliced_VRBP = VRBP.iloc[-20:].reindex(columns=self.secID)
        sliced_SPPI = SPPI.iloc[-20:].reindex(columns=self.secID)
        Y_X = np.dstack([sliced_VRBP.values.T, sliced_SPPI.values.T])
        alpha = [sm.OLS(*pair.T).fit().params[0] for pair in Y_X]

        return pd.DataFrame(alpha, index=self.secID).T

    def alpha103(self):
        #         W_data = DataAPI.MktEqudAdjGet(secID=self.secID,
        #                               beginDate=self.beginDate,
        #                               endDate=self.tradeDate,
        #                               field=u"secID,tradeDate,turnoverValue",
        #                               pandas="1")

        #         W = W_data.pivot('tradeDate', 'secID', 'turnoverValue').reindex(columns=self.secID)  # 日成交额
        W = self.turnoverValue
        Q = self.volume  # 成交量
        B = (self.high + self.close + self.low + self.open).rolling(window=5).mean() / 4  # 每日均价
        VRBP = B.diff() / B.shift(1)

        # 股本数据
        share_data = DataAPI.EquShareGet(secID=self.secID,
                                         ticker=u"",
                                         beginDate=self.beginDate,
                                         endDate=self.tradeDate,
                                         partyID=u"",
                                         field=u"secID,changeDate,floatA",
                                         pandas="1")
        share_df = share_data.pivot('changeDate', 'secID', 'floatA').reindex(columns=self.secID)
        QF = share_df.ffill().reindex(self.close.index).ffill().bfill()  # 自由流通股数
        SPPI = (W - B * Q) / (QF * B.shift(1))

        sliced_VRBP = VRBP.iloc[-20:].reindex(columns=self.secID)
        sliced_SPPI = SPPI.iloc[-20:].reindex(columns=self.secID)
        sqrt_SPPI = np.sqrt(sliced_SPPI.abs()) * np.sign(sliced_SPPI)
        Y_X = np.dstack([sliced_VRBP.values.T, sqrt_SPPI.values.T])
        alpha = [sm.OLS(*pair.T).fit().params[0] for pair in Y_X]

        return pd.DataFrame(alpha, index=self.secID).T


def from_alpha101(secID, tradeDate, *alpha_name):
    begdate = cal.advanceDate(tradeDate, "-100B").strftime("%Y%m%d")
    data = DataAPI.MktEqudAdjGet(secID=secID,
                                 beginDate=begdate,
                                 endDate=tradeDate,
                                 field=u"secID,tradeDate,openPrice,closePrice,lowestPrice,highestPrice,turnoverVol,turnoverValue",
                                 pandas="1")
    pn_data = (data.set_index(['secID', 'tradeDate'])  # 将需要的index独立出来
               .stack()  # 堆叠
               .reset_index()
               .rename(columns={0: 'values',
                                'level_2': 'priceType'})
               .pivot_table(index=['tradeDate', 'priceType'],  # 创建数据透视表
                            columns='secID',
                            values='values')
               .to_panel())  # 转为panel
    obj = Alphas(pn_data)
    arr = []
    for alp in alpha_name:
        df = eval('obj.%s()' % alp)
        arr.append(df.iloc[-1].rename(alp))
    return pd.concat(arr, axis=1)


def get_factor_data(secID, tradeDate, field):
    df = DataAPI.MktStockFactorsOneDayProGet(tradeDate=tradeDate,
                                             secID=secID,
                                             ticker=u"",
                                             field=field,
                                             pandas="1").set_index('secID')
    if isinstance(field, str):
        field_list = [s.strip() for s in field.split(',')]
    else:
        field_list = field

    self_field = filter(lambda x: x not in df.columns and x != 'secID', field_list)
    arr = []
    alpha_factors = []
    for func in self_field:
        # try:
        #     # print func
        #     # print func in globals()
        #     assert func in globals()
        #     # print func
        #     self_df = eval(func)(secID, tradeDate)
        # except AssertionError:  # 若名字不存在，则有可能是alpha101中的因子。将名字储存，然后统一执行
        #     alpha_factors.append(func)
        #     continue
        if func in globals():
            self_df = eval(func)(secID, tradeDate)
        else:
            alpha_factors.append(func)
            continue
        arr.append(self_df)
    if len(alpha_factors):
        arr.append(from_alpha101(secID, tradeDate, *alpha_factors))

    reindex_col = [col for col in field_list if col != 'secID']
    return pd.concat([df] + arr, axis=1).reindex(index=secID, columns=reindex_col)