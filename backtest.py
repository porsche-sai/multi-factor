# coding=utf-8

from __future__ import division
import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import quartz as qz
from quartz.api import *
from CAL.PyCAL import *

legendFont = font.copy()


# ************************************************
# 快速回测表现报告
# ************************************************
BENCHMARK_DICT = {'000905': u"中证500指数增强",
                  '000300': u"沪深300指数增强",
                  '多因子模型': u"多因子模型"}


def reportBt(bt, benchmark="多因子模型"):
    fig = plt.figure(figsize=(10, 8))
    fig.set_tight_layout(True)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.grid()
    ax2.grid()
    data = bt[[u'tradeDate', u'portfolio_value', u'benchmark_return']]
    data['portfolio_return'] = data.portfolio_value / data.portfolio_value.shift(1) - 1.0  # 总头寸每日回报率
    data['portfolio_return'].ix[0] = data['portfolio_value'].ix[0] / 100000000.0 - 1.0
    data['excess_return'] = data.portfolio_return - data.benchmark_return  # 总头寸每日超额回报率
    data['excess'] = data.excess_return + 1.0
    data['excess'] = data.excess.cumprod()  # 总头寸对冲指数后的净值序列
    data['portfolio'] = data.portfolio_return + 1.0
    data['portfolio'] = data.portfolio.cumprod()  # 总头寸不对冲时的净值序列
    data['benchmark'] = data.benchmark_return + 1.0
    data['benchmark'] = data.benchmark.cumprod()  # benchmark的净值序列
    data['hedged_max_drawdown'] = max(
        [1 - v / max(1, max(data['excess'][:i + 1])) for i, v in enumerate(data['excess'])])  # 对冲后净值最大回撤
    data['hedged_volatility'] = np.std(data['excess_return']) * np.sqrt(252)
    data['hedged_annualized_return'] = (data['excess'].values[-1]) ** (252.0 / len(data['excess'])) - 1.0
    ax1.plot(data['tradeDate'], data[['benchmark']], label="Index")
    ax1.plot(data['tradeDate'], data[['portfolio']], label="Portfolio", color='r')
    ax2.plot(data['tradeDate'], data[['excess']], label="Alpha", color='r')
    ax1.legend(loc=0, fontsize=12)
    ax2.legend(loc=0, fontsize=12)
    ax1.set_ylabel(u"净值", fontproperties=font, fontsize=16)
    ax2.set_ylabel(u"对冲净值", fontproperties=font, fontsize=16)
    ax1.set_title(u"%s - 净值走势" % BENCHMARK_DICT[benchmark], fontproperties=font, fontsize=16)
    ax2.set_title(u"%s - 对冲基准指数后净值走势" % BENCHMARK_DICT[benchmark], fontproperties=font, fontsize=16)


def turnoverBt(bt):
    fig = plt.figure(figsize=(10, 8))
    fig.set_tight_layout(True)
    ax1 = fig.add_subplot(211)
    ax1.grid()

    data = bt[['tradeDate', 'portfolio_value']].set_index('tradeDate')
    data.index = map(lambda x: x.strftime('%Y-%m-%d'), data.index)
    data['trade_vol'] = bt.blotter.apply(lambda x: sum([y.filled_amount * y.transact_price for y in x])).values
    data['buy_vol'] = bt.blotter.apply(
        lambda x: sum([y.filled_amount * y.transact_price if y.direction == 1 else 0 for y in x])).values
    data['sell_vol'] = bt.blotter.apply(
        lambda x: sum([y.filled_amount * y.transact_price if y.direction == -1 else 0 for y in x])).values
    tmpDf = (data[['trade_vol', 'buy_vol', 'sell_vol']].T / data['portfolio_value']).T
    data[['trade_vol', 'buy_vol', 'sell_vol']] = tmpDf[['trade_vol', 'buy_vol', 'sell_vol']].values
    data = data[(data.trade_vol > 0)]
    data[['trade_vol']].plot(kind='bar', ax=ax1)
    ax1.legend([u'交易换手率'], prop=legendFont, loc=0, fontsize=12)
    xmajorLocator = MultipleLocator(int(data.shape[0] / 20))
    ax1.xaxis.set_major_locator(xmajorLocator)
    ax1.set_ylabel(u"换手率", fontproperties=font, fontsize=16)
    ax1.set_title(u"组合换手率", fontproperties=font, fontsize=16)


def riskCoeffs(perf):
    '''根据perf给出回测的各个风险指标'''
    ex_value = perf['cumulative_returns'] - perf['benchmark_cumulative_returns'] + 1
    ex_returns = perf['returns'] - perf['benchmark_returns']
    ex_cumreturn = ex_returns + 1.0
    ex_cumreturn = ex_cumreturn.cumprod()  # 总头寸对冲指数后的净值序列

    perf['hedged_annualized_return'] = perf['annualized_return'] - perf['benchmark_annualized_return']
    perf['hedged_max_drawdown'] = max([1 - v / max(1, max(ex_cumreturn[:i + 1])) for i, v in enumerate(ex_cumreturn)])
    perf['hedged_volatility'] = np.std(ex_returns) * np.sqrt(252)
    perf['information_ratio2'] = perf['hedged_annualized_return'] / perf['hedged_volatility']
    resultsDf = pd.Series(perf)
    resultsDf = resultsDf[[u'alpha', u'beta', u'information_ratio', u'sharpe',
                           u'annualized_return', u'max_drawdown', u'volatility',
                           u'hedged_annualized_return', u'hedged_max_drawdown', u'hedged_volatility',
                           u'information_ratio2']]
    resultsDf = pd.DataFrame(resultsDf).T
    resultsDf.index = ['风险系数']
    for col in resultsDf.columns:
        resultsDf[col] = [np.round(x, 3) for x in resultsDf[col]]

    cols = [(u'风险指标', u'Alpha'), (u'风险指标', u'Beta'), (u'风险指标', u'信息比率'), (u'风险指标', u'夏普比率'),
            (u'纯股票多头时', u'年化收益'), (u'纯股票多头时', u'最大回撤'), (u'纯股票多头时', u'收益波动率'),
            (u'对冲后', u'年化收益'), (u'对冲后', u'最大回撤'),
            (u'对冲后', u'收益波动率'), (u'对冲后', u'信息比率')]
    resultsDf.columns = pd.MultiIndex.from_tuples(cols)
    return resultsDf
