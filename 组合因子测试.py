# coding=utf-8

from lib.self_factors import *
import pandas as pd
from pandas import DataFrame
from math import *
import numpy as np
from numpy.linalg import LinAlgError
import statsmodels.api as sm
import random
from CAL.PyCAL import *
import scipy.optimize as sco
import datetime as dt


# def choose_date(date, dt):
#     enddate = Date.parseISO(date)
#     span = Period(dt)
#     startdate = enddate - span
#     datedf = DataAPI.MktEqumAdjGet(secID=u"",ticker=u"000001",monthEndDate=u"",isOpen=u"",beginDate=startdate,endDate=enddate,field=u"secID,endDate",pandas="1")
#     datelist = datedf['endDate'].tolist()
#     return datelist

def choose_date(date, dt, freq=10):
    startdate = cal.advanceDate(date, (str(-(dt)) + "B"))
    datelist = cal.bizDatesList(startdate, date)
    res = map(lambda x: x.strftime("%Y-%m-%d"), datelist[::freq])
    return res


def choose_stk(stklist, date):
    # 去除ST股
    STdf = DataAPI.SecSTGet(secID=stklist, beginDate=date, endDate=date, field=['secID'])
    STlist = STdf['secID'].tolist()
    stklist = [s for s in stklist if s not in STlist]

    # 去除交易日停牌的
    tvdf = DataAPI.MktEqudGet(tradeDate=date, secID=stklist, field=u"secID,turnoverValue", pandas="1")  # 去除当日停牌股票
    tvdf = tvdf.dropna(how='any')
    tvdf = tvdf[tvdf['turnoverValue'] != 0]
    stklist = tvdf['secID'].tolist()

    startDate = cal.advanceDate(date, '-10B').strftime("%Y%m%d")
    # 去除交易天数小于10的股票
    tvdf = DataAPI.MktEqudAdjGet(tradeDate=u"",
                                 secID=stklist,
                                 ticker=u"",
                                 isOpen="",
                                 beginDate=startDate,
                                 endDate=date,
                                 field=u"secID,tradeDate,turnoverValue",
                                 pandas="1")
    grouped = tvdf.groupby('secID')
    states = grouped.agg([np.size, np.mean])
    stklist = states['turnoverValue'].query("size >= 5").index.tolist()

    # 去除流动性差的股票,日均成交额大于1000万
    # tvdf['avertv'] = tvdf['turnoverValue']/tvdf['tradeDays']
    # tvdf = tvdf[tvdf['avertv'] > 1e7]
    return stklist


def regular_stock(stock, tradeDate, freq=10):
    # 去除ST股
    STdf = DataAPI.SecSTGet(secID=stock, beginDate=tradeDate, endDate=tradeDate, field=['secID'])
    STlist = STdf['secID'].tolist()
    stock = [s for s in stock if s not in STlist]

    # 去除交易日停牌的
    tvdf = DataAPI.MktEqudGet(tradeDate=tradeDate, secID=stock, field=u"secID,turnoverValue", pandas="1")  # 去除当日停牌股票
    tvdf = tvdf.dropna(how='any')
    tvdf = tvdf[tvdf['turnoverValue'] != 0]
    stock = tvdf['secID'].tolist()

    startDate = cal.advanceDate(tradeDate, '-%dB' % (freq - 1)).strftime("%Y%m%d")
    # 去除交易天数小于10的股票
    tvdf = DataAPI.MktEqudAdjGet(tradeDate=u"",
                                 secID=stock,
                                 ticker=u"",
                                 isOpen="",
                                 beginDate=startDate,
                                 endDate=tradeDate,
                                 field=u"secID,tradeDate,turnoverValue",
                                 pandas="1")
    grouped = tvdf.groupby('secID')
    states = grouped.agg([np.count_nonzero, np.mean])
    stock = states['turnoverValue'].query("count_nonzero >= %d" % (freq / 2)).index.tolist()

    return stock


def test_factor(datelist, universe1, factor, filename):  # 每个截面计算f以求得协方差矩阵F,集合全部截面数据求f
    lengthF = (len(datelist) - 1)
    lengthF = (len(datelist))

    X_arr = []
    Y_arr = []
    fparams = DataFrame()
    for i in range(lengthF):
        startdate = cal.advanceDate(datelist[i], '-10B').strftime("%Y-%m-%d")

        stklist = get_index_data(universe1, startdate).index
        stklist = regular_stock(stklist, startdate)
        X1, Y1, stk = getPreFactor(stklist, startdate, datelist[i], factor)
        print startdate, datelist[i]
        # stklist=set_universe("ZZ500", date = datelist[i])+set_universe('HS300',date= datelist[i])              
        # X1, Y1, stk= getPreFactor(stklist, datelist[i], datelist[i+1], factor)
        X_arr.append(X1)
        Y_arr.append(Y1)
        # print datelist[i],X.shape,Y.shape
    X = pd.concat(X_arr, keys=datelist)
    Y = pd.concat(Y_arr, keys=datelist)
    for j in range(len(factor)):
        res_ols = sm.OLS(Y, X[factor[j]]).fit()
        temp = DataFrame(np.vstack([res_ols.params, res_ols.tvalues, res_ols.rsquared]),
                         index=['freturn', 'tvalues', 'rsquared'], columns=[factor[j]])
        fparams = pd.concat([fparams, temp.T])
    print fparams.head()

    # X.index = datelist
    # Y.index =datelist
    X.to_csv(filename + "X.csv")
    Y.to_csv(filename + "Y.csv")
    fparams.to_csv(filename + ".csv")
    return fparams


def getPreFactor(stklist, date1, date2, factor):
    field = ['secID'] + factor
    # 去除当日停牌股票
    # temp = DataAPI.MktEqudGet(tradeDate=date2,secID=stklist,field=u"secID,turnoverValue",pandas="1").set_index(['secID'])#去除当日停牌股票
    # stklist=temp[temp['turnoverValue']!=0].index
    # temp1 = DataAPI.MktStockFactorsOneDayProGet(tradeDate=date1,secID=stklist,field=field,pandas="1").set_index(['secID']) #取上月底的因子值
    temp1 = get_factor_data(secID=stklist, tradeDate=date1, field=field)  # 自定义因子

    date1 = Date.strptime(date1, '%Y-%m-%d').strftime('%Y%m%d')
    date2 = Date.strptime(date2, '%Y-%m-%d').strftime('%Y%m%d')
    temp2 = get_SpecificRetRange(stklist, cal.advanceDate(date1, ("1B")).strftime("%Y%m%d"),
                                 date2)  # 计算收益，从date1起第2天至date2
    df = pd.concat([temp1, temp2], axis=1, join='inner')
    stk = df.index.tolist()
    for fac in factor:
        df[fac] = winsorize(df[fac].replace([np.inf, -np.inf], np.nan).fillna(0.0))
        df[fac] = neutralize(standardize(df[fac]).fillna(0.0), date1)  # 因子中性化
        df[fac] = df[fac].fillna(0.0)
    df['spret'] = df['spret'].fillna(0.0)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    # 去极值、中性化、标准化
    return X, y, stk


def get_SpecificRetRange(stklist, start, end):
    df = DataAPI.RMSpecificRetDayGet(secID=stklist, beginDate=start, endDate=end, field=['secID', 'tradeDate', 'spret'])
    df = df.pivot(index='tradeDate', columns='secID', values='spret') / 100 + 1  # 月收益转为日收益
    return 100 * 250 * (df.prod().rename('spret') - 1) / df.shape[0]  # 年化值


def date_process(dates=None, start=None, end=None, period=None, tradeDate=None, forward=0):
    if dates is not None:
        assert isinstance(dates, (list, tuple, np.ndarray))
        if forward:
            dates = [cal.advanceDate(d, '-%dB' % forward).strftime("%Y-%m-%d") for d in dates]
        return dates

    if tradeDate is not None:
        assert isinstance(tradeDate, str)
        return date_process([tradeDate], forward=forward)

    if None not in (start, end, period):
        if forward:
            start, end = [cal.advanceDate(d, '-%dB' % forward).strftime("%Y-%m-%d") for d in [start, end]]
        datelist = [d.strftime("%Y-%m-%d") for d in cal.bizDatesList(start, end)]
        return datelist[::period]

    raise ValueError('Parameter Error')


def get_index_data(index_code, tradeDate, debug=False):
    if isinstance(index_code, (list, tuple, np.ndarray)):
        arr = [get_index_data(code, tradeDate) for code in index_code]
        wb = pd.concat(arr)
        return wb / wb.sum()
    beginDate = cal.advanceDate(tradeDate, "-50B").strftime("%Y-%m-%d")
    df = DataAPI.IdxCloseWeightGet(secID="",
                                   ticker=index_code,
                                   beginDate=beginDate,
                                   endDate=tradeDate,
                                   field=u"secID,effDate,consID,weight",
                                   pandas="1").set_index('consID').rename_axis('secID')

    update = df['effDate'].max()
    df = df[df['effDate'] == update]
    stklist = df.index.tolist()
    wb = df['weight'] / 100
    offset = cal.bizDatesNumber(update, tradeDate, includeFirst=False)

    data = DataAPI.MktEqudAdjGet(secID=stklist,
                                 beginDate=update,
                                 endDate=tradeDate,
                                 field=u"secID,tradeDate,preClosePrice,closePrice,isOpen").set_index('secID')
    data.eval("chg = closePrice/preClosePrice")
    if offset == 0:
        wb /= data['chg']

    elif offset > 1:
        wb *= data.groupby(level=0)['chg'].apply(lambda x: x.iloc[:-1].prod())

    wb /= wb.sum()
    last_data = data.query("tradeDate == tradeDate.max()")
    stk_halt = last_data.query("isOpen==0").index
    wb.loc[stk_halt] = 0
    if debug:
        print 'index_ret', wb.dot(last_data['chg'] - 1)

    return wb


# parameters
cal = Calendar('China.SSE')
dt = 1200
start = '2017-08-01'  # 留最新一年的数据做样本外测试
end = '2017-08-01'
benchmark = 'ZZ500'
universe = DynamicUniverse('ZZ500') + DynamicUniverse('HS300')
capital_base = 100000000  # 起始资金
freq = 'd'
refresh_rate = 1

period = 1

import time


def initialize(account):  # 初始化虚拟账户状态
    pass


def handle_data(account):
    a = time.clock()
    today = account.current_date.strftime("%Y-%m-%d")
    print today

    datelist = date_process(start='2012-01-01', end='2017-01-01', period=1, forward=0)
    print datelist

    FactorZoo = ['Price1M', 'MoneyFlow20', 'PEHist60', 'alpha042',
                 'ILLIQUIDITY', 'MOM_Est_EPS', 'central_degree_ZZ500']
    filename = "barra test"
    universe1 = "000906"
    fparam = test_factor(datelist, universe1, FactorZoo, filename)  # 回归计算因子的R2以及每期的X与Y
    print time.clock() - a