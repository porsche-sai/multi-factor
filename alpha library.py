# -*- coding: UTF-8 -*
# Copyright 2017/10/23. All Rights Reserved
# Author: sai
# self_factors.py 2017/10/23 14:31
# coding=utf-8

import numpy as np
import pandas as pd
import statsmodels.api as sm
from CAL.PyCAL import *
from numpy import abs
from numpy import log
from numpy import sign
from numpy.linalg import LinAlgError
from quartz_extensions.MFHandler.SignalProcess import standardize, neutralize, winsorize
from scipy.stats import rankdata
from DataAPI.DataCube import get_data_cube

cal = Calendar('China.SSE')
UQER_ALPHA = DataAPI.MktStockFactorsOneDayProGet(secID=u"",
                                                 ticker=u"000001",
                                                 tradeDate=u"20170612",
                                                 field=u"",
                                                 pandas="1").columns.tolist()[3:]
BETA = DataAPI.RMExposureDayGet(secID=u"",
                                ticker=u"000001",
                                tradeDate="20170517",
                                field=u"",
                                pandas="1").columns[5:-1].tolist()


def LinAlgErrorDeco(func):
    def wrapper(*args, **kwargs):
        try:
            res = func(*args, **kwargs)
            return res
        except LinAlgError:
            return

    return wrapper


def min_volatility(secID, tradeDate):
    index_data = get_data_cube(secID + ['000906.ZICN'], field='close', start=tradeDate, end=tradeDate, freq='1m')
    tot_data = index_data.transpose(2, 0, 1).to_frame()['close'].unstack().T
    tot_returns = tot_data.pct_change()
    idx_returns = tot_returns['000906.ZICN']
    idv = tot_returns.drop('000906.ZICN', axis=1)
    dastd = standardize(winsorize(idv.iloc[-30:].std()))
    resvol = idv.iloc[-30:].apply(lambda x: sm.OLS(x, idx_returns.iloc[-30:]).fit().resid).std()
    return (dastd + standardize(winsorize(resvol))).rename('min_volatility').to_frame()


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

    data = DataAPI.MktEqudAdjAfGet(tradeDate=u"",
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
        try:
            res = sm.OLS(Y, X).fit()  # 回归分析
            rsquared = res.rsquared  # R方
            resid = res.resid  # 残差数组
            dispersion = np.sqrt(1 - rsquared) * resid.std()  # 计算离散度
        except LinAlgError:
            print 'Dispersion', tradeDate
            dispersion = np.nan
        dis_arr.append(dispersion)  # 添加至列表
    print 'dis', len(dis_arr)
    print 'stklist', len(stklist)
    return pd.Series(dis_arr, index=stklist, name='Dispersion')


def composite_P(secID, tradeDate):
    forward_2y = cal.advanceDate(tradeDate, '-2Y')
    datelist = cal.bizDatesList(forward_2y, tradeDate)
    date_10 = [d.strftime("%Y-%m-%d") for d in datelist[::-10][::-1]]
    startday = cal.advanceDate(date_10[0], '-61B')
    data = DataAPI.MktEqudAdjAfGet(secID=secID,
                                   beginDate=startday,
                                   endDate=tradeDate,
                                   field=u"secID,tradeDate,closePrice,turnoverVol,turnoverValue,turnoverRate",
                                   pandas="1")

    closePrice = data.pivot('tradeDate', 'secID', 'closePrice')
    volume = data.pivot('tradeDate', 'secID', 'turnoverVol')
    turnoverValue = data.pivot('tradeDate', 'secID', 'turnoverValue')
    turnoverRate = data.pivot('tradeDate', 'secID', 'turnoverRate')
    turnoverPrice = turnoverValue / volume

    returns = closePrice.pct_change()
    V_1 = 1 - turnoverRate

    updata_arr = []
    arc_arr = []
    vrc_arr = []
    src_arr = []
    krc_arr = []

    up_date = []
    low_date = []

    for day in date_10:
        rolling = V_1.loc[:day][-60:][::-1].rolling(window=60, min_periods=1).apply(lambda x: x.prod())
        ATR = rolling * turnoverRate.shift().loc[:day]
        RC = (1 - turnoverPrice.loc[:day] / turnoverPrice.loc[day])

        arc = ((ATR[-60:] * RC[-60:]).sum() / ATR[-60:].sum()).rename('ARC')
        arc_arr.append(arc)

        vrc_t1 = (ATR[-60:] * (RC[-60:] - arc).pow(2)).sum()
        vrc_t2 = ATR[-60:].sum()
        vrc = (60 * vrc_t1 / (59 * vrc_t2)).rename('VRC')
        vrc_arr.append(vrc)

        src_t1 = (ATR[-60:] * (RC[-60:] - arc).pow(3)).sum()
        src_t2 = ATR[-60:].sum()
        src = (60 * src_t1 / (59 * src_t2 * vrc.pow(1.5))).rename('SRC')
        src_arr.append(src)

        krc_t1 = (ATR[-60:] * (RC[-60:] - arc).pow(4)).sum()
        krc_t2 = ATR[-60:].sum()
        krc = (60 * krc_t1 / (59 * krc_t2 * vrc.pow(2))).rename('KRC')
        krc_arr.append(krc)

        updata = arc.median() > 0
        updata_arr.append(updata)

        if updata:
            up_date.append(day)
        else:
            low_date.append(day)
    ret_arr = []
    for i in xrange(len(date_10) - 1):
        d1 = date_10[i]
        d2 = date_10[i + 1]
        this_ret = (returns.loc[d1:d2][1:] + 1).prod() - 1
        ret_arr.append(this_ret)

    Y = pd.concat(ret_arr, keys=date_10[:-1])
    tmp = map(lambda x: pd.concat(x, keys=date_10), [arc_arr, vrc_arr, src_arr, krc_arr])
    X = pd.concat(tmp, axis=1)

    if X.loc[date_10[-1], 'ARC'].median() > 0:
        res = sm.OLS(Y.loc[up_date],
                     X.loc[np.intersect1d(up_date,
                                          date_10[:-1]).tolist()], missing='drop').fit().params

    else:
        res = sm.OLS(Y.loc[low_date],
                     X.loc[np.intersect1d(low_date,
                                          date_10[:-1]).tolist()], missing='drop').fit().params
    return X.loc[date_10[-1]].dot(res).rename('composite_P').to_frame()


def enterprise_perf(secID, tradeDate, n=20, forcast=10):
    tradeDate = pd.to_datetime(tradeDate).strftime("%Y-%m-%d")
    # 取业绩状况总表
    try:
        data = ENTERPRISE_PERFORMANCE
    except NameError:
        from lib.EnterprisePerf import ENTERPRISE_PERFORMANCE
        data = ENTERPRISE_PERFORMANCE
    # 截取最近n日数据
    step = np.true_divide(1, forcast)
    begdate = cal.advanceDate(tradeDate, '-%dB' % n).strftime("%Y-%m-%d")
    slice_df = data.truncate(begdate, tradeDate)
    # 取最近N日存在变动、并且在股票池内的股票
    df = slice_df.loc[:, (slice_df.columns.isin(secID) & slice_df.notnull().any()).values]
    tmp = df.ffill().sum()
    res = np.sign(tmp) * (1 - (tmp.abs() - (n - forcast) - 1).clip_lower(0) * step)
    return res.reindex(secID).rename('enterprise_perf')


def shares_excit(secID, tradeDate, n=20, forcast=10):
    step = np.true_divide(1, forcast)
    beginDate = cal.advanceDate(tradeDate, "-%dB" % n).strftime("%Y%m%d")
    datelist = [d.strftime("%Y-%m-%d") for d in cal.bizDatesList(beginDate, tradeDate)]
    events = DataAPI.EquSharesExcitGet(secID=secID,
                                       beginDate=beginDate,
                                       endDate=tradeDate,
                                       field=u"secID,publishDate,projRemark")
    events.columns = ['secID', 'publishDate', 'eventType']
    events = events.drop_duplicates(['secID', 'publishDate'])
    events['publishDate'] = events['publishDate'].apply(
        lambda x: cal.adjustDate(x, BizDayConvention.ModifiedPreceding).strftime("%Y-%m-%d"))
    events = events.pivot('publishDate', 'secID', 'eventType').reindex(index=datelist, columns=secID)
    df = events.loc[:, events.notnull().any().values]
    tmp = df.ffill().notnull().astype(int).sum()
    res = 1 - (tmp - (n - forcast) - 1).clip_lower(0) * step
    return res.reindex(secID).rename('shares_excit')


def EquMsChanges(secID, tradeDate, n=20, forcast=10):
    step = np.true_divide(1, forcast)
    beginDate = cal.advanceDate(tradeDate, "-%dB" % n).strftime("%Y%m%d")
    datelist = [d.strftime("%Y-%m-%d") for d in cal.bizDatesList(beginDate, tradeDate)]
    events = DataAPI.EquMsChangesGet(secID=secID,
                                     beginDate=beginDate,
                                     endDate=tradeDate,
                                     field=u"secID,changeDate,shareChanges")
    events.columns = ['secID', 'publishDate', 'eventType']
    events = events.drop_duplicates(['secID', 'publishDate'])
    events['publishDate'] = events['publishDate'].apply(
        lambda x: cal.adjustDate(x, BizDayConvention.ModifiedPreceding).strftime("%Y-%m-%d"))
    events = events.pivot('publishDate', 'secID', 'eventType').reindex(index=datelist, columns=secID)
    df = events.loc[:, events.notnull().any().values]
    tmp = (df.ffill() > 0).astype(int).sum()
    res = 1 - (tmp - (n - forcast) - 1).clip_lower(0) * step
    return res.reindex(secID).rename('EquMsChanges')


def participant(secID, tradeDate, forcast=10, n=10):
    tradeDate = pd.to_datetime(tradeDate).strftime("%Y-%m-%d")
    # 取业绩状况总表
    try:
        data = PARTICIPANT_EVENTS
    except NameError:
        from lib.ParticipantQA import PARTICIPANT_EVENTS
        data = PARTICIPANT_EVENTS
    # 截取最近n日数据
    step = np.true_divide(1, forcast)
    begdate = cal.advanceDate(tradeDate, '-%dB' % n).strftime("%Y-%m-%d")
    slice_df = data.truncate(begdate, tradeDate)
    # 取最近N日存在变动、并且在股票池内的股票
    df = slice_df.loc[:, (slice_df.columns.isin(secID) & slice_df.notnull().any()).values]
    tmp = df.ffill().sum()
    res = np.sign(tmp) * (1 - (tmp.abs() - (n - forcast) - 1).clip_lower(0) * step)
    return res.reindex(secID).rename('participant')


def Vol_of_Vol(secID, tradeDate):
    beginDate = cal.advanceDate(tradeDate, "-40B").strftime("%Y%m%d")
    data = get_data_cube(secID, ['highPrice', 'lowPrice'], start=beginDate, end=tradeDate, freq='5m')
    df = data.transpose(1, 2, 0).to_frame().T
    g = group_by_day(df)

    def vov(gp):
        min_data = gp.iloc[3:-3]
        sigma = np.sqrt((np.log(min_data['highPrice'] / min_data['lowPrice']) ** 2).mean() / (4 * np.log(2)))
        return standardize(winsorize(sigma))

    res = g.apply(vov)
    return res.std().rename('Vol_of_Vol')


def smart_Q(secID, tradeDate):
    tradeDate_5 = cal.advanceDate(tradeDate, "-5B").strftime("%Y%m%d")
    min_data = get_data_cube(secID,
                             field=['close', 'volume'],
                             start=tradeDate_5,
                             end=tradeDate,
                             freq='1m',
                             style='ast')
    min_data['returns'] = min_data['close'].pct_change()
    min_data['smartS'] = min_data['returns'].abs() / min_data['volume'].pow(0.5)
    grouped = min_data.groupby(lambda x: np.datetime64(x, 'D'))

    def get_Q(df):
        tv = df.sort_values('smartS', ascending=False)['volume']
        df['accumVolPct'] = tv.cumsum() / tv.sum()
        try:
            smart_VWAP = df.query('accumVolPct<0.2').eval('close * volume / volume.sum()').sum()
        except ZeroDivisionError:
            return
        all_VWAP = df.eval('close * volume / volume.sum()').sum()
        if all_VWAP != 0:
            return smart_VWAP / all_VWAP

    return grouped.apply(lambda x: x.iloc[:, -30:, :].apply(get_Q, axis=[0, 1]).to_frame()).mean(axis=1).rename(
        'smart_Q')


def NewsSentimentIndex(secID, tradeDate, N=5):
    startDate = cal.advanceDate(tradeDate, '-%dB' % N).strftime("%Y-%m-%d")
    news = DataAPI.NewsSentimentIndexV2Get(secID=secID,
                                           beginDate=startDate,
                                           endDate=tradeDate,
                                           field='secID,sentimentIndex,newsPublishDate')

    return news.pivot('newsPublishDate', 'secID', 'sentimentIndex').sum().rename('NewsSentimentIndex')


def NewsHeatIndex(secID, tradeDate, N=5):
    startDate = cal.advanceDate(tradeDate, '-%dB' % N).strftime("%Y-%m-%d")
    news = DataAPI.NewsHeatIndexV2Get(secID=secID,
                                      beginDate=startDate,
                                      endDate=tradeDate,
                                      field='secID,heatIndex,newsPublishDate')

    return news.pivot('newsPublishDate', 'secID', 'heatIndex').sum().rename('NewsHeatIndex')


def APM(secID, tradeDate):
    beginDate = cal.advanceDate(tradeDate, '-20B').strftime("%Y%m%d")
    df = get_data_cube(secID,
                       field=['close'],
                       start=beginDate,
                       end=tradeDate,
                       freq='60m').minor_xs('close')

    half_day_ret = group_by_day(df).apply(
        lambda day: (day.pct_change() + 1).groupby(lambda x: (pd.to_datetime(x).strftime('%H:%M') <= '11:30' and 'am')
                                                             or 'pm').apply(lambda y: y.prod()) - 1)

    half_day_ret = half_day_ret.replace([-np.inf, np.inf], np.nan)
    index_data = half_day_ret.mean(axis=1).fillna(0)
    individual = half_day_ret.dropna(axis=1)
    resid = individual.apply(lambda x: sm.OLS(x.dropna(), index_data.reindex_like(x.dropna())).fit().resid).swaplevel()

    sigma = resid.loc['am'] - resid.loc['pm']
    return (sigma.mean() / (sigma.std() / np.sqrt(sigma.shape[0]))).rename('APM')


def MixLIQ(secID, tradeDate):
    beginDate = cal.advanceDate(tradeDate, '-20B').strftime("%Y%m%d")
    df = get_data_cube(secID,
                       field=['turnoverValue'],
                       start=beginDate,
                       end=tradeDate,
                       freq='30m').minor_xs('turnoverValue')
    quarter_day_tv = group_by_day(df).apply(
        lambda day: day.groupby(lambda x: ('10:30' <= pd.to_datetime(x).strftime('%H:%M') <= '11:30' and 'two')
                                          or ('14:00' <= pd.to_datetime(x).strftime(
            '%H:%M') <= '15:00' and 'four') or None).sum())

    neg = DataAPI.MktEqudGet(secID=secID,
                             beginDate=beginDate,
                             endDate=tradeDate,
                             field=u"secID,tradeDate,negMarketValue", pandas="1").pivot('tradeDate', 'secID',
                                                                                        'negMarketValue')
    two = quarter_day_tv.swaplevel().loc['two']
    four = quarter_day_tv.swaplevel().loc['four']

    two_tr = np.log((two / neg).sum()).replace([-np.inf, np.inf], np.nan)
    four_tr = np.log((four / neg).sum()).replace([-np.inf, np.inf], np.nan)

    return (standardize(four_tr) - standardize(two_tr)).rename('MixLIQ')


def MixMom(secID, tradeDate):
    beginDate = cal.advanceDate(tradeDate, '-20B').strftime("%Y%m%d")
    df = get_data_cube(secID,
                       field=['close'],
                       start=beginDate,
                       end=tradeDate,
                       freq='60m').minor_xs('close').pct_change()
    M = df.groupby(lambda x: pd.to_datetime(x).strftime("%H:%M")).sum().replace([-np.inf, np.inf], np.nan)
    return M.T.dot([-0.47, -0.59, 0.76, 1.50, 1.00]).rename('MixMom')


# def vol_ratio(secID, tradeDate):
#     min_volume = get_data_cube(secID,
#                                field='volume',
#                                start=tradeDate,
#                                end=tradeDate,
#                                freq='1m',
#                                style='ast')['volume']
#     min_volume.index = min_volume.index.map(lambda x: x.split(' ')[-1])
#     ma_volume = min_volume.loc[(min_volume.index <= '10:00') | (min_volume.index >= '14:30')]
#     return (ma_volume.sum() / min_volume.sum()).rename('vol_ratio')
#
#
# def total_consistent_volume(secID, tradeDate, a=0.95):
#     min_data = get_data_cube(secID,
#                              field=['close', 'high', 'open', 'low', 'volume'],
#                              start=tradeDate,
#                              end=tradeDate,
#                              freq='5m',
#                              style='ast')
#     consistent = (min_data['close'] - min_data['open']).abs() <= a * (min_data['high'] - min_data['low']).abs()
#     consistent_volume = min_data['volume'][consistent].sum()
#     tot_volume = min_data['volume'].sum()
#     return (consistent_volume / tot_volume).rename('total_consistent_volume')
#
#
# def total_consistent_volume_rise(secID, tradeDate, a=0.95):
#     min_data = get_data_cube(secID,
#                              field=['close', 'high', 'open', 'low', 'volume'],
#                              start=tradeDate,
#                              end=tradeDate,
#                              freq='5m',
#                              style='ast')
#     consistent = (min_data['close'] - min_data['open']).abs() <= a * (min_data['high'] - min_data['low']).abs()
#     consistent_volume_rise = min_data['volume'][consistent & (min_data['close'] >= min_data['open'])].sum()
#     tot_volume = min_data['volume'].sum()
#     return (consistent_volume_rise / tot_volume).rename('total_consistent_volume_rise')
#
#
# def total_consistent_volume_fall(secID, tradeDate, a=0.95):
#     min_data = get_data_cube(secID,
#                              field=['close', 'high', 'open', 'low', 'volume'],
#                              start=tradeDate,
#                              end=tradeDate,
#                              freq='5m',
#                              style='ast')
#     consistent = (min_data['close'] - min_data['open']).abs() <= a * (min_data['high'] - min_data['low']).abs()
#     consistent_volume_fall = min_data['volume'][consistent & (min_data['close'] <= min_data['open'])].sum()
#     tot_volume = min_data['volume'].sum()
#     return (consistent_volume_fall / tot_volume).rename('total_consistent_volume_fall')


# region Auxiliary functions
def cmp_min(df, threshold):
    if isinstance(threshold, pd.DataFrame):
        threshold = threshold.fillna(np.inf)
    return df.clip_upper(threshold)


def cmp_max(df, threshold):
    if isinstance(threshold, pd.DataFrame):
        threshold = threshold.fillna(-np.inf)
    return df.clip_lower(threshold)


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


def gt_sma(df, n, m):
    alpha = np.true_divide(m, n)
    return df.ewm(alpha=alpha).mean()


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


def count(condition, window=5):
    return ts_sum(condition.astype(int), window)


def sumif(df, condition, window):
    return ts_sum(df[condition].fillna(0), window)


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


def group_by_day(df):
    return df.groupby(by=lambda x: np.datetime64(x, 'D'))


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
        self.turnoverRate = data.minor_xs('turnoverRate')
        self.VWAP = self.turnoverValue / self.volume
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

    def alpha042(self):
        df = correlation(self.high, rank(self.volume), 10)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * df

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
        #         W_data = DataAPI.MktEqudAdjAfGet(secID=self.secID,
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

        sliced_VRBP = VRBP.iloc[-20:].reindex(columns=self.secID).dropna(how='all', axis=1)
        sliced_SPPI = SPPI.iloc[-20:].reindex(columns=self.secID).dropna(how='all', axis=1)
        panel = pd.concat([sliced_VRBP, sliced_SPPI], keys=['VRBP', 'SPPI']).to_panel()
        return panel.apply(lambda x: sm.OLS(x.iloc[0], x.iloc[1]).fit().params[0], axis=(1, 2))

    def alpha103(self):
        #         W_data = DataAPI.MktEqudAdjAfGet(secID=self.secID,
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

        sliced_VRBP = VRBP.iloc[-20:].reindex(columns=self.secID).dropna(how='all', axis=1)
        sliced_SPPI = SPPI.iloc[-20:].reindex(columns=self.secID).dropna(how='all', axis=1)
        sqrt_SPPI = np.sqrt(sliced_SPPI.abs()) * np.sign(sliced_SPPI)
        panel = pd.concat([sliced_VRBP, sqrt_SPPI], keys=['VRBP', 'SPPI']).to_panel()
        return panel.apply(lambda x: sm.OLS(x.iloc[0], x.iloc[1]).fit().params[0], axis=(1, 2))

    def pingpong(self):
        volume_5 = self.volume.iloc[-5:]
        volume_60 = self.volume.iloc[-60:]
        volume_5 /= volume_5.sum()
        volume_60 /= volume_60.sum()

        res = (self.close.iloc[-5:] * volume_5).sum() / (self.close.iloc[-60:] * volume_60).sum()
        return res.fillna(0).rename('pingpong').to_frame().T

    def CGO(self):
        V = self.turnoverRate
        P = self.turnoverValue / self.volume
        V_1 = 1 - V
        rolling = V_1.rolling(window=99, min_periods=1).apply(lambda x: x.prod())
        t1 = rolling * V
        RP = (t1.iloc[-100:] * P.iloc[-100:] / t1.iloc[-100:].sum()).sum()

        return (P.iloc[-2] / RP - 1).rename('CGO').to_frame().T

    # 上个月最大日收益
    def MaxRet(self):
        return self.returns.iloc[-22:].max().rename('MaxRet').to_frame().T

    # 过去三个月日收益率数据计算的标准差
    def RealizedVolatility(self):
        return self.returns.iloc[-66:].std().rename('RealizedVolatility').to_frame().T

    # 股价相比最近一个月均价涨幅
    def Momentumave1M(self):
        return (self.close.iloc[-1] / self.close.iloc[-22:].mean() - 1).rename('Momentumave1M').to_frame().T

    # Momentumlast6M
    def Momentumlast6M(self):
        return (self.close.iloc[-1] / self.close.iloc[-126] - 1).rename('Momentumlast6M').to_frame().T

    # 过去一个月日均成交额/过去三个月日均成交额
    def AmountAvg_1M_3M(self):
        return (self.volume[-22:].mean() / self.volume[-66:].mean()).rename('AmountAvg_1M_3M').to_frame().T

    def central_degree_ZZ800(self):
        beginDate = self.dates[-20].replace('-', '')
        endDate = self.dates[-1].replace('-', '')
        index_data = DataAPI.MktIdxdGet(tradeDate=u"",
                                        indexID=u"",
                                        ticker=u"000906",
                                        beginDate=beginDate, endDate=endDate,
                                        exchangeCD=u"XSHE,XSHG",
                                        field=u"tradeDate,CHGPct", pandas="1").set_index('tradeDate')
        rs = self.returns[-20:].dropna(how='all', axis=1).apply(lambda x: sm.OLS(index_data, x).fit().rsquared)
        # rs = self.returns[-20:].apply(debug)
        return np.log(rs / (1 - rs)).replace([-np.inf, np.inf], np.nan).to_frame().T

    def central_degree_ZZ500(self):
        beginDate = self.dates[-20].replace('-', '')
        endDate = self.dates[-1].replace('-', '')
        index_data = DataAPI.MktIdxdGet(tradeDate=u"",
                                        indexID=u"",
                                        ticker=u"000905",
                                        beginDate=beginDate, endDate=endDate,
                                        exchangeCD=u"XSHE,XSHG",
                                        field=u"tradeDate,CHGPct", pandas="1").set_index('tradeDate')
        rs = self.returns[-20:].dropna(how='all', axis=1).apply(lambda x: sm.OLS(index_data, x).fit().rsquared)
        # rs = self.returns[-20:].apply(debug)
        return np.log(rs / (1 - rs)).replace([-np.inf, np.inf], np.nan).to_frame().T

    def central_degree_HS300(self):
        beginDate = self.dates[-20].replace('-', '')
        endDate = self.dates[-1].replace('-', '')
        index_data = DataAPI.MktIdxdGet(tradeDate=u"",
                                        indexID=u"",
                                        ticker=u"000300",
                                        beginDate=beginDate, endDate=endDate,
                                        exchangeCD=u"XSHE,XSHG",
                                        field=u"tradeDate,CHGPct", pandas="1").set_index('tradeDate')
        rs = self.returns[-20:].dropna(how='all', axis=1).apply(lambda x: sm.OLS(index_data, x).fit().rsquared)
        return np.log(rs / (1 - rs)).replace([-np.inf, np.inf], np.nan).to_frame().T

    def central_degree_ZZ800(self):
        beginDate = self.dates[-20].replace('-', '')
        endDate = self.dates[-1].replace('-', '')
        index_data = DataAPI.MktIdxdGet(tradeDate=u"",
                                        indexID=u"",
                                        ticker=u"000906",
                                        beginDate=beginDate, endDate=endDate,
                                        exchangeCD=u"XSHE,XSHG",
                                        field=u"tradeDate,CHGPct", pandas="1").set_index('tradeDate')
        rs = self.returns[-20:].dropna(how='all', axis=1).apply(lambda x: sm.OLS(index_data, x).fit().rsquared)
        return np.log(rs / (1 - rs)).replace([-np.inf, np.inf], np.nan).to_frame().T

    def central_degree_ZZ500_corr(self):
        beginDate = self.dates[-20].replace('-', '')
        endDate = self.dates[-1].replace('-', '')
        index_data = DataAPI.MktIdxdGet(tradeDate=u"",
                                        indexID=u"",
                                        ticker=u"000905",
                                        beginDate=beginDate, endDate=endDate,
                                        exchangeCD=u"XSHE,XSHG",
                                        field=u"tradeDate,CHGPct", pandas="1").set_index('tradeDate').iloc[:, 0]

        return self.returns[-20:].dropna(how='all', axis=1).apply(
            lambda x: x.corr(index_data, method='spearman')).to_frame().T

    def central_degree_HS300_corr(self):
        beginDate = self.dates[-20].replace('-', '')
        endDate = self.dates[-1].replace('-', '')
        index_data = DataAPI.MktIdxdGet(tradeDate=u"",
                                        indexID=u"",
                                        ticker=u"000300",
                                        beginDate=beginDate, endDate=endDate,
                                        exchangeCD=u"XSHE,XSHG",
                                        field=u"tradeDate,CHGPct", pandas="1").set_index('tradeDate').iloc[:, 0]
        return self.returns[-20:].dropna(how='all', axis=1).apply(
            lambda x: x.corr(index_data, method='spearman')).to_frame().T

    def central_degree_ZZ800_corr(self):
        beginDate = self.dates[-20].replace('-', '')
        endDate = self.dates[-1].replace('-', '')
        index_data = DataAPI.MktIdxdGet(tradeDate=u"",
                                        indexID=u"",
                                        ticker=u"000906",
                                        beginDate=beginDate, endDate=endDate,
                                        exchangeCD=u"XSHE,XSHG",
                                        field=u"tradeDate,CHGPct", pandas="1").set_index('tradeDate').iloc[:, 0]
        return self.returns[-20:].dropna(how='all', axis=1).apply(
            lambda x: x.corr(index_data, method='spearman')).to_frame().T

    def ARC(self):
        V = self.turnoverRate
        P = self.turnoverValue / self.volume
        V_1 = 1 - V
        rolling = V_1[::-1].rolling(window=60, min_periods=1).apply(lambda x: x.prod())
        ATR = rolling * V.shift()
        RC = (1 - P / P.iloc[-1])
        res = (ATR[-60:] * RC[-60:]).sum() / ATR[-60:].sum()
        return res.to_frame().T

    def VRC(self):
        order = pd.Series((np.arange(len(self.dates) - 1, -1, -1)), index=self.dates, name='tradeDate')
        V = self.turnoverRate  # 换手率
        P = self.turnoverValue / self.volume  # 成交均价
        V_1 = 1 - V
        rolling = V_1[::-1].rolling(window=60, min_periods=1).apply(lambda x: x.prod())
        ATR = rolling * V.shift()
        RC = (1 - P / P.iloc[-1])

        t1 = (ATR[-60:] * (RC[-60:] - self.ARC().iloc[0]).pow(2)).sum()
        t2 = ATR[-60:].sum()
        res = 60 * t1 / (59 * t2)
        return res.to_frame().T

    def SRC(self):
        order = pd.Series((np.arange(len(self.dates) - 1, -1, -1)), index=self.dates, name='tradeDate')
        V = self.turnoverRate  # 换手率
        P = self.turnoverValue / self.volume  # 成交均价
        V_1 = 1 - V
        rolling = V_1[::-1].rolling(window=60, min_periods=1).apply(lambda x: x.prod())
        ATR = rolling * V.shift()
        RC = (1 - P / P.iloc[-1])

        t1 = (ATR[-60:] * (RC[-60:] - self.ARC().iloc[0]).pow(3)).sum()
        t2 = ATR[-60:].sum()
        res = 60 * t1 / (59 * t2 * self.VRC().iloc[0].pow(1.5))
        return res.to_frame().T

    def KRC(self):
        order = pd.Series((np.arange(len(self.dates) - 1, -1, -1)), index=self.dates, name='tradeDate')
        V = self.turnoverRate  # 换手率
        P = self.turnoverValue / self.volume  # 成交均价
        V_1 = 1 - V
        rolling = V_1[::-1].rolling(window=60, min_periods=1).apply(lambda x: x.prod())
        ATR = rolling * V.shift()
        RC = (1 - P / P.iloc[-1])

        t1 = (ATR[-60:] * (RC[-60:] - self.ARC().iloc[0]).pow(4)).sum()
        t2 = ATR[-60:].sum()
        res = 60 * t1 / (59 * t2 * self.VRC().iloc[0].pow(2))
        return res.to_frame().T

    def VOL_diff(self, n=20):
        log_ret = np.log(self.close).diff()
        positive = (log_ret > 0).astype(int)
        negative = (log_ret < 0).astype(int)

        r_positive = ts_sum(log_ret * positive, window=n) / ts_sum(positive, window=n)
        r_negtive = ts_sum(log_ret * negative, window=n) / ts_sum(negative, window=n)

        VOL_positive = ts_sum((log_ret - r_positive).pow(2) * positive, window=n)
        VOL_negative = ts_sum((log_ret - r_negtive).pow(2) * negative, window=n)

        return VOL_positive - VOL_negative

    def min_VOL_Diff(self, n=20, freq=30):
        min_data = self._get_min_close(freq=freq)
        log_ret = np.log(min_data).diff()
        positive = (log_ret > 0).astype(int)
        negative = (log_ret < 0).astype(int)

        r_positive = ts_sum(log_ret * positive, window=n) / ts_sum(positive, window=n)
        r_negtive = ts_sum(log_ret * negative, window=n) / ts_sum(negative, window=n)
        r_mean = r_positive + r_negtive

        VOL_positive = ts_sum((log_ret - r_positive).pow(2) * positive, window=n)
        VOL_negative = ts_sum((log_ret - r_negtive).pow(2) * negative, window=n)
        VOL = VOL_positive + VOL_negative

        return np.sqrt(n) * ts_sum((log_ret - r_mean).pow(3), window=n) / VOL.pow(1.5)

    def VOL_SKEW(self, n=20):
        log_ret = np.log(self.close).diff()
        positive = (log_ret > 0).astype(int)
        negative = (log_ret < 0).astype(int)

        r_positive = ts_sum(log_ret * positive, window=n) / ts_sum(positive, window=n)
        r_negtive = ts_sum(log_ret * negative, window=n) / ts_sum(negative, window=n)
        r_mean = r_positive + r_negtive

        VOL_positive = ts_sum((log_ret - r_positive).pow(2) * positive, window=n)
        VOL_negative = ts_sum((log_ret - r_negtive).pow(2) * negative, window=n)
        VOL = VOL_positive + VOL_negative

        return np.sqrt(n) * ts_sum((log_ret - r_mean).pow(3), window=n) / VOL.pow(1.5)

    def VOL_KURSIS(self, n=20):
        log_ret = np.log(self.close).diff()
        positive = (log_ret > 0).astype(int)
        negative = (log_ret < 0).astype(int)

        r_positive = ts_sum(log_ret * positive, window=n) / ts_sum(positive, window=n)
        r_negtive = ts_sum(log_ret * negative, window=n) / ts_sum(negative, window=n)
        r_mean = r_positive + r_negtive

        VOL_positive = ts_sum((log_ret - r_positive).pow(2) * positive, window=n)
        VOL_negative = ts_sum((log_ret - r_negtive).pow(2) * negative, window=n)
        VOL = VOL_positive + VOL_negative
        return n * ts_sum((log_ret - r_mean).pow(4), window=n) / VOL.pow(2)

    def VOL_diff_10(self):
        return self.VOL_diff(10)

    def VOL_SKEW_10(self):
        return self.VOL_SKEW(10)

    def VOL_KURSIS_10(self):
        return self.VOL_KURSIS(10)

    def VOL_diff_50(self):
        return self.VOL_diff(50)

    def VOL_SKEW_50(self):
        return self.VOL_SKEW(50)

    def VOL_KURSIS_50(self):
        return self.VOL_KURSIS(50)

    def _get_min_close(self, freq):
        if not hasattr(self, '_min_close'):
            freq = '%dm' % freq
            beg_min = cal.advanceDate(self.tradeDate, '-5B').strftime("%Y-%m-%d")
            _min_close = get_data_cube(self.secID, field=['close'], start=beg_min, end=self.tradeDate, freq=freq,
                                       style='ast')
            setattr(self, '_min_close', _min_close['close'])
        return self._min_close

    def _get_RDVar(self, freq=5):
        if not hasattr(self, '_RDVar'):
            data = self._get_min_close(freq=freq)
            _RDVar = (data.groupby(by=lambda x: np.datetime64(x, 'D'))
                .apply(lambda x: np.log(x).diff().pow(2).sum()))
            setattr(self, '_RDVar', _RDVar)
        return self._RDVar

    def RVol(self, freq=5):
        RDVar = self._get_RDVar(freq=freq)
        return (np.sqrt(252.0 / 5 * RDVar.sum())).to_frame().T

    def RSkew(self, freq=5):
        min_data = self._get_min_close(freq=freq)
        RDVar = self._get_RDVar(freq=freq)
        N = 4 * 60.0 / freq
        RDSkew = np.sqrt(N) * (min_data.groupby(by=lambda x: np.datetime64(x, 'D'))
            .apply(lambda x: np.log(x).diff().pow(3).sum())) / RDVar.pow(1.5)

        return (RDSkew.sum() / 5).to_frame().T

    def RKurt(self, freq=5):
        min_data = self._get_min_close(freq=freq)
        RDVar = self._get_RDVar(freq=freq)
        N = 4 * 60.0 / freq
        RDKurt = N * (min_data.groupby(by=lambda x: np.datetime64(x, 'D'))
            .apply(lambda x: np.log(x).diff().pow(4).sum())) / RDVar.pow(2)
        return (RDKurt.sum() / 5).to_frame().T

    def _get_DriftRDVar(self, freq):
        if not hasattr(self, '_DriftRDVar'):
            data = self._get_min_close(freq=freq)
            miu = np.log(data).diff().mean()
            _DriftRDVar = (data.groupby(by=lambda x: np.datetime64(x, 'D'))
                .apply(lambda x: (np.log(x).diff() - miu).pow(2).sum()))
            setattr(self, '_DriftRDVar', _DriftRDVar)
        return self._DriftRDVar

    def DriftRVol(self, freq=5):
        DriftRDVar = self._get_DriftRDVar(freq=freq)
        return (np.sqrt(252.0 / 5 * DriftRDVar.sum())).to_frame().T

    def DriftRSkew(self, freq=5):
        min_data = self._get_min_close(freq=freq)
        miu = np.log(min_data).diff().mean()
        DriftRDVar = self._get_DriftRDVar(freq=freq)
        N = 4 * 60.0 / freq
        DriftRDSkew = np.sqrt(N) * (min_data.groupby(by=lambda x: np.datetime64(x, 'D'))
            .apply(lambda x: (np.log(x).diff() - miu).pow(3).sum())) / DriftRDVar.pow(1.5)
        return (DriftRDSkew.sum() / 5).to_frame().T

    def DriftKurt(self, freq=5):
        min_data = self._get_min_close(freq=freq)
        miu = np.log(min_data).diff().mean()
        DriftRDVar = self._get_DriftRDVar(freq=freq)
        N = 4 * 60.0 / freq
        DriftRDKurt = N * (min_data.groupby(by=lambda x: np.datetime64(x, 'D'))
            .apply(lambda x: (np.log(x).diff() - miu).pow(4).sum())) / DriftRDVar.pow(2)
        return (DriftRDKurt.sum() / 5).to_frame().T

    def GT_alpha1(self):
        a = rank(np.log(self.volume.iloc[-7:]).diff())
        b = rank(self.close / self.open - 1).iloc[-7:]
        return -1 * correlation(a, b, 6).dropna(axis=1, how='all')

    def GT_alpha2(self):
        res = ((self.close - self.low) - (self.high - self.close)) / (self.high - self.low).iloc[-2:]
        return -1 * res.dropna(axis=1, how='all')

    def GT_alpha3(self):
        pos2 = self.close > delay(self.close, 1)
        b = self.low.clip_upper(delay(self.close, 1).fillna(np.inf))
        b[~pos2] = self.high.clip_lower(delay(self.close, 1).fillna(-np.inf))[~pos2]

        pos1 = self.close == delay(self.close, 1)
        a = self.close - b
        a[pos1] = 0

        return ts_sum(a, 6)

    def GT_alpha4(self):
        pos3 = self.volume / sma(self.volume, 20) >= 1
        res3 = self.volume.copy()
        res3[pos3] = 1
        res3[~pos3] = -1

        pos2 = ts_sum(self.close, 2) / 2 < (ts_sum(self.volume, 8) / 8 - stddev(self.close, 8))
        res2 = res3
        res2[pos2] = 1

        pos1 = (ts_sum(self.close, 8) / 8 + stddev(self.close, 8)) < ts_sum(self.close, 2) / 2
        res = res2
        res[pos1] = -1

        return res

    def GT_alpha5(self):
        a = ts_rank(self.volume, 5)
        b = ts_rank(self.high, 5)
        return -1 * ts_max(correlation(a, b, 5), 3)

    def GT_alpha6(self):
        return -1 * rank(delta(0.85 * self.open + 0.15 * self.high, 4))

    def GT_alpha7(self):
        res = rank(self.VWAP.clip_lower(3)) + rank(self.VWAP.clip_upper(3)) * rank(delta(self.volume, 3))
        return -1 * res

    def GT_alpha8(self):
        res = rank(delta((self.high + self.low) / 2 * 0.2 + self.VWAP * 0.8, 4) * -1)
        return res

    def GT_alpha9(self):
        res = sma((self.high + self.low) / 2 - (delay(self.high, 1) + delay(self.low, 1)) / 2 * (
                self.high - self.low) / self.volume, 7)
        return res

    def GT_alpha10(self):
        pos = self.returns < 0
        res = stddev(self.returns, 20)
        res[~pos] = self.close[~pos]
        res = res.pow(2).clip_lower(5)
        return rank(res)

    def GT_alpha11(self):
        res = ts_sum(((self.close - self.low) - (self.high - self.close)) / (self.high - self.low) * self.volume, 6)
        return res

    def GT_alpha12(self):
        res = rank(self.open - ts_sum(self.VWAP, 10) / 10) * (rank(abs(self.close - self.VWAP)))
        return -1 * res

    def GT_alpha13(self):
        res = (self.high * self.low).pow(0.5) - self.VWAP
        return res

    def GT_alpha14(self):
        res = self.close - delay(self.close, 5)
        return res

    def GT_alpha15(self):
        res = self.open / delay(self.close, 1) - 1
        return res

    def GT_alpha16(self):
        res = ts_max(rank(correlation(rank(self.volume), rank(self.VWAP), 5)), 5)
        return -1 * res

    def GT_alpha17(self):
        res = rank((self.VWAP - self.VWAP.clip_lower(15))).pow(delta(self.close, 5))
        return res

    def GT_alpha18(self):
        res = self.close / delay(self.close, 5)
        return res

    def GT_alpha19(self):
        pos2 = self.close == delay(self.close, 5)
        res2 = (self.close - delay(self.close, 5)) / self.close
        res2[pos2] = 0
        pos1 = self.close < delay(self.close, 5)
        res = res2
        res[pos1] = ((self.close - delay(self.close, 5)) / delay(self.close, 5))[pos1]
        return res

    def GT_alpha20(self):
        res = (self.close - delay(self.close, 6)) / delay(self.close, 6) * 100
        return res

    @LinAlgErrorDeco
    def GT_alpha21(self):
        X = sma(self.close, 6).iloc[-6:].dropna(how='all', axis=1).fillna(0)
        Y = np.arange(1, 7)

        res = sm.OLS(Y, X).fit().params
        return res.to_frame().T

    def GT_alpha22(self):
        res = sma(self.close - sma(self.close, 6) / sma(self.close, 6) \
                  - delay(self.close - sma(self.close, 6) / sma(self.close, 6), 3), 12)
        return res

    def GT_alpha23(self):
        pos = self.close > delay(self.close, 1)
        tmp = stddev(self.close, 20)
        res1 = tmp.copy()
        res2 = tmp.copy()
        res1[~pos] = 0
        res2[pos] = 0
        return gt_sma(res1, 20, 1) / (gt_sma(res1, 20, 1) + gt_sma(res2, 20, 1)) * 100

    def GT_alpha24(self):
        res = gt_sma(self.close - delay(self.close, 5), 5, 1)
        return res

    def GT_alpha25(self):
        res = -1 * rank(delta(self.close, 7) \
                        * (1 - rank(decay_linear(self.volume / sma(self.volume, 20), 9)))) \
              * (1 + rank(ts_sum(self.returns, 250)))
        return res

    def GT_alpha26(self):
        res = (ts_sum(self.close, 7) / 7 - self.close) - correlation(self.VWAP, delay(self.close, 5), 230)
        return res

    def GT_alpha28(self):
        a = (self.close - ts_min(self.low, 9)) / (ts_max(self.high, 9) - ts_min(self.low, 9)) * 100
        b = (self.close - ts_min(self.low, 9)) / (cmp_max(self.high, 9) - ts_max(self.low, 9)) * 100
        return 3 * gt_sma(a, 3, 1) - 2 * gt_sma(gt_sma(b, 3, 1), 3, 1)

    def GT_alpha29(self):
        res = (self.close - delay(self.close, 6)) / delay(self.close, 6) * self.volume
        return res

    def GT_alpha31(self):
        res = (self.close - sma(self.close, 12)) / sma(self.close, 12) * 100
        return res

    def GT_alpha32(self):
        res = -1 * ts_sum(rank(correlation(rank(self.high), rank(self.volume), 3)), 3)
        return res

    def GT_alpha33(self):
        a = -1 * ts_min(self.low, 5) + delay(ts_min(self.low, 5), 5)
        b = rank((ts_sum(self.returns, 240) - ts_sum(self.returns, 20)) / 220)
        c = ts_rank(self.volume, 5)
        res = a * b * c
        return res

    def GT_alpha34(self):
        res = sma(self.close, 12) / self.close
        return res

    def GT_alpha35(self):
        a = rank(decay_linear(delta(self.open, 1), 15))
        b = rank(decay_linear(correlation(self.volume, self.open * 0.65 + self.open * 0.35, 17), 7))
        res = a.clip_upper(b.fillna(np.inf))
        return -1 * res

    def GT_alpha36(self):
        res = rank(ts_sum(correlation(rank(self.volume), rank(self.VWAP), 6), 2))
        return res

    def GT_alpha37(self):
        res = rank(
            ts_sum(self.open, 5) * ts_sum(self.returns, 5) - delay(ts_sum(self.open, 5) * ts_sum(self.returns, 5), 10))
        return -1 * res

    def GT_alpha38(self):
        pos = ts_sum(self.high, 20) / 20 < self.high
        res = -1 * delta(self.high, 2)
        res[~pos] = 0
        return res

    def GT_alpha39(self):
        a = rank(decay_linear(delta(self.close, 2), 8))
        b = rank(
            decay_linear(correlation(self.VWAP * 0.3 + self.open * 0.7, ts_sum(sma(self.volume, 180), 37), 14), 12))
        res = b - a
        return res

    def GT_alpha40(self):
        pos1 = self.close > delay(self.close, 1)
        res1 = self.volume.copy()
        res1[pos1] = 0

        res2 = self.volume.copy()
        res2[~pos1] = 0
        res = ts_sum(res1, 26) / ts_sum(res2, 26) * 100
        return res

    def GT_alpha41(self):
        res = rank(-delta(self.VWAP, 3).clip_lower(5))
        return res

    def GT_alpha42(self):
        a = -1 * rank(stddev(self.high, 10))
        b = correlation(self.high, self.volume, 10)
        res = a * b
        return res

    def GT_alpha43(self):
        pos2 = self.close < delay(self.close, 1)
        res2 = -self.volume.copy()
        res2[~pos2] = 0

        pos1 = self.close > delay(self.close, 1)
        res = self.volume.copy()
        res[~pos1] = res2[~pos1]
        return ts_sum(res, 6)

    def GT_alpha44(self):
        a = ts_rank(decay_linear(correlation(self.low, sma(self.volume, 10), 7), 6), 4)
        b = ts_rank(decay_linear(delta(self.VWAP, 3), 10), 15)
        res = a + b
        return res

    def GT_alpha45(self):
        a = rank(delta(self.close * 0.6 + self.open * 0.4, 1))
        b = rank(correlation(self.VWAP, sma(self.volume, 150), 15))
        res = a * b
        return res

    def GT_alpha46(self):
        a = sma(self.close, 3)
        b = sma(self.close, 6)
        c = sma(self.close, 12)
        d = sma(self.close, 24)
        res = (a + b + c + d) / (4 * self.close)
        return res

    def GT_alpha47(self):
        res = gt_sma((ts_max(self.high, 6) - self.close) / (ts_max(self.high, 6) - ts_min(self.low, 6)) * 100, 9, 1)
        return res

    def GT_alpha48(self):
        a = sign(self.close - delay(self.close, 1))
        b = sign(delay(self.close, 1) - delay(self.close, 2))
        c = sign(delay(self.close, 2) - delay(self.close, 3))
        res = -1 * rank(a + b + c) * ts_sum(self.volume, 5) / ts_sum(self.volume, 20)
        return res

    def GT_alpha49(self):
        pos = (self.high + self.low) >= (delay(self.high, 1) + delay(self.low, 1))
        tmp = cmp_max(abs(self.high - delay(self.high, 1)), abs(self.low - delay(self.low, 1)))
        res1 = tmp.copy()
        res2 = tmp.copy()
        res1[pos] = 0
        res2[~pos] = 0
        return ts_sum(res1, 12) / (ts_sum(res1, 12) + ts_sum(res2, 12))

    def GT_alpha50(self):
        pos = (self.high + self.low) <= (delay(self.high, 1) + delay(self.low, 1))
        tmp = cmp_max(abs(self.high - delay(self.high, 1)), abs(self.low - delay(self.low, 1)))
        res1 = tmp.copy()
        res2 = tmp.copy()
        res1[pos] = 0
        res2[~pos] = 0
        res1 = ts_sum(res1, 12)
        res2 = ts_sum(res2, 12)
        return res1 / (res1 + res2) - res2 / (res2 + res1)

    def GT_alpha52(self):
        a = (self.high - delay((self.high + self.low + self.close) / 3, 1)).clip_upper(0)
        b = (delay((self.high + self.low + self.close) / 3, 1) - self.low).clip_upper(0)
        res = ts_sum(a, 26) / ts_sum(b, 26) * 100
        return res

    def GT_alpha53(self):
        res = count(self.close > delay(self.close, 1), 12) / 12 * 100
        return res

    def GT_alpha54(self):
        res = -1 * rank(stddev(abs(self.close - self.open)) + self.close - self.open) + correlation(self.close,
                                                                                                    self.open, 10)
        return res

    # def GT_alpha55(self):

    def GT_alpha56(self):
        a = rank(self.open - ts_min(self.open, 12))
        b = rank(rank(correlation(ts_sum((self.high + self.low) / 2, 19), ts_sum(sma(self.volume, 40), 19), 13)).pow(5))
        return (a < b).astype(int)

    def GT_alpha57(self):
        res = gt_sma((self.close - ts_min(self.low, 9)) / (ts_max(self.high, 9) - ts_min(self.low, 9)) * 100, 3, 1)
        return res

    def GT_alpha58(self):
        res = count(self.close > delay(self.close, 1), 20) / 20 * 100
        return res

    def GT_alpha59(self):
        pos2 = self.close > delay(self.close, 1)
        res2 = self.high.clip_lower(delay(self.close, 1).fillna(-np.inf))
        res2[pos2] = self.low.clip_upper(delay(self.close, 1).fillna(np.inf))

        res = self.close - res2
        pos = self.close == delay(self.close, 1)
        res[pos] = 0

        return ts_sum(res, 20)

    def GT_alpha60(self):
        res = ts_sum(((self.close - self.low) - (self.high - self.close)) / (self.high - self.low) * self.volume, 20)
        return res

    def GT_alpha61(self):
        a = rank(decay_linear(delta(self.VWAP, 1), 12))
        b = rank(decay_linear(rank(correlation(self.low, sma(self.volume, 80), 8)), 17))
        res = a.clip_lower(b.fillna(-np.inf)) * -1
        return res

    def GT_alpha62(self):
        res = -1 * correlation(self.high, rank(self.volume), 5)
        return res

    def GT_alpha63(self):
        tmp = self.close - delay(self.close, 1)
        a = tmp.clip_lower(0)
        b = abs(tmp)

        res = gt_sma(a, 6, 1) / gt_sma(b, 6, 1)
        return res * 100

    def GT_alpha64(self):
        a = rank(decay_linear(correlation(rank(self.VWAP), rank(self.volume), 4), 4))
        b = rank(decay_linear(correlation(rank(self.close), rank(sma(self.volume, 60)), 4).clip_lower(13), 14))
        res = a.clip_lower(b.fillna(-np.inf)) * -1
        return res

    def GT_alpha65(self):
        return sma(self.close, 6) / self.close

    def GT_alpha66(self):
        return (self.close - sma(self.close, 6)) / sma(self.close, 6) * 100

    def GT_alpha67(self):
        tmp = self.close - delay(self.close, 1)
        a = cmp_max(tmp, 0)
        b = abs(tmp)
        return gt_sma(a, 24, 1) / gt_sma(b, 24, 1) * 100

    def GT_alpha68(self):
        a = (self.high + self.low) / 2
        b = (delay(self.high, 1) + delay(self.low, 1)) / 2
        res = (a - b) * (self.high - self.low) / self.volume
        return gt_sma(res, 15, 2)

    def GT_alpha69(self):
        pos_D = self.open <= delay(self.open, 1)
        DTM = cmp_max(self.high - self.open, self.open - delay(self.open, 1))
        DTM[pos_D] = 0
        DBM = cmp_max(self.open - self.high, self.open - delay(self.open, 1))
        DBM[pos_D] = 0

        sum_DTM = ts_sum(DTM, 20)
        sum_DBM = ts_sum(DBM, 20)
        pos2 = sum_DTM == sum_DBM
        res2 = (sum_DTM - sum_DBM) / sum_DBM
        res2[pos2] = 0

        pos1 = sum_DTM > sum_DBM
        res1 = res2
        res1[pos1] = ((sum_DTM - sum_DBM) / sum_DTM)[pos1]
        return res1

    def GT_alpha70(self):
        return stddev(self.turnoverValue, 6)

    def GT_alpha71(self):
        return (self.close - sma(self.close, 24)) / sma(self.close, 24) * 100

    def GT_alpha72(self):
        a = ts_max(self.high, 6) - self.close
        b = ts_max(self.high, 6) - ts_min(self.low, 6)
        return gt_sma(a / b * 100, 15, 1)

    def GT_alpha73(self):
        a = ts_rank(decay_linear(decay_linear(correlation(self.close, self.volume, 10), 16), 4), 5)
        b = rank(decay_linear(correlation(self.VWAP, sma(self.volume, 30), 4), 3))

        return b - a

    def GT_alpha74(self):
        a = rank(correlation(ts_sum(self.low * 0.35 + self.VWAP * 0.65, 20), ts_sum(sma(self.volume, 40), 20), 7))
        b = rank(correlation(rank(self.VWAP), rank(self.volume), 6))
        return a + b

    def _GT_alpha75(self, benchmark):
        benchmark_data = DataAPI.MktIdxdGet(tradeDate=u"",
                                            indexID=u"",
                                            ticker=benchmark,
                                            beginDate=self.dates[0],
                                            endDate=self.dates[-1],
                                            exchangeCD=u"XSHE,XSHG",
                                            field=u"tradeDate,closeIndex,openIndex",
                                            pandas="1").set_index('tradeDate')
        pos1 = self.close > self.open
        pos2 = benchmark_data['closeIndex'] > benchmark_data['openIndex']
        res1 = count(pos1.loc[pos2].reindex(index=self.dates).fillna(False), 50)
        res2 = count(pos2, 50)
        return res1.div(res2, axis=0)

    def GT_alpha75_300(self):
        return self._GT_alpha75('000300')

    def GT_alpha75_500(self):
        return self._GT_alpha75('000905')

    def GT_alpha76(self):
        tmp = abs(self.close / delay(self.close) - 1) / self.volume
        return stddev(tmp, 20) / sma(tmp, 20)

    def GT_alpha77(self):
        a = rank(decay_linear((self.high + self.low) / 2 + self.high - (self.VWAP + self.high), 20))
        b = rank(decay_linear(correlation((self.high + self.low) / 2, sma(self.volume, 40), 3), 6))

        return cmp_min(a, b)

    # def GT_alpha78(self):
    #     tmp = (self.high+self.low+self.close)/3

    def GT_alpha79(self):
        tmp = self.close - delay(self.close, 1)
        a = cmp_max(tmp, 0)
        b = abs(tmp)
        return gt_sma(a, 12, 1) / gt_sma(b, 12, 1) * 100

    def GT_alpha80(self):
        return (self.volume - delay(self.volume, 5)) / delay(self.volume, 5) * 100

    def GT_alpha81(self):
        return gt_sma(self.volume, 21, 2)

    def GT_alpha82(self):
        a = ts_max(self.high, 6) - self.close
        b = ts_max(self.high, 6) - ts_max(self.low, 6)
        return gt_sma(a / b * 100, 20, 1)

    def GT_alpha83(self):
        return -1 * rank(covariance(rank(self.high), rank(self.volume), 5))

    def GT_alpha84(self):
        pos2 = self.close < delay(self.close, 1)
        res2 = -self.volume.copy()
        res2[~pos2] = 0

        pos1 = self.close > delay(self.close, 1)
        res1 = self.volume.copy()
        res1[~pos1] = res2[~pos1]

        return ts_sum(res1, 20)

    def GT_alpha85(self):
        return ts_rank(self.volume / sma(self.volume, 20), 20) * ts_rank(-1 * delta(self.close, 7), 8)

    def GT_alpha86(self):
        tmp = (delay(self.close, 20) - delay(self.close, 10)) / 10 - (delay(self.close, 10) - self.close) / 10
        pos2 = tmp < 0
        res = delay(self.close, 1) - self.close
        res[pos2] = 1

        pos1 = tmp > 0.25
        res[pos1] = -1
        return res

    def GT_alpha87(self):
        a = rank(decay_linear(delta(self.VWAP, 4), 7))
        b = ts_rank(
            decay_linear((self.low * 0.9 + self.high * 0.1 - self.VWAP) / (self.open - (self.high + self.low) / 2), 11),
            7)
        return -(a + b)

    def GT_alpha88(self):
        return (self.close - delay(self.close, 20)) / delay(self.close, 20) * 100

    def GT_alpha89(self):
        a = gt_sma(self.close, 13, 2)
        b = gt_sma(self.close, 27, 2)
        c = gt_sma(gt_sma(self.close, 13, 2) - gt_sma(self.close, 27, 2), 10, 2)
        return 2 * (a - b - c)

    def GT_alpha90(self):
        return -1 * rank(correlation(rank(self.VWAP), rank(self.volume), 5))

    def GT_alpha91(self):
        a = rank(self.close - self.close.clip_lower(5))
        b = rank(correlation(sma(self.volume, 40), self.low, 5))
        return -1 * a * b

    def GT_alpha92(self):
        a = rank(decay_linear(delta(self.close * 0.35 + self.VWAP * 0.65, 2), 3))
        b = ts_rank(decay_linear(abs(correlation(sma(self.volume, 180), self.close, 13)), 5), 15)
        return a.clip_lower(b.fillna(-np.inf)) * -1

    def GT_alpha93(self):
        pos = self.open >= delay(self.open, 1)
        res = (self.open - self.low).clip_lower((self.open - delay(self.open, 1)).fillna(-np.inf))
        res[pos] = 0
        return ts_sum(res, 20)

    def GT_alpha94(self):
        pos2 = self.close < delay(self.close, 1)
        res2 = -self.volume.copy()
        res2[~pos2] = 0

        pos1 = self.close > delay(self.close, 1)
        res1 = res2
        res1[pos1] = self.volume[pos1]

        return ts_sum(res1, 30)

    def GT_alpha95(self):
        return stddev(self.turnoverValue, 20)

    def GT_alpha96(self):
        a = self.close - ts_min(self.low, 9)
        b = ts_max(self.high, 9) - ts_min(self.low, 9)
        return gt_sma(gt_sma(a / b * 100, 3, 1), 3, 1)

    def GT_alpha97(self):
        return stddev(self.volume, 10)

    def GT_alpha98(self):
        pos1 = (delta(ts_sum(self.close, 100) / 100, 100) / delay(self.close, 100) < 0.05)
        pos2 = delta(ts_sum(self.close, 100), 100) / delay(self.close, 100) == 0.05
        pos = pos1 | pos2
        res = -1 * delta(self.close, 3)
        res[pos] = (-1 * (self.close - ts_min(self.close, 100)))[pos]
        return res

    def GT_alpha99(self):
        return -1 * rank(covariance(rank(self.close), rank(self.volume), 5))

    def GT_alpha100(self):
        return stddev(self.volume, 20)

    def GT_alpha101(self):
        a = rank(correlation(self.close, ts_sum(sma(self.volume, 30), 37), 15))
        b = rank(correlation(rank(self.high * 0.1 + self.VWAP * 0.9), rank(self.volume), 11))
        return (a < b).astype(int) * -1

    def GT_alpha102(self):
        tmp = self.volume - delay(self.volume, 1)
        a = cmp_max(tmp, 0)
        b = abs(tmp)
        return gt_sma(a, 6, 1) / gt_sma(b, 6, 1) * 100

    def GT_alpha103(self):
        return (20 - ts_argmin(self.low, 20)) / 20 * 100

    def GT_alpha104(self):
        a = delta(correlation(self.high, self.volume, 5), 5)
        b = rank(stddev(self.close, 20))
        return -a * b

    def GT_alpha105(self):
        return -1 * correlation(rank(self.open), rank(self.volume), 10)

    def GT_alpha106(self):
        return self.close - delay(self.close, 20)

    def GT_alpha107(self):
        a = rank(self.open - delay(self.high, 1))
        b = rank(self.open - delay(self.close, 1))
        c = rank(self.open - delay(self.low, 1))
        return -a * b * c

    def GT_alpha108(self):
        a = rank(self.high - cmp_min(self.high, 2))
        b = rank(correlation(self.VWAP, sma(self.volume, 120), 6))
        return -1 * a.pow(b)

    def GT_alpha109(self):
        tmp = gt_sma(self.high - self.low, 10, 2)
        return tmp / gt_sma(tmp, 10, 2)

    def GT_alpha110(self):
        a = cmp_max(self.high - delay(self.close, 1), 0)
        b = cmp_max(delay(self.close, 1) - self.low, 0)
        return ts_sum(a, 20) / ts_sum(b, 20) * 100

    def GT_alpha111(self):
        tmp = self.volume * ((self.close - self.low) - (self.high - self.close)) / (self.high - self.low)
        return gt_sma(tmp, 11, 2) - gt_sma(tmp, 4, 2)

    def GT_alpha112(self):
        tmp = self.close - delay(self.close, 1)
        pos1 = tmp > 0
        pos2 = tmp < 0
        res1 = tmp.copy()
        res2 = abs(tmp)

        res1[~pos1] = 0
        res2[~pos2] = 0

        a = ts_sum(res1)
        b = ts_sum(res2)

        return (a - b) / (a + b) * 100

    def GT_alpha113(self):
        a = rank(ts_sum(delay(self.close, 5), 20) / 20)
        b = correlation(self.close, self.volume, 2)
        c = rank(correlation(ts_sum(self.close, 5), ts_sum(self.close, 20), 2))
        return -a * b * c

    def GT_alpha114(self):
        a = rank(delay((self.high - self.low) / (ts_sum(self.close, 5) / 5), 2))
        b = rank(rank(self.volume))
        c = (self.high - self.low) / (ts_sum(self.close, 5) / 5)
        d = self.VWAP - self.close
        return a * b / (c / d)

    def GT_alpha115(self):
        a = rank(correlation(self.high * 0.9 + self.close * 0.1, sma(self.volume, 30), 10))
        b = rank(correlation(ts_rank((self.high + self.low) / 2, 4), ts_rank(self.volume, 10), 7))
        return a.pow(b)

    def GT_alpha116(self):
        X = self.close[-20:].dropna(how='all', axis=1).fillna(0)
        Y = range(1, 21)
        return X.apply(lambda x: sm.OLS(Y, x).fit().params[0]).to_frame().T

    def GT_alpha117(self):
        a = ts_rank(self.volume, 32)
        b = 1 - ts_rank(self.close + self.high - self.low, 16)
        c = 1 - ts_rank(self.returns, 32)
        return a * b * c

    def GT_alpha118(self):
        return ts_sum(self.high - self.open, 20) / ts_sum(self.open - self.low, 20) * 100

    def GT_alpha119(self):
        a = rank(decay_linear(correlation(self.VWAP, ts_sum(sma(self.volume, 5), 26), 5), 7))
        b = rank(decay_linear(ts_rank(cmp_min(correlation(rank(self.open), rank(sma(self.volume, 15)), 21), 9), 7), 8))
        return a - b

    def GT_alpha120(self):
        return rank(self.VWAP - self.close) / rank(self.VWAP + self.close)

    def GT_alpha121(self):
        a = rank(self.VWAP - cmp_min(self.VWAP, 12))
        b = ts_rank(correlation(ts_rank(self.VWAP, 20), ts_rank(sma(self.volume, 60), 2), 18), 3)
        return -1 * a.pow(b)

    def GT_alpha122(self):
        tmp = gt_sma(gt_sma(gt_sma(np.log(self.close), 13, 2), 13, 2), 13, 2)
        return (tmp - delay(tmp, 1)) / delay(tmp, 1)

    def GT_alpha123(self):
        a = rank(correlation(ts_sum((self.high + self.low) / 2, 20), ts_sum(sma(self.volume, 60), 20), 9))
        b = rank(correlation(self.low, self.volume, 6))
        res = a < b
        return -1 * res.astype(int)

    def GT_alpha124(self):
        return (self.close - self.VWAP) / decay_linear(rank(ts_max(self.close, 30)), 2)

    def GT_alpha125(self):
        a = rank(decay_linear(correlation(self.VWAP, sma(self.volume, 80), 17), 20))
        b = rank(decay_linear(delta(0.5 * self.close + 0.5 * self.VWAP, 3), 16))
        return a / b

    def GT_alpha126(self):
        return (self.close + self.high + self.low) / 3

    def GT_alpha127(self):
        return sma(100 * (self.close - cmp_max(self.close, 12)) / cmp_max(self.close, 12), 2).pow(0.5)

    def GT_alpha128(self):
        tmp = (self.close + self.high + self.low) / 3
        pos = tmp > delay(tmp, 1)
        res1 = tmp * self.volume
        res2 = tmp * self.volume

        res1[~pos] = 0
        res2[pos] = 0
        return 100 - 100 / (1 + ts_sum(res1, 14) / ts_sum(res2, 14))

    def GT_alpha129(self):
        tmp = self.close - delay(self.close, 1)
        pos = tmp < 0
        res = abs(tmp)
        res[~pos] = 0
        return ts_sum(res, 12)

    def GT_alpha130(self):
        a = rank(decay_linear(correlation((self.high + self.low) / 2, sma(self.volume, 40), 9), 10))
        b = rank(decay_linear(correlation(rank(self.VWAP), rank(self.volume), 7), 3))
        return a / b

    def GT_alpha131(self):
        a = rank(delta(self.VWAP, 1))
        b = ts_rank(correlation(self.close, sma(self.volume, 50), 18), 18)
        return a.pow(b)

    def GT_alpha132(self):
        return sma(self.turnoverValue, 20)

    def GT_alpha133(self):
        a = (20 - ts_argmax(self.high, 20) / 20)
        b = (20 - ts_argmin(self.low, 20) / 20)
        return (a - b) * 100

    def GT_alpha134(self):
        return (self.close - delay(self.close, 12)) / delay(self.close, 12) * self.volume

    def GT_alpha135(self):
        return gt_sma(delay(self.close / delay(self.close, 20), 1), 20, 1)

    def GT_alpha136(self):
        return -1 * rank(delta(self.returns, 3)) * correlation(self.open, self.volume, 10)

    # def GT_alpha137(self):

    def GT_alpha138(self):
        a = rank(decay_linear(delta(self.low * 0.7 + self.VWAP * 0.3, 3), 20))
        b = ts_rank(
            decay_linear(ts_rank(correlation(ts_rank(self.low, 8), ts_rank(sma(self.volume, 60), 17), 5), 19), 16), 7)
        return b - a

    def GT_alpha139(self):
        return -1 * correlation(self.open, self.volume, 10)

    def GT_alpha140(self):
        a = rank(decay_linear(rank(self.open) + rank(self.low) - (rank(self.high) + rank(self.close)), 8))
        b = ts_rank(decay_linear(correlation(ts_rank(self.close, 8), ts_rank(sma(self.volume, 60), 20), 8), 7), 3)
        return cmp_min(a, b)

    def GT_alpha141(self):
        return rank(correlation(rank(self.high), rank(sma(self.volume, 15)), 9)) * -1

    def GT_alpha142(self):
        a = rank(ts_rank(self.close, 10))
        b = rank(delta(delta(self.close, 1), 1))
        c = rank(ts_rank(self.volume / sma(self.volume, 20), 5))

        return -1 * a * b * c

    # def GT_alpha143(self):
    def GT_alpha144(self):
        a = sumif(abs(self.close / delay(self.close, 1) - 1) / self.turnoverValue, self.close < delay(self.close, 1),
                  20)
        b = count(self.close < delay(self.close, 1), 20)
        return a / b

    def GT_alpha145(self):
        return (sma(self.volume, 9) - sma(self.volume, 26)) / sma(self.volume, 12) * 100

    def GT_alpha146(self):
        tmp1 = (self.close - delay(self.close, 1)) / delay(self.close, 1)
        tmp2 = gt_sma(tmp1, 61, 2)
        tmp3 = tmp1 - tmp2
        a = sma(tmp3) * tmp3
        b = sma((tmp1 - tmp3).pow(2), 60)
        return a / b

    def GT_alpha147(self):
        Y = range(1, 13)
        X = sma(self.close, 12).iloc[-12:].dropna(how='all', axis=1).fillna(0)
        return X.apply(lambda x: sm.OLS(Y, x).fit().params[0]).to_frame().T

    def GT_alpha148(self):
        a = rank(correlation(self.open, ts_sum(sma(self.volume, 60), 9), 6))
        b = rank(self.open - ts_min(self.open, 14))
        res = a < b
        return -1 * res.astype(int)

    # def GT_alpha149(self):

    def GT_alpha150(self):
        return (self.close + self.high + self.low) / 3 * self.volume

    def GT_alpha151(self):
        return gt_sma(self.close - delay(self.close, 20), 20, 1)

    def GT_alpha152(self):
        tmp = delay(gt_sma(delay(self.close / delay(self.close, 9), 1), 9, 1), 1)
        return gt_sma(sma(tmp, 12) - sma(tmp, 26), 9, 1)

    def GT_alpha153(self):
        return (sma(self.close, 3) + sma(self.close, 6) + sma(self.close, 12) + sma(self.close, 24)) / 4

    def GT_alpha154(self):
        a = self.VWAP - cmp_min(self.VWAP, 16)
        b = correlation(self.VWAP, sma(self.volume, 180), 18)
        res = a < b
        return res.astype(int)

    def GT_alpha155(self):
        tmp = gt_sma(self.volume, 13, 2) - gt_sma(self.volume, 27, 2)
        return tmp - gt_sma(tmp, 10, 2)

    def GT_alpha156(self):
        a = rank(decay_linear(delta(self.VWAP, 5), 3))
        b = rank(
            decay_linear((delta(self.open * 0.15 + self.low * 0.85, 2) / (self.open * 0.15 + self.low * 0.85) * -1), 3))
        return -1 * cmp_max(a, b)

    def GT_alpha157(self):
        a = cmp_min(
            product(rank(rank(np.log(ts_sum(ts_min(rank(rank(-1 * rank(delta(self.close - 1, 5)))), 2), 1)))), 1), 5)
        b = ts_rank(delay(-1 * self.returns, 6), 5)
        return a + b

    def GT_alpha158(self):
        tmp = gt_sma(self.close, 15, 2)
        return (self.high - tmp - (tmp - self.low)) / self.close

    def GT_alpha159(self):
        tmp1 = cmp_min(self.low, delay(self.close, 1))
        tmp2 = cmp_max(self.high, delay(self.close, 1))
        a = (self.close - ts_sum(tmp1, 6)) / ts_sum(tmp2 - tmp1, 6) * 24 * 12
        b = (self.close - ts_sum(tmp1, 12)) / ts_sum(tmp2 - tmp1, 12) * 6 * 24
        c = (self.close - ts_sum(tmp1, 24)) / ts_sum(tmp2 - tmp1, 24) * 6 * 24
        return (a + b + c) * 100 / (6 * 12 + 6 * 24 + 12 * 24)

    def GT_alpha160(self):
        pos = self.close <= delay(self.close, 1)
        res = stddev(self.close, 20)
        res[~pos] = 0
        return gt_sma(res, 20, 1)

    def GT_alpha161(self):
        a = cmp_max(self.high - self.low, abs(delay(self.close, 1) - self.high))
        b = abs(delay(self.close, 1) - self.low)
        return sma(cmp_max(a, b), 12)

    def GT_alpha162(self):
        tmp = self.close - delay(self.close, 1)
        tmp_max = cmp_max(tmp, 0)
        tmp_sma_max = gt_sma(tmp_max, 12, 1)
        tmp_sma_max_div = tmp_sma_max / gt_sma(abs(tmp), 12, 1) * 100
        a = tmp_sma_max_div - cmp_min(tmp_sma_max_div, 12)
        b = cmp_max(tmp_sma_max_div, 12) - cmp_min(tmp_sma_max_div, 12)
        return a / b

    def GT_alpha163(self):
        return rank(-self.returns * sma(self.volume, 20) * self.VWAP * (self.high - self.close))

    def GT_alpha164(self):
        pos = self.close > delay(self.close, 1)
        res = 1 / (self.close - delay(self.close, 1))
        res[~pos] = 1
        return gt_sma((res - cmp_min(res, 12)) / (self.high - self.low) * 100, 13, 2)

    # def GT_alpha165(self):
    # def GT_alpha166(self):
    #     tmp = self.close/delay(self.close, 1)
    #     a = -20 * (20-1) ** 1.5
    #     b = ts_sum(tmp-1-sma(tmp-1, 20), 20)
    #     c = (20-2)*ts_sum()

    def GT_alpha167(self):
        return ts_sum(cmp_max(self.close - delay(self.close, 1), 0), 12)

    def GT_alpha168(self):
        return -self.volume / sma(self.volume, 20)

    def GT_alpha169(self):
        tmp = delay(gt_sma(self.close - delay(self.close, 1), 9, 1), 1)
        return gt_sma(sma(tmp, 12) - sma(tmp, 26), 10, 1)

    def GT_alpha170(self):
        a = rank((1 / self.close) * self.volume) / sma(self.volume, 20)
        b = (self.high * rank(self.high - self.close)) / (ts_sum(self.high, 5) / 5)
        c = rank(self.VWAP - delay(self.VWAP, 5))
        return a * b - c

    def GT_alpha171(self):
        return -(self.low - self.close) * self.open.pow(5) / ((self.close - self.high) * self.close.pow(5))

    def GT_alpha172(self):
        TR = cmp_max(cmp_max(self.high - self.low, abs(self.high - delay(self.close, 1))),
                     abs(self.low - delay(self.close, 1)))
        HD = self.high - delay(self.high, 1)
        LD = delay(self.low, 1) - self.low

        pos1 = (LD > 0) & (LD > HD)
        pos2 = (HD > 0) & (HD > LD)

        res1 = LD
        res2 = HD
        res1[~pos1] = 0
        res2[~pos2] = 0

        tmp = ts_sum(TR, 14)
        a = ts_sum(res1, 14) * 100 / tmp
        b = ts_sum(res2, 14) * 100 / tmp

        return sma(abs((a - b) / (a + b)), 6)

    def GT_alpha173(self):
        a = 3 * gt_sma(self.close, 13, 2)
        b = 2 * gt_sma(gt_sma(self.close, 13, 2), 13, 2)
        c = gt_sma(gt_sma(gt_sma(np.log(self.close), 13, 2), 13, 2), 13, 2)
        return a - b + c

    def GT_alpha174(self):
        pos = self.close > delay(self.close, 1)
        res = stddev(self.close, 20)
        res[~pos] = 0
        return gt_sma(res, 20, 1)

    def GT_alpha175(self):
        a = cmp_max(self.high - self.low, abs(delay(self.close, 1) - self.high))
        b = cmp_max(a, abs(delay(self.close, 1) - self.low))
        return sma(b, 6)

    def GT_alpha176(self):
        a = rank((self.close - ts_min(self.low, 12)) / (ts_max(self.high, 12) - ts_min(self.low, 12)))
        b = rank(self.volume)
        return correlation(a, b, 6)

    def GT_alpha177(self):
        return (20 - ts_argmax(self.high, 20)) / 20 * 100

    def GT_alpha178(self):
        return (self.close - delay(self.close, 1)) / delay(self.close, 1) * self.volume

    def GT_alpha179(self):
        a = rank(correlation(self.VWAP, self.volume, 4))
        b = rank(correlation(rank(self.low), rank(sma(self.volume, 50)), 12))
        return a * b

    def GT_alpha180(self):
        pos = sma(self.volume, 20) < self.volume
        res = -self.volume.copy()
        res[pos] = -sign(delta(self.close, 7)) * ts_rank(abs(delta(self.close, 7)), 60)[pos]
        return res

    def _GT_alpha182(self, benchmark):
        benchmark_data = DataAPI.MktIdxdGet(tradeDate=u"",
                                            indexID=u"",
                                            ticker=benchmark,
                                            beginDate=self.dates[0],
                                            endDate=self.dates[-1],
                                            exchangeCD=u"XSHE,XSHG",
                                            field=u"tradeDate,closeIndex,openIndex",
                                            pandas="1").set_index('tradeDate')
        pos1 = benchmark_data['closeIndex'] > benchmark_data['openIndex']
        pos2 = self.close > self.open
        res1 = pos2.loc[pos1].reindex(index=self.dates).fillna(False)
        res2 = (~pos2).loc[~pos1].reindex(index=self.dates).fillna(False)
        return count(res1 | res2, 20) / 20

    def GT_alpha182_300(self):
        return self._GT_alpha182('000300')

    def GT_alpha182_500(self):
        return self._GT_alpha182('000905')

    def GT_alpha184(self):
        return rank(correlation(delay(self.open - self.close, 1), self.close, 200)) + rank(self.open - self.close)

    def GT_alpha185(self):
        return rank(-1 * (1 - self.open / self.close).pow(2))

    def GT_alpha186(self):
        res = self.GT_alpha172()
        return (res + delay(res, 6)) / 2

    def GT_alpha187(self):
        pos = self.open <= delay(self.open, 1)
        res = cmp_max(self.high - self.open, self.open - delay(self.open, 1))
        res[pos] = 0
        return ts_sum(res, 20)

    def GT_alpha188(self):
        tmp = gt_sma(self.high - self.low, 11, 2)
        return (self.high - self.low - tmp) / tmp * 100

    def GT_alpha189(self):
        return sma(abs(self.close - sma(self.close, 6)), 6)

    def GT_alpha190(self):
        tmp = self.close / delay(self.close, 19)
        tmp1 = tmp - 1
        tmp2 = tmp.pow(1.0 / 20) - 1

        condition1 = tmp1 > tmp2
        condition2 = tmp1 < tmp2
        tmp_count1 = count(condition1, 20)
        tmp_count2 = count(condition2, 20)
        tmp_sumif1 = sumif(tmp1 - tmp2.pow(2), condition1, 20)
        tmp_sumif2 = sumif(tmp1 - tmp2.pow(2), condition2, 20)
        return np.log(-tmp_count1 * tmp_sumif1 / (tmp_count2 * tmp_sumif2))

    def GT_alpha191(self):
        return correlation(sma(self.volume, 20), self.low, 5) + (self.high + self.low) / 2 - self.close

    def RSRS(self, n=20, m=40):
        panel = pd.concat([self.high, self.low], keys=['high', 'low']).to_panel()

        def _rsrs(df):
            df = df.T
            try:
                res = pd.ols(y=df['high'], x=df['low'], window_type='rolling', window=n)
                z_score = res.beta['x'].rolling(m).apply(lambda x: (x[-1] - x.mean()) / x.std())
                return z_score * res.r2
            except ValueError:
                return

        return panel.apply(_rsrs, axis=(1, 2))

    def TK(self, a=0.88, lbd=2.25, gamma=0.61, delta=0.69, rho=0.98):
        tmp_5 = (self.returns + 1).rolling(5).apply(lambda x: x.prod()) - 1
        week_ret = tmp_5.iloc[:-300:-5][::-1]
        pos = week_ret >= 0
        rho_list = [rho ** (i + 1) for i in range(60)][::-1]

        pos_ret = week_ret[pos]
        neg_ret = week_ret[~pos]

        pos_v = pos_ret.pow(a)
        neg_v = (-neg_ret).pow(a) * (-lbd)

        def w_(p, power):
            return p.pow(power) / (p.pow(power) + (1 - p).pow(power)).pow(1.0 / power)

        pos_w = w_((pos_ret.count() - pos_ret.rank() + 1) / 60, delta) - w_((pos_ret.count() - pos_ret.rank()) / 60,
                                                                            delta)
        neg_w = w_((neg_ret.count() - neg_ret.rank(ascending=False) + 1) / 60, gamma) - w_(
            (neg_ret.count() - neg_ret.rank(ascending=False)) / 60, gamma)

        return (pos_v * pos_w).multiply(rho_list, axis=0).sum() + (neg_v * neg_w).multiply(rho_list, axis=0).sum()


class MinAlphas(object):
    def __init__(self, min_data):
        data = min_data
        self.secID = data.minor_axis.tolist()
        self.minutes = data.major_axis
        self.open = data['open']
        self.high = data['high']
        self.low = data['low']
        self.close = data['close']
        self.volume = data['volume']
        self.turnoverValue = data['turnoverValue']
        self.returns = self.close.pct_change()

        self.daily_volume_group = group_by_day(self.volume)

    def vol_ratio(self):
        def _vol(gp):
            gp.index = gp.index.map(lambda x: x.split(' ')[-1])
            ma = gp.loc[(gp.index <= '10:00') | (gp.index >= '14:30')]
            return ma.sum() / gp.sum()

        return self.daily_volume_group.apply(_vol).mean()

    def vol_ratio_10(self):
        tradeDate = cal.adjustDate(self.minutes[-1]).strftime("%Y-%m-%d")
        tradeDate_10 = cal.advanceDate(tradeDate, "-10B").strftime("%Y-%m-%d")
        min_data = get_data_cube(self.secID,
                                 field=['close', 'high', 'open', 'low', 'volume', 'turnoverValue'],
                                 start=tradeDate_10,
                                 end=tradeDate,
                                 freq='5m',
                                 style='ast')
        obj = MinAlphas(min_data)
        return obj.vol_ratio()

    def vol_ratio_15(self):
        tradeDate = cal.adjustDate(self.minutes[-1]).strftime("%Y-%m-%d")
        tradeDate_15 = cal.advanceDate(tradeDate, "-15B").strftime("%Y-%m-%d")
        min_data = get_data_cube(self.secID,
                                 field=['close', 'high', 'open', 'low', 'volume', 'turnoverValue'],
                                 start=tradeDate_15,
                                 end=tradeDate,
                                 freq='5m',
                                 style='ast')
        obj = MinAlphas(min_data)
        return obj.vol_ratio()

    def vol_ratio_20(self):
        tradeDate = cal.adjustDate(self.minutes[-1]).strftime("%Y-%m-%d")
        tradeDate_20 = cal.advanceDate(tradeDate, "-20B").strftime("%Y-%m-%d")
        min_data = get_data_cube(self.secID,
                                 field=['close', 'high', 'open', 'low', 'volume', 'turnoverValue'],
                                 start=tradeDate_20,
                                 end=tradeDate,
                                 freq='5m',
                                 style='ast')
        obj = MinAlphas(min_data)
        return obj.vol_ratio()

    def vol_ratio_25(self):
        tradeDate = cal.adjustDate(self.minutes[-1]).strftime("%Y-%m-%d")
        tradeDate_25 = cal.advanceDate(tradeDate, "-25B").strftime("%Y-%m-%d")
        min_data = get_data_cube(self.secID,
                                 field=['close', 'high', 'open', 'low', 'volume', 'turnoverValue'],
                                 start=tradeDate_25,
                                 end=tradeDate,
                                 freq='5m',
                                 style='ast')
        obj = MinAlphas(min_data)
        return obj.vol_ratio()

    def vol_ratio_30(self):
        tradeDate = cal.adjustDate(self.minutes[-1]).strftime("%Y-%m-%d")
        tradeDate_30 = cal.advanceDate(tradeDate, "-30B").strftime("%Y-%m-%d")
        min_data = get_data_cube(self.secID,
                                 field=['close', 'high', 'open', 'low', 'volume', 'turnoverValue'],
                                 start=tradeDate_30,
                                 end=tradeDate,
                                 freq='5m',
                                 style='ast')
        obj = MinAlphas(min_data)
        return obj.vol_ratio()

    def vol_ratio_50(self):
        tradeDate = cal.adjustDate(self.minutes[-1]).strftime("%Y-%m-%d")
        tradeDate_50 = cal.advanceDate(tradeDate, "-50B").strftime("%Y-%m-%d")
        min_data = get_data_cube(self.secID,
                                 field=['close', 'high', 'open', 'low', 'volume', 'turnoverValue'],
                                 start=tradeDate_50,
                                 end=tradeDate,
                                 freq='5m',
                                 style='ast')
        obj = MinAlphas(min_data)
        return obj.vol_ratio()

    def total_consistent_volume(self, a=0.95):
        consistent = (self.close - self.open).abs() <= a * (self.high - self.low).abs()
        consistent_volume = self.volume[consistent]
        grouped = group_by_day(consistent_volume)
        tot_volume = self.daily_volume_group.sum()
        return (grouped.sum() / tot_volume).mean()

    def total_consistent_volume_rise(self, a=0.95):
        consistent = (self.close - self.open).abs() <= a * (self.high - self.low).abs()
        consistent_volume = self.volume[consistent & (self.close >= self.open)]
        grouped = group_by_day(consistent_volume)
        tot_volume = self.daily_volume_group.sum()
        return (grouped.sum() / tot_volume).mean()

    def total_consistent_volume_fall(self, a=0.95):
        consistent = (self.close - self.open).abs() <= a * (self.high - self.low).abs()
        consistent_volume = self.volume[consistent & (self.close <= self.open)]
        grouped = group_by_day(consistent_volume)
        tot_volume = self.daily_volume_group.sum()
        return (grouped.sum() / tot_volume).mean()

    def min_illiquidity(self):
        return (self.close.iloc[-1] / self.close.iloc[-7] - 1) / self.turnoverValue.iloc[-6:].sum() * 1e11

    def min_reverse(self):
        tmp_ret = self.close.iloc[-1] / self.close.iloc[-7] - 1
        tmp_ma = (self.turnoverValue.iloc[-6:].sum() / self.volume.iloc[-6:].sum())
        return standardize(winsorize(tmp_ret)) + standardize(winsorize(tmp_ret / tmp_ma))


class MachineLearning(object):
    path = 'run/ml obj'

    def __init__(self, tradeDate):
        obj = pd.read_pickle(self.path)
        self.tradeDate = tradeDate
        self.startday = cal.advanceDate(tradeDate, '-500B').strftime("%Y-%m-%d")
        futureday = cal.advanceDate(tradeDate, '%dB' % obj.forward).strftime("%Y-%m-%d")
        self.obj = obj.redate(start=self.startday, end=futureday, period=1, forward=obj.forward)

        enddates = [cal.advanceDate(day, '%dB' % self.obj.forward).strftime("%Y-%m-%d")
                    for day in self.obj.dates]
        enddates = filter(lambda x: x < self.tradeDate, enddates)
        self.enddates = enddates
        spret_data = self.obj.get_spret_data()
        spret = spret_data.rolling(self.obj.forward).apply(lambda x: x.prod() - 1) * 100 * 250 / self.obj.forward
        tmp_spret = spret.loc[self.enddates]
        self.spret_rank = tmp_spret.rank(axis=1, pct=True)
        continuous_y = tmp_spret.stack()
        expo = self.obj.get_expo(orth=False)
        X = pd.concat(expo.values[:len(self.enddates)], keys=self.enddates)
        self.cross_index = pd.MultiIndex.from_tuples(np.intersect1d(X.index, continuous_y.index))
        self.X = X.reindex(index=self.cross_index).fillna(0)
        self.continuous_y = continuous_y.reindex(index=self.cross_index)
        self.to_predict = expo.loc[self.tradeDate]

    def _classify(self, classifier, groups):
        eps = 1.0 / groups
        spret_label = (self.spret_rank.floordiv(eps) + 1).clip_upper(groups)
        label_y = spret_label.stack()
        label_y = label_y.reindex(index=self.cross_index)
        classifier.fit(self.X, label_y)

        return classifier.predict(self.to_predict)

    def _regress(self, regressor):
        regressor.fit(self.X, self.continuous_y)
        return regressor.predict(self.to_predict)

    def _GBClassifier(self, groups):
        from sklearn.ensemble import GradientBoostingClassifier
        subsample = 1
        max_depth = 3
        gbc = GradientBoostingClassifier(subsample=subsample, max_depth=max_depth)
        return self._classify(gbc, groups)

    def _XGBoostClassifier(self, groups):
        import xgboost as xgb
        subsample = 0.95
        max_depth = 3
        xgbc = xgb.XGBClassifier(subsample=subsample, max_depth=max_depth)
        return self._classify(xgbc, groups)

    def _RFClassifier(self, groups):
        from sklearn.ensemble import RandomForestClassifier
        n_estimators = 50
        min_samples_split = 50
        min_samples_leaf = 10
        rfc = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split,
                                     min_samples_leaf=min_samples_leaf)
        return self._classify(rfc, groups)

    def GBClassifier_5(self):
        res = self._GBClassifier(5)
        return pd.Series(res, index=self.to_predict.index)

    def XGBoostClassifier_5(self):
        res = self._XGBoostClassifier(5)
        return pd.Series(res, index=self.to_predict.index)

    def RFClassifier_5(self):
        res = self._RFClassifier(5)
        return pd.Series(res, index=self.to_predict.index)

    def GBClassifier_10(self):
        res = self._GBClassifier(10)
        return pd.Series(res, index=self.to_predict.index)

    def XGBoostClassifier_10(self):
        res = self._XGBoostClassifier(10)
        return pd.Series(res, index=self.to_predict.index)

    def RFClassifier_10(self):
        res = self._RFClassifier(10)
        return pd.Series(res, index=self.to_predict.index)

    def GBClassifier_50(self):
        res = self._GBClassifier(10)
        return pd.Series(res, index=self.to_predict.index)

    def XGBoostClassifier_50(self):
        res = self._XGBoostClassifier(50)
        return pd.Series(res, index=self.to_predict.index)

    def RFClassifier_50(self):
        res = self._RFClassifier(50)
        return pd.Series(res, index=self.to_predict.index)

    def GBClassifier_100(self):
        res = self._GBClassifier(100)
        return pd.Series(res, index=self.to_predict.index)

    def XGBoostClassifier_100(self):
        res = self._XGBoostClassifier(100)
        return pd.Series(res, index=self.to_predict.index)

    def RFClassifier_100(self):
        res = self._RFClassifier(100)
        return pd.Series(res, index=self.to_predict.index)

    def GBCRegressor(self):
        from sklearn.ensemble import GradientBoostingRegressor
        subsample = 1
        max_depth = 3
        gbr = GradientBoostingRegressor(subsample=subsample, max_depth=max_depth)
        return pd.Series(self._regress(gbr), index=self.to_predict.index)

    def XGBoostRegressor(self):
        import xgboost as xgb
        subsample = 0.9
        max_depth = 3
        xgbr = xgb.XGBRegressor(subsample=subsample, max_depth=max_depth)
        return pd.Series(self._regress(xgbr), index=self.to_predict.index)

    def RFRegressor(self):
        from sklearn.ensemble import RandomForestRegressor
        n_estimators = 50
        min_samples_split = 50
        min_samples_leaf = 10
        rfr = RandomForestRegressor(n_estimators=n_estimators, min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf)
        return pd.Series(self._regress(rfr), index=self.to_predict.index)


def from_daily_alpha(secID, tradeDate, *alpha_name):
    begdate = cal.advanceDate(tradeDate, "-400B").strftime("%Y%m%d")
    data = DataAPI.MktEqudGet(secID=secID,
                              beginDate=begdate,
                              endDate=tradeDate,
                              field=u"secID,tradeDate,openPrice,closePrice,lowestPrice,highestPrice,turnoverVol,turnoverValue,turnoverRate",
                              pandas="1")
    _pn = data.set_index(['secID', 'tradeDate']).stack().reorder_levels([2, 0, 1]).unstack().T

    dates = _pn.index

    # 以当前交易日定点复权
    adj_data = DataAPI.MktAdjfGet(secID=secID,
                                  beginDate="",
                                  endDate=tradeDate,
                                  field=u"secID,adjFactor,endDate,exDivDate")

    def _get_adj_factor(group):
        # 复权因子对应的时间段为endDate至exDivDate，最后一个exDivDate至今填充为1
        res = group.set_index('endDate')['adjFactor']
        res.loc[group['exDivDate'].iloc[0]] = 1
        return res.sort_index(ascending=False).cumprod()

    a = adj_data.groupby('secID').apply(_get_adj_factor).unstack().T.ffill()
    b = a.reindex(dates).ffill()
    adj_factor = b.fillna(a.truncate(after=dates[0]).iloc[-1])
    # _pn = pn_data.transpose(1, 2, 0).to_frame().T
    assign_dict = {}
    for price_type in [u'closePrice', u'highestPrice', u'lowestPrice', u'openPrice']:
        t = _pn[price_type]
        assign_dict[price_type] = t.mul(adj_factor.reindex_like(t))

    pn_data = _pn.assign(**assign_dict).T.to_panel().transpose(2, 0, 1)

    # print 'pn data done'
    obj = Alphas(pn_data)
    arr = []
    for alp in alpha_name:
        df = eval('obj.%s()' % alp)
        print '%s alpha done' % alp

        if df is None:
            arr.append(None)
        elif isinstance(df, pd.DataFrame):
            arr.append(df.iloc[-1].rename(alp))
        else:
            arr.append(df.rename(alp))
    return pd.concat(arr, axis=1)


def from_minute_alpha(secID, tradeDate, *alpha_name):
    tradeDate_5 = cal.advanceDate(tradeDate, '-5B').strftime("%Y%m%d")
    min_data = get_data_cube(secID,
                             field=['close', 'high', 'open', 'low', 'volume', 'turnoverValue'],
                             start=tradeDate_5,
                             end=tradeDate,
                             freq='5m',
                             style='ast')
    min_obj = MinAlphas(min_data)
    arr = []
    for alp in alpha_name:
        df = eval('min_obj.%s()' % alp)
        print '%s alpha done' % alp
        if isinstance(df, pd.DataFrame):
            arr.append(df.iloc[-1].rename(alp))
        else:
            arr.append(df.rename(alp))
    return pd.concat(arr, axis=1)


def from_ml_alpha(tradeDate, *alpha_name):
    ml_obj = MachineLearning(tradeDate)
    arr = []
    for alp in alpha_name:
        df = eval('ml_obj.%s()' % alp)
        print '%s alpha done' % alp
        if isinstance(df, pd.DataFrame):
            arr.append(df.iloc[-1].rename(alp))
        else:
            arr.append(df.rename(alp))
    return pd.concat(arr, axis=1)


def get_factor_data(secID, tradeDate, field):
    if isinstance(field, str):
        field_list = [s.strip() for s in field.split(',')]
    else:
        field_list = field

    alpha_factor = ['secID'] + list(set(field_list) & set(UQER_ALPHA))
    beta_factor = ['secID'] + list(set(field_list) & set(BETA))
    self_field = list(set(field_list) - set(BETA) - set(UQER_ALPHA) - set(['secID']))

    df1 = DataAPI.MktStockFactorsOneDayProGet(tradeDate=tradeDate,
                                              secID=secID,
                                              ticker=u"",
                                              field=alpha_factor,
                                              pandas="1")
    # print 'df1 done!'
    df2 = DataAPI.RMExposureDayGet(tradeDate=tradeDate.replace('-', ''),
                                   secID=secID,
                                   ticker=u"",
                                   field=beta_factor,
                                   pandas="1")
    # print 'df2 done!'
    df = pd.merge(df1, df2, on='secID', how='inner').set_index('secID')
    # print 'df done!'

    arr = []
    alpha_factors = []
    for func in self_field:
        # print '%s try self factors' % func
        if func in globals():
            self_df = eval(func)(secID, tradeDate)
        else:
            alpha_factors.append(func)
            # print '%s is alpha' % func
            continue
        arr.append(self_df)

    if len(alpha_factors):
        day_alpha = []
        minute_alpha = []
        ml_alpha = []
        for alpha_fac in alpha_factors:
            if alpha_fac in dir(Alphas):
                day_alpha.append(alpha_fac)

            elif alpha_fac in dir(MinAlphas):
                minute_alpha.append(alpha_fac)

            elif alpha_fac in dir(MachineLearning):
                ml_alpha.append(alpha_fac)

        if len(day_alpha):
            arr.append(from_daily_alpha(secID, tradeDate, *day_alpha))

        if len(minute_alpha):
            arr.append(from_minute_alpha(secID, tradeDate, *minute_alpha))

        if len(ml_alpha):
            arr.append(from_ml_alpha(tradeDate, *ml_alpha))

    reindex_col = [col for col in field_list if col != 'secID']
    return pd.concat([df] + arr, axis=1).reindex(index=secID, columns=reindex_col)


# startday = '2013-09-23'
# secID = set_universe('ZZ500', '2013-10-14') + set_universe('HS300', '2013-10-14')
# get_factor_data(secID, startday, ['secID', 'central_degree_ZZ500'])


if __name__ == '__main__':
    pass
