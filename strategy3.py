# encoding=utf-8

from lib.self_factors import *
import pandas as pd
from pandas import DataFrame
from math import *
import numpy as np
import statsmodels.api as sm
import random
from CAL.PyCAL import *
import scipy.optimize as sco
import datetime as dt
import time
from cvxopt import matrix, solvers
from dateutil.parser import parse


# solvers.options['maxiters'] = 300
# solvers.options['abstol'] = 1e-13
# solvers.options['reltol'] = 1e-13
def FdmtEfGet_stk(stklist, end_date, nday):
    begin_date = cal.advanceDate(end_date, str(-nday) + 'B').strftime('%Y%m%d')
    events = DataAPI.FdmtEfGet(secID=stklist,
                               publishDateBegin=begin_date,
                               publishDateEnd=end_date,
                               forecastType='', field=['secID', 'actPubtime', 'forecastType'], pandas="1")
    events['actPubtime'] = events['actPubtime'].apply(
        lambda x: cal.adjustDate(x, BizDayConvention.ModifiedPreceding).strftime("%Y-%m-%d"))
    events = events[events['actPubtime'] < parse(end_date).strftime('%Y-%m-%d')]
    events['actPubtime'] = events['actPubtime'].map(lambda x: x[0:10])
    events = events.drop_duplicates(['secID', 'actPubtime'])
    # forecastType = DataAPI.SysCodeGet(codeTypeID=u"70006",field=["valueCD",u"valueName"])    
    # events = events.replace(dict(zip([int(x) for x in  forecastType["valueCD"]], forecastType["valueName"]))) 
    return events


def industry_get(stklist, date):
    # date=Date.strptime(date,'%Y-%m-%d').strftime('%Y%m%d')
    Hdf = DataAPI.EquIndustryGet(industryVersionCD=u"010303", secID=stklist, intoDate="20170801",
                                 field=['secID', 'industryID1', 'industryID2', 'industryName1'], pandas="1").set_index(
        'secID')
    Hdf['industryID1'][Hdf['industryID1'] == "01030322"] = Hdf['industryID2'][Hdf['industryID1'] == "01030322"]
    Hdf = Hdf.drop(['industryID2', 'industryName1'], 1).reindex(index=stklist)
    return Hdf


def transfer_dummy(industryDict, Hdf):
    dummy = np.zeros([len(Hdf), len(industryDict)])
    for i in range(len(Hdf)):
        dummy[i, :] = industryDict[Hdf['industryID1'].iloc[i]]
    return dummy


def industry_dict_weigth():
    df = DataAPI.IndustryGet(industryVersion=u"SW", industryVersionCD=u"", industryLevel=u"1", isNew=u"1", field=u"",
                             pandas="1")
    industryList = df['industryID'].tolist()  # .remove(1030322)#.extend(['103032201', '103032202', '103032203'])
    del industryList[21]
    industryList.extend(["0103032201", "0103032202", "0103032203"])
    industryList.sort()  # 按行业代码排序
    dummy = sm.categorical(np.array(industryList), drop=True)
    industryDict = dict(zip(industryList, dummy))
    weightBaseDf = pd.DataFrame(dict(zip(industryList, np.zeros([len(industryList), 1])))).T
    weightBaseDf.rename(columns={0: 'weight0'}, inplace=True)
    weightBaseDf.index.name = 'industryID1'
    return industryDict, weightBaseDf


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

    startDate = cal.advanceDate(date, '-9B').strftime("%Y%m%d")
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
    states = grouped.agg([np.count_nonzero, np.mean])
    stklist = states['turnoverValue'].query("count_nonzero >= 5").index.tolist()

    # 去除流动性差的股票,日均成交额大于1000万
    # tvdf['avertv'] = tvdf['turnoverValue']/tvdf['tradeDays']
    # tvdf = tvdf[tvdf['avertv'] > 1e7]
    return stklist


# 提取hs300行业权重及指数权重
def get_hH(date, index_id, weightBaseDf):  # 取得bench各行业权重和h_bench，以及股票池个股票的行业哑变量H_stk，当天停牌股票权重为0
    newdate1 = cal.advanceDate(date, '-22B', BizDayConvention.Preceding).strftime("%Y-%m-%d")
    df = DataAPI.IdxCloseWeightGet(secID=u"", ticker=index_id, beginDate=newdate1, endDate=date,
                                   field=['consID', 'weight', 'effDate'], pandas="1").set_index(['consID']).fillna(0)

    df = df.query("effDate == effDate.max()")
    # 得到stklist的行业，并pivoted,输出求解用stklist
    df.index.name = 'secID'
    # 找到当日停牌的股票，将权重设为0
    temp = DataAPI.MktEqudAdjGet(tradeDate=date, secID=df.index, field=u"secID,turnoverValue,closePrice",
                                 pandas="1").set_index(['secID']).reindex(index=df.index)  # 去除当日停牌股票
    if date != df.iloc[0]['effDate']:  # 调整最新的指数成分股权重
        temp2 = DataAPI.MktEqudAdjGet(tradeDate=df.iloc[0]['effDate'], secID=df.index, field=u"secID,closePrice",
                                      pandas="1").set_index(['secID']).reindex(index=df.index)
        df['weight'] = df['weight'] * temp['closePrice'] / temp2['closePrice']
        df = df.fillna(0.0)
        weight_sum = df['weight'].sum()
        df['weight'] = np.round(100.0 * df['weight'] / weight_sum, 3)

    stklist = temp[temp['turnoverValue'] == 0].index
    df['weight'].loc[stklist] = 0

    tempstk = df.index.tolist()
    dfind = industry_get(tempstk, date)
    dfind = dfind.sort_index(ascending=True)
    stklist = dfind.index.tolist()
    df2 = pd.concat([dfind, df], axis=1)
    df['industryID1'] = dfind['industryID1']
    dfind['weight'] = df['weight']
    dfg = df2.groupby('industryID1').sum()  # groupby自动排序
    dfg.sort_index(axis=0)  # 按行业代码排序
    weightDf = dfg.join(weightBaseDf, how='outer').fillna(0)
    h_bench = weightDf['weight'] / 100
    wb = (dfind['weight']) / 100
    wb = pd.Series.to_frame(wb)  # 获得指数成分股权重
    return h_bench, stklist, wb


def get_month_fF(X_save, Y_save, universe1, bench, datelist, factor):  # 每个截面计算f以求得协方差矩阵F,集合全部截面数据求f
    lengthF = (len(datelist) - 1)
    if len(Y_save.index) == 0:
        for i in range(lengthF):
            universe1 = set_universe('ZZ500', date=datelist[i])
            stklist = choose_stk(universe1, datelist[i])  # 用于回归的stk
            X, Y, stk = getPreFactor(stklist, bench, datelist[i], datelist[i + 1], factor)
            # X=schmidt_orth(X) ##因子正交
            X = pd.DataFrame(X, columns=[factor], index=np.repeat(datelist[i + 1], len(stk)))
            Y = pd.DataFrame(Y, columns=['Spret'], index=np.repeat(datelist[i + 1], len(stk)))
            X_save = pd.concat([X_save, X])
            Y_save = pd.concat([Y_save, Y])
    else:
        universe1 = set_universe('ZZ500', date=datelist[-1])
        stklist = choose_stk(universe1, datelist[-1])  # 用于回归的stk
        X, Y, stk = getPreFactor(stklist, bench, datelist[-2], datelist[-1], factor)
        # X=schmidt_orth(X) ##因子正交
        X = pd.DataFrame(X, columns=[factor], index=np.repeat(datelist[-1], len(stk)))
        Y = pd.DataFrame(Y, columns=['spret'], index=np.repeat(datelist[-1], len(stk)))
        X_save = pd.concat([X_save, X])
        Y_save = pd.concat([Y_save, Y])
        X_save = X_save.loc[datelist[1:]]
        Y_save = Y_save.loc[datelist[1:]]
    X_save = X_save.fillna(0.0)
    Y_save = Y_save.fillna(0.0)
    res_ols = sm.OLS(Y_save['spret'], X_save).fit()
    f = res_ols.params
    return f, stk, X_save, Y_save


def getPreFactor(stklist, bench, date1, date2, factor):
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
        df[fac] = standardize(winsorize(df[fac].replace([np.inf, -np.inf], np.nan).fillna(0.0)))  # 标准化去极值
        # df[fac]=benchmark_neutral(df[fac],bench,date1)    #基准alpha中性
        df[fac] = neutralize(df[fac].fillna(0.0),
                             target_date=cal.advanceDate(date1, ("-1B")).strftime("%Y%m%d")).fillna(0.0)  # 行业、风格alpha中性
    df['spret'] = df['spret'].fillna(0.0)
    dfArray = np.array(df)
    X = df.iloc[:, 0:-1]
    Y = pd.DataFrame(df.iloc[:, -1], columns=['spret'])
    # 去极值、中性化、标准化
    return X, Y, stk


def get_X(stklist, bench, factor, date):  # factor是传统alpha因子
    field = ['secID'] + factor
    date = Date.strptime(date, '%Y-%m-%d').strftime('%Y%m%d')
    # AFExposure= DataAPI.MktStockFactorsOneDayProGet(tradeDate=date,secID =stklist ,field=field,pandas="1").set_index(['secID']).reindex(index=stklist).fillna(0.0)
    AFExposure = get_factor_data(secID=stklist, tradeDate=date, field=field)  # 自定义因子
    for fac in factor:
        AFExposure[fac] = standardize(
            winsorize(AFExposure[fac].replace([np.inf, -np.inf], np.nan).fillna(0.0)))  # 标准化去极值
        # AFExposure[fac]=benchmark_neutral(AFExposure[fac],bench,date)    #基准alpha中性
        AFExposure[fac] = neutralize(AFExposure[fac].fillna(0.0), target_date=date).fillna(0.0)  # 行业、风格alpha中性
        AFExposure = AFExposure.fillna(0.0)
    AFExposure[factor] = schmidt_orth(np.array(AFExposure[factor]))  ##因子正交
    # AFExposure[factor[0:6]]=schmidt_orth(np.array(AFExposure[factor[0:6]])) ##因子正交
    AFExposure = AFExposure.fillna(0.0)
    return AFExposure[factor]


def benchmark_neutral(X, bench, date):  # 基准中性化
    stklist, wb = get_stkWeight(date, bench)
    stock_common = filter(lambda x: x in X.index, stklist)  # 找到共同股票
    Xbench_mean = np.dot(wb.loc[stock_common].T, np.array(X.loc[stock_common]))  # 计算指数的alpha因子暴露
    X = X - Xbench_mean
    return X


def risk_model(date, stklist, riskfactor, alphafactor):  # EARNYILD BTOP风险和收益
    field1 = ['secID'] + riskfactor
    field2 = ['Factor'] + riskfactor

    date = Date.strptime(date, '%Y-%m-%d')

    date1 = cal.advanceDate(date, (str(-(250)) + "B")).strftime("%Y%m%d")  # 使用过去1年的数据
    date = date.strftime("%Y%m%d")
    FactorRet = DataAPI.RMFactorRetDayGet(beginDate=date1, endDate=date, field=field1, pandas="1").fillna(
        0.0).mean() * 250 * 100  # 因子收益数据用最近1年的均值,年化

    FactorRet[riskfactor] = 0
    FactorRet[alphafactor] = [5.8, 5.2]

    FCovariance = DataAPI.RMCovarianceShortGet(beginDate=date, endDate=date, Factor=riskfactor, field=field2,
                                               pandas="1").set_index(['Factor']).reindex(index=riskfactor).fillna(
        0.0)  # 因子协方差数据
    RMExposure = DataAPI.RMExposureDayGet(secID=stklist, beginDate=date, endDate=date, field=field1,
                                          pandas="1").set_index(['secID']).reindex(index=stklist).fillna(
        0.0)  # 因子暴露用最新数据
    SRisk = DataAPI.RMSriskShortGet(secID=stklist, beginDate=date, endDate=date, field=["secID,tradeDate,SRISK"],
                                    pandas="1").set_index(['secID'])
    SRisk["SRISK"] = SRisk["SRISK"] ** 2
    SRisk = SRisk.reindex(stklist).fillna(0.0)
    SRisk = np.diag(SRisk["SRISK"])
    return RMExposure[riskfactor], FactorRet[riskfactor], FCovariance[riskfactor], SRisk


def objFunc(w, w0, w_last, R, V):  # 优化函数目标
    w_active = w - w0  # 主动权重
    w_change = 8 * np.sum(abs(w - w_last))
    Eret = np.dot(w_active.T, R)  # alpha因子收益加上风格因子收益
    Evar = np.dot(np.dot(w_active.T, V), w_active)  # 风格因子风险与股票特质风险
    return -1 * (Eret - Evar - w_change)


def get_SpecificRetRange(stklist, start, end):
    df = DataAPI.RMSpecificRetDayGet(secID=stklist, beginDate=start, endDate=end, field=['secID', 'tradeDate', 'spret'])
    df = df.pivot(index='tradeDate', columns='secID', values='spret') / 100 + 1  # 月收益转为日收益
    return 100 * 250 * (df.prod().rename('spret') - 1) / df.shape[0]  # 年化值


def schmidt_orth(arr):  # 施密特因子正交
    res = arr.copy()
    for i in range(arr.shape[1] - 1):
        temp = np.zeros(arr.shape[0])
        for j in range(i + 1):
            temp[:] = temp[:] + np.dot(res[:, i + 1], res[:, j]) / np.dot(res[:, j], res[:, j]) * res[:, j]
        res[:, i + 1] = res[:, i + 1] - temp[:]
        u = np.mean(res[:, i + 1])
        sigma = np.std(res[:, i + 1])
        if (sigma == 0):
            sigma = 1
        res[:, i + 1] = (res[:, i + 1] - u) / sigma  # 正交后标准化
    return res


# parameters
cal = Calendar('China.SSE')
dt = 1200
start = '2012-01-01'
end = '2017-11-10'
benchmark = 'ZZ500'  # '000906.ZICN'
universe = DynamicUniverse('HS300') + DynamicUniverse('ZZ500')
capital_base = 100000000  # 起始资金
freq = 'd'
refresh_rate = 5

forbidden = ['300166.XSHE', '002309.XSHE', '300032.XSHE', '300027.XSHE', '000778.XSHE', '002340.XSHE', '600537.XSHG',
             '002221.XSHE', '002048.XSHE', '600166.XSHG', '601777.XSHG', '600614.XSHG', '000630.XSHE', '601099.XSHG',
             '000727.XSHE', '002663.XSHE', '600068.XSHG', '002277.XSHE', '600482.XSHG', '000536.XSHE', '000793.XSHE',
             '600094.XSHG', '600773.XSHG', '600487.XSHG']
forbidden = []

factor = ['Price1M', 'MoneyFlow20', 'PEHist60', 'alpha042', 'ILLIQUIDITY', 'MOM_Est_EPS',
          'enterprise_perf']  # , 'participant', 'EquMsChanges' , 0.71, 0.71
f = [-6.47, -5.18, -4.47, 3.81, 3.41, 3.19, 3.43]

# factor = ['Price1M', 'MoneyFlow20', 'PEHist60', 'GT_alpha129', 'alpha042', 'ILLIQUIDITY', 'MOM_Est_EPS',
# 'GT_alpha111', 'GT_alpha156', 'GT_alpha128']
# f=[-7.81, -5.36, -4.78, -4.0  ,  4.36,  4.05,  3.15,  3.14,  3.12,  2.67]

# factor=['enterprise_perf']
# f=[44.35]

# factor=['shares_excit']
# f=[71.7]

riskfactor = ["EARNYILD", "BTOP", "LIQUIDTY", "MOMENTUM", "GROWTH", "BETA", "SIZE", "RESVOL", "LEVERAGE", "SIZENL",
              'Bank', 'RealEstate', 'Health', 'Transportation', 'Mining', 'NonFerMetal', 'HouseApp', 'LeiService',
              'MachiEquip', 'BuildDeco', 'CommeTrade', 'CONMAT', 'Auto', 'Textile', 'FoodBever', 'Electronics',
              'Computer', 'LightIndus', 'Utilities', 'Telecom', 'AgriForest', 'CHEM', 'Media', 'IronSteel',
              'NonBankFinan', 'ELECEQP', 'AERODEF', 'Conglomerates']
alphafactor = ["EARNYILD", "BTOP"]

commission = Commission(buycost=0.0001, sellcost=0.0011)
slippage = Slippage(value=0.0005, unit='perValue')


def initialize(account):  # 初始化虚拟账户状态
    account.X_save = pd.DataFrame()
    account.Y_save = pd.DataFrame()
    pass


def handle_data(account):  # 优矿回测是开盘价买入，不能用当天数据

    today = cal.advanceDate(account.current_date, ("-1B")).strftime("%Y-%m-%d")
    benchindex = "000905"
    universe1 = set_universe('ZZ500', date=today) + set_universe('HS300', date=today)
    datelist = choose_date(today, dt)
    print time.ctime(time.time()), 'begin!'

    industryDict, weightBaseDf = industry_dict_weigth()  # 行业字典建立

    stockset = choose_stk(universe1, today)  # 最新股票池，不包含当日停牌股票
    h_bench, stklist, wb = get_hH(today, benchindex, weightBaseDf)
    # stock_halt=filter(lambda x: x not in stockset, stklist) #找出停牌股票
    # print len(stockset),len(stklist),len(stock_halt) 

    print time.ctime(time.time()), 'alpha mdoel done!'

    # 设置初始权重，即指数成分权重
    numAsset = len(stockset)
    w0 = wb.loc[stockset]
    w0 = np.array(w0['weight'])
    w0[np.isnan(w0)] = 0.0

    # 计算当前股票池停牌股票权重
    stockholding = pd.Series(account.security_position)
    if len(stockholding) == 0:
        w_last = stockholding.reindex(index=stockset).fillna(0)
        w_halt = 0.0
    else:
        stockprice = account.get_attribute_history("closePrice", 1)
        stockprice = pd.Series(stockprice.values(), index=stockprice.keys())
        stockprice = stockprice.loc[stockholding.index]
        stockprice = pd.Series(np.array([x[0] for x in stockprice.values.tolist()]), index=stockprice.index)
        stockholding = stockholding * stockprice
        stockholding_weight = stockholding / account.reference_portfolio_value
        w_last = stockholding_weight.reindex(index=stockset).fillna(0)
        w_halt = stockholding_weight.sum() - stockholding_weight.loc[stockset].sum()

    Hdf = industry_get(stockset, end)  # 行业分类取最新数据
    H = transfer_dummy(industryDict, Hdf)  # 获得股票行业哑变量
    X = get_X(stockset, benchindex, factor, today)  # 取得股票池因子暴露
    # print X


    XRM, fRM, FRM, SRISK = risk_model(today, stockset, riskfactor, alphafactor)  # 获取股票池风格因子暴露
    XRMbench, fRMbench, FRMbench, SRISKbench = risk_model(today, stklist, riskfactor, alphafactor)  # 获取指数成分股风格因子暴露
    XRMbench_mean = np.dot(wb.T, XRMbench)  # 计算指数的风格因子暴露
    XRM = np.array(XRM) - XRMbench_mean  # 基准风格因子暴露中性化

    # 附加值函数
    CRM = np.dot(np.dot(XRM, FRM), XRM.T)
    R = np.dot(X, f) + np.dot(XRM, fRM)
    V = (0.3 * CRM + 0.3 * SRISK)

    print '----------------'
    field = ["secID,EARNYILD,BTOP,MOMENTUM,RESVOL,GROWTH,BETA,LEVERAGE,LIQUIDTY,SIZENL,SIZE"]
    ExposureX = DataAPI.RMExposureDayGet(secID=stockset, tradeDate=Date.strptime(today, '%Y-%m-%d').strftime('%Y%m%d'),
                                         field=field, pandas="1").set_index(['secID']).reindex(index=stockset).fillna(
        0.0)
    stock_common = filter(lambda x: x in stockset, stklist)  # 找到共同股票
    ExposureX_bench = np.dot(wb.loc[stock_common].T, np.array(ExposureX.loc[stock_common]))
    ExposureX = np.array(ExposureX) - ExposureX_bench  # 将风格因子基准中性化

    bound = list()
    for i in range(numAsset):
        if stockset[i] in forbidden:
            bound.append(0)
        else:
            bound.append(0.01)

    # 约束条件，依次是权重和风格中性、行业中性
    threshold = (0.05 * h_bench).clip_lower(0.005)
    # 市值中性右边界
    h1 = 0.01
    G1 = ExposureX[:, 9]
    # 市值中性左边界
    h2 = 0.01
    G2 = -ExposureX[:, 9]
    # 行业中性右边界
    h3 = threshold + h_bench
    G3 = H.T
    # 行业中性左边界
    h4 = threshold - h_bench
    G4 = -H.T

    h5 = bound
    h6 = np.zeros_like(bound)
    G5 = np.eye(len(bound))
    G6 = -np.eye(len(bound))

    h = np.hstack([h1, h2, h3, h4, h5, h6])
    G = np.vstack([G1, G2, G3, G4, G5, G6])

    # 换手率约束
    V_exchange = np.ones_like(V) * 560
    A_exchange = w_last.values * 560
    # print 'V_exchange,A_exchange'
    # print V_exchange,2*A_exchange
    # 和约束
    A = np.ones_like(w0)
    b = min(sum(w0), 1 - w_halt)

    P = matrix(2 * (V + V_exchange))
    q = matrix(-R - 2 * A_exchange)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A).T
    print A.size
    b = matrix(b)

    result = solvers.qp(P, q, G, h, A, b)
    w_opt = np.array(result['x'].T)[0]
    w_opt[w_opt < 1e-4] = 0.0

    print time.ctime(time.time()), 'Optimizer model done!'
    print result['iterations']  # 打印求解信息
    print result['status']  # 打印求解信息
    w_active = w_opt - w0  # 主动权重

    print "Factor return----------------------------"
    print  f, fRM[alphafactor]
    print "Eret----------------------------"
    print np.dot(w_active.T, np.dot(X, f)), np.dot(w_active.T, np.dot(XRM, fRM))
    print "Evar---------------------------"
    print np.dot(np.dot(w_active.T, SRISK), w_active), np.dot(np.dot(w_active.T, CRM), w_active)
    print "Turnover---------------------------"
    print np.sum(abs(w_opt - w_last))

    AlphaFactorExpo = np.dot(w_opt, X)
    StyleFactorExpo = np.dot(w_opt, ExposureX)
    print '----------------'
    print "Alpha Exposure", AlphaFactorExpo
    # print "Industry Exposure",100.0*(np.dot(w_opt,H)-h_bench)
    print "Factor Exposure", StyleFactorExpo
    print "Bench Factor Exposure", ExposureX_bench
    print "Stock Number", len(w_opt[w_opt != 0]), "Total Weight", w_opt.sum(), "Halt Weight", w_halt

    print str(today)
    buy_dict = dict(zip(stockset, w_opt))
    for key in buy_dict.keys():
        if buy_dict[key] == 0:
            buy_dict.pop(key)
            # 获取当天收盘价，看是否停牌
    dfprice = DataAPI.MktEqudAdjGet(tradeDate=today,
                                    secID=buy_dict.keys(),
                                    field=u"secID,closePrice",
                                    pandas="1").set_index(['secID'])

    # 先卖出当天没有停牌的股票
    for key in account.security_position.keys():
        if key not in buy_dict.keys():
            order_to(key, 0)

    tradelist = dfprice.index.tolist()
    for key in buy_dict.keys():
        if key not in tradelist:
            del buy_dict[key]

    for key in buy_dict.keys():
        # print key
        stkPrice = dfprice.loc[key, 'closePrice']
        if stkPrice != 0 and not np.isnan(stkPrice):
            if abs(buy_dict[key]) >= 1:
                pass
            else:
                order_pct_to(key, buy_dict[key])
        else:
            pass