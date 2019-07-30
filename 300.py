# -*- coding: UTF-8 -*

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from CAL.PyCAL import *

cal = Calendar('China.SSE')


def choose_datelist(startdate, enddate, freq=10):
    startdate = cal.advanceDate(startdate, "-1B")
    datelist = cal.bizDatesList(startdate, enddate)
    res = map(lambda x: x.strftime("%Y-%m-%d"), datelist[::freq])
    return res


def choose_stk(stklist, date):
    # 去除交易日停牌的
    tvdf = DataAPI.MktEqudGet(tradeDate=date, secID=stklist, field=u"secID,turnoverValue", pandas="1")  # 去除当日停牌股票
    tvdf = tvdf.dropna(how='any')
    tvdf = tvdf[tvdf['turnoverValue'] != 0]
    stklist = tvdf['secID'].tolist()
    return stklist


# 提取hs300行业权重及指数权重
def get_stkWeight(date, index_id):  # 取得指数成分股票的权重，当天停牌股票权重为0
    newdate1 = cal.advanceDate(date, '-22B', BizDayConvention.Preceding).strftime("%Y-%m-%d")
    df = DataAPI.IdxCloseWeightGet(ticker=index_id,
                                   beginDate=newdate1,
                                   endDate=date,
                                   field=['consID', 'weight', 'effDate']).set_index(['consID']).fillna(0)
    df = df.query("effDate == effDate.max()")
    # 得到stklist的行业，并pivoted,输出求解用stklist
    df.index.name = 'secID'
    wb = pd.DataFrame()
    wb['weight'] = (df['weight']) / 100
    stklist = df.index.tolist()

    if date == cal.advanceDate(df['effDate'].max(), '1B').strftime("%Y-%m-%d"):
        pass
        # print wb
    elif date == df['effDate'].max():
        stk_chg = DataAPI.MktEqudGet(secID=stklist,
                                     beginDate=date,
                                     endDate=date,
                                     field=u"secID,chgPct")  # .set_index(['secID'])#.reindex(index=stklist).fillna(0.0)
        stk_chg = stk_chg.groupby('secID')['chgPct'].apply(lambda x: np.prod(x + 1) - 1)
        w_adj = pd.DataFrame()
        w_adj['weight'] = (1 + stk_chg)
        w_adj = w_adj.reindex(index=stklist)
        wb['weight'] = wb['weight'] / w_adj['weight']
        wb['weight'] = wb['weight'] / sum(wb['weight'])
    else:
        begindate = cal.advanceDate(df['effDate'].max(), '1B', BizDayConvention.Preceding).strftime("%Y-%m-%d")
        endDate = cal.advanceDate(date, '-1B', BizDayConvention.Preceding).strftime("%Y-%m-%d")
        print begindate, endDate
        if date == '2017-09-04':
            print "begindate"
            # stk_chg.to_csv('stk_chg_bef.csv')
            wb.to_csv('wb_bef.csv')
        stk_chg = DataAPI.MktEqudGet(secID=stklist,
                                     beginDate=begindate,
                                     endDate=endDate,
                                     field=u"secID,chgPct")  # .set_index(['secID'])#.reindex(index=stklist).fillna(0.0)
        stk_chg = stk_chg.groupby('secID')['chgPct'].apply(lambda x: np.prod(x + 1) - 1)
        if date == '2017-09-04':
            print "begindate"
            stk_chg.to_csv('stk_chg_aft.csv')
            wb.to_csv('wb_aft.csv')
        w_adj = pd.DataFrame()
        w_adj['weight'] = (1 + stk_chg)
        w_adj = w_adj.reindex(index=stklist)
        wb['weight'] = wb['weight'] * w_adj['weight']
        wb['weight'] = wb['weight'] / sum(wb['weight'])
    temp = DataAPI.MktEqudGet(tradeDate=date, secID=stklist, field=u"secID,turnoverValue").set_index(
        ['secID'])  # 去除当日停牌股票
    stk_halt = temp[temp['turnoverValue'] == 0].index
    wb.loc[stk_halt] = 0
    stk_chg = DataAPI.MktEqudGet(secID=stklist,
                                 beginDate=date,
                                 endDate=date,
                                 field=u"secID,chgPct").set_index(['secID']).reindex(index=stklist).fillna(0.0)
    index_ret = sum(wb['weight'] * stk_chg['chgPct'])
    print 'index_ret', index_ret * 100
    return stklist, wb, index_ret


def RiskModel(date1, date2, stklist, riskfactor):  # 因子暴露是昨天，收益是今天
    field1 = ['secID'] + riskfactor
    field2 = ['Factor'] + riskfactor
    date1 = Date.strptime(date1, '%Y-%m-%d').strftime("%Y%m%d")
    date2 = Date.strptime(date2, '%Y-%m-%d').strftime("%Y%m%d")

    # 因子收益数据
    FactorRet = DataAPI.RMFactorRetDayGet(beginDate=date2,
                                          endDate=date2,
                                          field=riskfactor).mean().fillna(0.0) * 100

    # 因子暴露用最新数据
    RMExposure = DataAPI.RMExposureDayGet(secID=stklist,
                                          beginDate=date1,
                                          endDate=date1,
                                          field=field1).set_index(['secID']).reindex(index=stklist).fillna(0.0)

    return RMExposure[riskfactor], FactorRet[riskfactor]


class Performance:
    def __init__(self, bt, perf, benchindex, riskfactor=None):
        self.bt = bt
        self.perf = perf
        self.benchindex = benchindex
        self.returns = perf['returns']
        self.benchmark_returns = perf['benchmark_returns']
        self.datelist = bt['tradeDate'].astype(str)
        if riskfactor is None:
            riskfactor = ["EARNYILD", "BTOP", "LIQUIDTY", "MOMENTUM", "GROWTH", "BETA", "SIZE", "RESVOL", "LEVERAGE",
                          "SIZENL",
                          'Bank', 'RealEstate', 'Health', 'Transportation', 'Mining', 'NonFerMetal', 'HouseApp',
                          'LeiService',
                          'MachiEquip', 'BuildDeco', 'CommeTrade', 'CONMAT', 'Auto', 'Textile', 'FoodBever',
                          'Electronics',
                          'Computer', 'LightIndus', 'Utilities', 'Telecom', 'AgriForest', 'CHEM', 'Media', 'IronSteel',
                          'NonBankFinan', 'ELECEQP', 'AERODEF', 'Conglomerates', 'COUNTRY']
        self.riskfactor = riskfactor

    def risk_analysis(self, date2):
        date1 = cal.advanceDate(date2, "-1B").strftime("%Y-%m-%d")
        if date2 == self.datelist.iloc[0]:  # 交易第一天用今天(date2)的股票池，因为前一天没有持仓
            date_data = self.bt.query("tradeDate == '%s'" % date2)
            bt_data = pd.DataFrame(date_data['security_position'].values[0]).T
            stockset = bt_data.index.tolist()
            weight = (bt_data['value'] - bt_data['P/L'])
            weight /= weight.sum()
        else:  # 其他日期用今天(date2)的股票池，权重为昨天
            date_data = self.bt.query("tradeDate == '%s'" % date1)
            bt_data = pd.DataFrame(date_data['security_position'].values[0]).T
            stockset = choose_stk(bt_data.index.tolist(), date2)  # 今天(date2)股票涨跌的结果，停牌的股票有可能复牌
            portfolio_value = date_data['portfolio_value'].values
            weight = bt_data.loc[stockset]
            weight = weight['value'] / portfolio_value
        # 行业配置数据
        industryDict, weightBaseDf = industry_dict_weigth()  # 行业字典建立
        Hdf = industry_get(stockset, date2)  # 行业分类取最新数据
        H = transfer_dummy(industryDict, Hdf)  # 获得股票行业哑变量
        H_date = pd.DataFrame(np.dot(weight, H), index=weightBaseDf.index, columns=[date2])

        stklist, wb = get_stkWeight(date2, self.benchindex)  # 要用date2，返回到昨天(date1)的权重，不能把停牌股权重设置为0，date2还要用到
        stklist = choose_stk(stklist, date2)  # 今天(date2)股票涨跌的结果，停牌的股票今天有可能复牌
        wb = wb.loc[stklist]
        # 计算股票池的风格因子暴露
        XRM, fRM = RiskModel(date1, date2, stockset, self.riskfactor)
        XRM = XRM.T.dot(weight)
        # 计算指数成分股的风格因子暴露
        XRMbench, fRMbench = RiskModel(date1, date2, stklist, self.riskfactor)
        XRMbench = XRMbench.T.dot(wb['weight'])
        # 风险暴露
        Risk_date = (XRM - XRMbench).rename(date1)
        # 收益分解
        date_speci = date2.replace('-', '')
        R_speci = DataAPI.RMSpecificRetDayGet(secID=stockset,
                                              beginDate=date_speci,
                                              endDate=date_speci,
                                              field=['secID', 'spret']).set_index('secID').reindex(index=stockset)
        R_speci_bench = DataAPI.RMSpecificRetDayGet(secID=stklist,
                                                    beginDate=date_speci,
                                                    endDate=date_speci,
                                                    field=['secID', 'spret']).set_index('secID').reindex(index=stklist)
        R_speci = (weight * R_speci['spret']).sum()
        R_speci_bench = np.sum(wb['weight'] * R_speci_bench['spret'])
        R_Risk = (XRM - XRMbench) * fRM
        R_Risk['R_speci'] = R_speci - R_speci_bench
        R_Risk['DELETA'] = self.returns.loc[date2].values[0] * 100 - self.benchmark_returns.loc[date2].values[
                                                                         0] * 100 - R_Risk.sum()
        return R_Risk.rename(date2), Risk_date, H_date

    def R_risk(self):
        # 调整第一天的基准收益
        benchmark_returns = self.benchmark_returns
        benchmark_returns_first = DataAPI.MktIdxdGet(tradeDate=self.datelist[0],
                                                     ticker=self.benchindex,
                                                     exchangeCD=u"XSHE,XSHG",
                                                     field=u"ticker,preCloseIndex,openIndex",
                                                     pandas="1")
        benchmark_returns.loc[self.datelist[0]] = (benchmark_returns.loc[self.datelist[0]] + 1) * \
                                                  (benchmark_returns_first.eval("preCloseIndex/openIndex ")).values[
                                                      0] - 1
        # 组合收益 returns

        R_arr = []
        Risk_arr = []
        H_arr = []
        for date2 in self.datelist:
            # 股票池与日期
            print date2
            R_tmp, Risk_tmp, H_tem = self.risk_analysis(date2)
            R_arr.append(R_tmp)
            Risk_arr.append(Risk_tmp)
            H_arr.append(H_tem)
        # 组合超额收益

        Returns = self.returns.rename('returns').to_frame().T * 100
        Returns.columns = Returns.columns.astype(str)
        R = pd.concat(R_arr, axis=1).append(Returns).dropna(how='any', axis=1)
        Risk = pd.concat(Risk_arr, axis=1)
        H = pd.concat(H_arr, axis=1)
        return R, Risk, H


def Risk_Analysis(returns, benchmark_returns, benchindex, bt, riskfactor, date2):
    '''
    datelist：股票池持仓日期列表，%Y-%M-%D，文本格式
    trade_datelist :股票池调仓日期列表，%Y-%M-%D，文本格式
    '''
    datelist = bt['tradeDate'].astype(str)
    date1 = cal.advanceDate(date2, "-1B").strftime("%Y-%m-%d")
    if date2 == datelist.iloc[0]:  # 交易第一天用今天(date2)的股票池，因为前一天没有持仓
        bt_date = pd.DataFrame(bt['security_position'][bt['tradeDate'] == date2].values[0]).T
        stockset = bt_date.index.tolist()
        weight = (bt_date['value'] - bt_date['P/L'])
        weight /= weight.sum()
    else:  # 其他日期用今天(date2)的股票池，权重为昨天
        bt_date = pd.DataFrame(bt['security_position'][bt['tradeDate'] == date1].values[0]).T
        stockset = choose_stk(bt_date.index.tolist(), date2)  # 今天(date2)股票涨跌的结果，停牌的股票有可能复牌
        portfolio_value = bt['portfolio_value'][bt['tradeDate'] == date1].values
        weight = bt_date.loc[stockset]
        weight = weight['value'] / portfolio_value
    # 行业配置数据
    industryDict, weightBaseDf = industry_dict_weigth()  # 行业字典建立
    Hdf = industry_get(stockset, date2)  # 行业分类取最新数据
    H = transfer_dummy(industryDict, Hdf)  # 获得股票行业哑变量
    H_date = pd.DataFrame(np.dot(weight, H), index=weightBaseDf.index, columns=[date2])

    stklist, wb = get_stkWeight(date2, benchindex)  # 要用date2，返回到昨天(date1)的权重，不能把停牌股权重设置为0，date2还要用到
    stklist = choose_stk(stklist, date2)  # 今天(date2)股票涨跌的结果，停牌的股票今天有可能复牌
    wb = wb.loc[stklist]
    # 计算股票池的风格因子暴露
    XRM, fRM = RiskModel(date1, date2, stockset, riskfactor)
    XRM = XRM.T.dot(weight)
    # 计算指数成分股的风格因子暴露
    XRMbench, fRMbench = RiskModel(date1, date2, stklist, riskfactor)
    XRMbench = XRMbench.T.dot(wb['weight'])
    # 风险暴露
    Risk_date = (XRM - XRMbench).rename(date1)
    # 收益分解
    date_speci = date2.replace('-', '')
    R_speci = DataAPI.RMSpecificRetDayGet(secID=stockset,
                                          beginDate=date_speci,
                                          endDate=date_speci,
                                          field=['secID', 'spret']).set_index('secID').reindex(index=stockset)
    R_speci_bench = DataAPI.RMSpecificRetDayGet(secID=stklist,
                                                beginDate=date_speci,
                                                endDate=date_speci,
                                                field=['secID', 'spret']).set_index('secID').reindex(index=stklist)
    R_speci = (weight * R_speci['spret']).sum()
    R_speci_bench = np.sum(wb['weight'] * R_speci_bench['spret'])
    R_Risk = (XRM - XRMbench) * fRM
    R_Risk['R_speci'] = R_speci - R_speci_bench
    R_Risk['DELETA'] = returns.loc[date2].values[0] * 100 - benchmark_returns.loc[date2].values[0] * 100 - R_Risk.sum()
    return R_Risk.rename(date2), Risk_date, H_date


def R_Risk(bt, perf, benchindex):
    '''
    benchindex = "000300"
    '''
    riskfactor = ["EARNYILD", "BTOP", "LIQUIDTY", "MOMENTUM", "GROWTH", "BETA", "SIZE", "RESVOL", "LEVERAGE", "SIZENL",
                  'Bank', 'RealEstate', 'Health', 'Transportation', 'Mining', 'NonFerMetal', 'HouseApp', 'LeiService',
                  'MachiEquip', 'BuildDeco', 'CommeTrade', 'CONMAT', 'Auto', 'Textile', 'FoodBever', 'Electronics',
                  'Computer', 'LightIndus', 'Utilities', 'Telecom', 'AgriForest', 'CHEM', 'Media', 'IronSteel',
                  'NonBankFinan', 'ELECEQP', 'AERODEF', 'Conglomerates', 'COUNTRY']
    datelist = bt['tradeDate'].astype(str)
    # 调整第一天的基准收益
    benchmark_returns = pd.DataFrame()
    benchmark_returns['benchmark_returns'] = perf['benchmark_returns']
    benchmark_returns_first = DataAPI.MktIdxdGet(tradeDate=datelist[0],
                                                 ticker=benchindex,
                                                 exchangeCD=u"XSHE,XSHG",
                                                 field=u"ticker,preCloseIndex,openIndex",
                                                 pandas="1")
    benchmark_returns.loc[datelist[0]] = (benchmark_returns.loc[datelist[0]] + 1) * \
                                         (benchmark_returns_first.eval("preCloseIndex/openIndex ")).values[0] - 1
    # 组合收益 returns
    returns = pd.DataFrame()
    returns['returns'] = perf['returns']

    R_arr = []
    Risk_arr = []
    H_arr = []
    for date2 in datelist:
        # 股票池与日期
        print date2
        R_tmp, Risk_tmp, H_tem = Risk_Analysis(returns, benchmark_returns, benchindex, bt, riskfactor, date2)
        R_arr.append(R_tmp)
        Risk_arr.append(Risk_tmp)
        H_arr.append(H_tem)
    # 组合超额收益
    index = perf['returns'].index.strftime("%Y-%m-%d")
    Returns = pd.DataFrame((returns['returns'] * 100 - benchmark_returns['benchmark_returns'] * 100).values,  #
                           columns=['returns'],
                           index=index).T
    R = pd.concat(R_arr, axis=1).append(Returns).dropna(how='any', axis=1)
    Risk = pd.concat(Risk_arr, axis=1)
    H = pd.concat(H_arr, axis=1)
    return R, Risk, H


def H_report(H):
    H_mean = H.mean(axis=1) * 100
    # H_mean = pd.DataFrame(H_mean,columns=['H_mean'],index=H_mean.index)
    H_mean.index = [x.decode('utf8') for x in H_mean.index]
    H_mean = H_mean.sort_values(ascending=False)
    H_mean.plot(kind='bar', alpha=0.7, color=['b'], figsize=(15, 7))
    plt.xticks(np.arange(len(H_mean)), H_mean.index, fontproperties=font, fontsize=16, rotation=60)
    # plt.title(u"行业配置",fontproperties=font, fontsize=16)
    plt.ylabel(u"行业配置比例(%)", fontproperties=font, fontsize=16)
    plt.show()
    # print 'H_mean'
    # H_mean.to_csv('H_mean.csv')


def Risk_report(Risk):
    Risk_mean = Risk.mean(axis=1)
    Risk_mean.plot(kind='bar', alpha=0.7, color=['b'], figsize=(15, 7))
    plt.xlabel(u"风险因子", fontproperties=font, fontsize=16)
    plt.ylabel(u"每日因子暴露均值", fontproperties=font, fontsize=16)
    plt.show()


def R_report(R):
    R_sum = (R / 100).sum(axis=1)
    alphafactor = ["EARNYILD", "BTOP"]
    riskfactor = ["LIQUIDTY", "MOMENTUM", "GROWTH", "BETA", "SIZE", "RESVOL", "LEVERAGE", "SIZENL"]
    Industryfactor = ['Bank', 'RealEstate', 'Health', 'Transportation', 'Mining', 'NonFerMetal', 'HouseApp',
                      'LeiService', 'MachiEquip', 'BuildDeco', 'CommeTrade', 'CONMAT', 'Auto', 'Textile', 'FoodBever',
                      'Electronics', 'Computer', 'LightIndus', 'Utilities', 'Telecom', 'AgriForest', 'CHEM', 'Media',
                      'IronSteel', 'NonBankFinan', 'ELECEQP', 'AERODEF', 'Conglomerates', 'COUNTRY']
    alpha = R_sum[alphafactor + riskfactor]
    alpha = alpha * 100
    color = ['b']
    alpha.plot(kind='bar', alpha=0.7, color=color, figsize=(10, 6))
    plt.xticks(np.arange(len(alpha)), alpha.index, fontproperties=font, fontsize=16, rotation=50)
    # plt.title(u"超额收益分解",fontproperties=font, fontsize=16)
    plt.ylabel(u"风格超额收益(%)", fontproperties=font, fontsize=16)
    plt.show()
    print alpha


def R_summary(R):
    R_sum = (R / 100).sum(axis=1)
    alphafactor = ["EARNYILD", "BTOP"]
    riskfactor = ["LIQUIDTY", "MOMENTUM", "GROWTH", "BETA", "SIZE", "RESVOL", "LEVERAGE", "SIZENL"]
    Industryfactor = ['Bank', 'RealEstate', 'Health', 'Transportation', 'Mining', 'NonFerMetal', 'HouseApp',
                      'LeiService', 'MachiEquip', 'BuildDeco', 'CommeTrade', 'CONMAT', 'Auto', 'Textile', 'FoodBever',
                      'Electronics', 'Computer', 'LightIndus', 'Utilities', 'Telecom', 'AgriForest', 'CHEM', 'Media',
                      'IronSteel', 'NonBankFinan', 'ELECEQP', 'AERODEF', 'Conglomerates', 'COUNTRY']
    alpha = R_sum[alphafactor + ['R_speci'] + ['DELETA']].sum()
    riskalpha = R_sum[riskfactor].sum()
    Industry = R_sum[Industryfactor].sum()
    Returns = R_sum['returns']
    R_sum = pd.Series([Returns, alpha, riskalpha, Industry], index=['总超额收益', '选股', '风格', '行业'])
    R_sum = R_sum * 100
    R_sum.index = [x.decode('utf8') for x in R_sum.index]
    color = ['r'] + ['b'] * (len(R_sum) - 1)
    R_sum.plot(kind='bar', alpha=0.7, color=color, figsize=(10, 6))
    plt.xticks(np.arange(len(R_sum)), R_sum.index, fontproperties=font, fontsize=16, rotation=0)
    plt.title(u"超额收益分解", fontproperties=font, fontsize=16)
    plt.ylabel(u"累计超额收益(%)", fontproperties=font, fontsize=16)
    plt.show()
    print R_sum


def Risk_report(Risk):
    Risk_sum = Risk.mean(axis=1)
    fig = plt.figure(figsize=(15, 7))
    fig.set_tight_layout(True)
    fig.add_subplot(111)
    color = ['b']
    Risk_sum.plot(kind='bar', alpha=0.7, color=color)
    plt.xlabel(u"风险因子", fontproperties=font, fontsize=16)
    plt.ylabel(u"每日因子暴露均值", fontproperties=font, fontsize=16)
    plt.show()


def R_report(R):
    R_sum = (R / 100 + 1).prod(axis=1) - 1
    fig = plt.figure(figsize=(15, 7))
    fig.set_tight_layout(True)
    fig.add_subplot(111)
    color = ['b'] * (len(R_sum) - 2) + ['r', 'r']
    R_sum.plot(kind='bar', alpha=0.7, color=color)
    plt.xlabel(u"风险因子", fontproperties=font, fontsize=16)
    plt.ylabel(u"因子累计超额收益", fontproperties=font, fontsize=16)
    plt.show()
