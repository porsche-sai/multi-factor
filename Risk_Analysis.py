# coding=utf-8

# coding=utf-8

from lib.self_factors import *
from quartz.api import *
import pandas as pd
import numpy as np
import statsmodels.api as sm
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib as mpl
from CAL.PyCAL import *
# from lib.Barra_test import *
from lib.tools import *

BENCHMARK_DICT = {'000905': u"中证500指数增强",
                  '000300': u"沪深300指数增强",
                  '大类行业': u"大类行业"}
riskfactor = ["EARNYILD", "GROWTH", "LIQUIDTY", "BTOP", "MOMENTUM", "BETA", "SIZE", "RESVOL", "LEVERAGE", "SIZENL",
              'Bank', 'RealEstate', 'Health', 'Transportation', 'Mining', 'NonFerMetal', 'HouseApp', 'LeiService',
              'MachiEquip', 'BuildDeco', 'CommeTrade', 'CONMAT', 'Auto', 'Textile', 'FoodBever', 'Electronics',
              'Computer', 'LightIndus', 'Utilities', 'Telecom', 'AgriForest', 'CHEM', 'Media', 'IronSteel',
              'NonBankFinan', 'ELECEQP', 'AERODEF', 'Conglomerates', 'COUNTRY']


class Risk_Performance:
    def __init__(self, bt, perf, benchmark, alpha_dict, target_factors, bt_model, expo):
        self.bt = bt
        self.perf = perf
        self.benchmark = benchmark
        self.bt_model = bt_model
        self.expo = expo
        self.datelist = bt['tradeDate'].astype(str)
        self.target_factors = target_factors
        self.factor = target_factors.keys().tolist()
        self.riskfactor = riskfactor
        self.alpha_dict = alpha_dict
        ##################################################### 调整组合第一天收益
        returns = perf['returns'].copy()
        firstdate_data = self.bt.query("tradeDate == '%s'" % self.datelist.iloc[0])
        bt_firstdate = pd.DataFrame(firstdate_data['security_position'].values[0]).T
        stockset = bt_firstdate.index.tolist()
        weight_first = (bt_firstdate['value'] - bt_firstdate['P/L'])
        weight_first /= weight_first.sum()  # (firstdate_data['portfolio_value'][0]-bt_firstdate['P/L'].sum(0))#第一天收盘后的权重，需要调整到开盘前的权重
        # print weight_first.sum()
        ret = DataAPI.MktEqudAdjGet(tradeDate=self.datelist.iloc[0],
                                    secID=stockset,
                                    field=u"secID,preClosePrice,openPrice,closePrice",
                                    pandas="1").set_index('secID')
        ret.eval('returns_opentoclose=openPrice/preClosePrice')
        ret['returns_opentoclose'] = ret['returns_opentoclose'].replace(0, 1)
        ret = ret.replace({0: 1})
        weight_first = weight_first * (returns[0] + 1) / ret['returns_opentoclose']
        weight_first /= weight_first.sum()
        # print weight_first.sum()
        ret.eval('returns_closetoclose=closePrice/preClosePrice')
        ret['returns_closetoclose'] = ret['returns_closetoclose'].replace(0, 1)
        returns_first = ret['returns_closetoclose'].mul(weight_first).sum() - 1
        returns[0] = returns_first
        print perf['returns'][0] * 100, returns[0] * 100, perf['returns'][0] * 100 - returns_first * 100
        self.returns = returns
        ##################################################### 基准收益率
        data = bt[[u'tradeDate', u'portfolio_value', u'benchmark_return']]
        if benchmark not in BENCHMARK_DICT:
            arr = []
            for day in bt['tradeDate']:
                symbols_weights = get_index_data3(benchmark, day, False)
                ret = DataAPI.MktEqudAdjGet(tradeDate=day,
                                            secID=symbols_weights.index,
                                            field=u"secID,preClosePrice,closePrice",
                                            pandas="1").set_index('secID')
                ret.eval('returns=closePrice/preClosePrice-1')
                arr.append(ret['returns'].mul(symbols_weights).sum())
            data['benchmark_return'] = arr
            data = data.set_index('tradeDate')
            self.benchmark_return = data['benchmark_return']
        else:
            self.benchmark_return = perf['benchmark_returns']
        ##################################################### 计算
        Returns = (self.returns - self.benchmark_return).rename('returns').to_frame().T * 100
        R_arr = []
        Risk_arr = []
        H_arr = []
        Hindex_arr = []
        f_arr = []
        for date2 in self.datelist:
            print date2
            R_tmp, Risk_tmp, H_tem, Hindex_tem, f_tem = self.risk_analysis(date2)
            R_arr.append(R_tmp)
            Risk_arr.append(Risk_tmp)
            H_arr.append(H_tem)
            Hindex_arr.append(Hindex_tem)
            f_arr.append(f_tem)
        # 组合超额收益
        Returns.columns = Returns.columns.astype(str)
        R = pd.concat(R_arr, axis=1).append(Returns).dropna(how='any', axis=1)
        Risk = pd.concat(Risk_arr, axis=1)
        H = pd.concat(H_arr, axis=1)
        Hindex = pd.concat(Hindex_arr, axis=1)
        f = pd.concat(f_arr, axis=1)
        self.R = R
        self.Risk = Risk
        self.H = H
        self.Hindex = Hindex
        self.f = f

    def risk_analysis(self, date2):
        date1 = cal.advanceDate(date2, "-1B").strftime("%Y-%m-%d")
        if date2 == self.datelist.iloc[0]:  # 交易第一天用今天(date2)的股票池，因为前一天没有持仓
            date_data = self.bt.query("tradeDate == '%s'" % date2)
            bt_data = pd.DataFrame(date_data['security_position'].values[0]).T
            stk_weight = (bt_data['value'] - bt_data['P/L'])
            stk_weight /= stk_weight.sum()
        else:  # 其他日期用今天(date2)的股票池，权重为昨天
            date_data = self.bt.query("tradeDate == '%s'" % date1)
            bt_data = pd.DataFrame(date_data['security_position'].values[0]).T
            portfolio_value = date_data['portfolio_value'].values
            stk_weight = bt_data['value'] / portfolio_value
        ##################################################### 行业配置数据
        symbols = bt_data.index.tolist()
        symbols_industry_dict = get_industry(symbols, date2)
        # 计算股票池行业权重
        symbols_industry = stk_weight.rename(index=symbols_industry_dict)
        industry_weights = (symbols_industry.groupby(level=0)
            .sum()
            .reindex(index=INDUSTRY_LIST)
            .fillna(0))
        # print 'symbols_industry_dict',symbols_industry_dict
        ##################################################### 指数成分股和权重
        today_data = self.bt_model.block_chain[date1]  # 取得当日数据
        # bench_symbols = today_data.bench_symbols
        # bench_symbol_weight = today_data.bench_symbol_weight
        # bench_industry_weights = today_data.bench_industry_weights
        # bench_symbol_weight = get_index_data3(self.benchmark, date2, False).to_frame(name='weight')['weight']# 要用date2，返回到昨天(date1)收盘后的权重
        ##################################################### 计算股票池的ALPHA因子暴露
        X_factor = self.expo[date1].loc[symbols].fillna(0).T.dot(stk_weight)
        Xbench_factor = self.expo[date1].loc[today_data.bench_symbols].fillna(0).T.dot(today_data.bench_symbol_weight)
        Factor_date = (X_factor - Xbench_factor).rename(date1)
        ##################################################### 计算股票池的停牌股票的因子暴露和行业权重
        unhalt_position = filter_halt(symbols, date2)  # 今天(date2)股票涨跌的结果，停牌的股票有可能复牌
        halt_stock = set(set(symbols) - set(unhalt_position))
        halt_weight = stk_weight.loc[halt_stock]
        field = "secID,EARNYILD,BTOP,MOMENTUM,RESVOL,GROWTH,BETA,LEVERAGE,LIQUIDTY,SIZENL,SIZE"
        if len(halt_stock):
            halt_Exposure = DataAPI.RMExposureDayGet(secID=halt_stock,
                                                     tradeDate=Date.strptime(date2, '%Y-%m-%d').strftime('%Y%m%d'),
                                                     field=field, pandas="1").set_index('secID')
            halt_expo_sum = halt_weight.dot(halt_Exposure)
            halt_industry_weight = halt_weight.dot(
                get_industry(halt_stock, date2).apply(lambda x: pd.Series(INDUSTRY_LIST == x,
                                                                          index=INDUSTRY_LIST,
                                                                          dtype=int)))
        else:
            halt_expo_sum = 0
            halt_industry_weight = 0
        bench_industry_weights = today_data.bench_industry_weights  # - halt_industry_weight  不用减去，symbols包含了停牌股票
        ##################################################### 计算相对基准的风险因子暴露
        risk_obj = today_data.risk_model(self.riskfactor)
        XRM_relative = risk_obj.relative_expo(bench_weight=today_data.bench_symbol_weight)
        XRM_relative = XRM_relative.loc[symbols].fillna(0).T.dot(stk_weight)
        fRM = DataAPI.RMFactorRetDayGet(beginDate=date2.replace('-', ''),
                                        endDate=date2.replace('-', ''),
                                        field=self.riskfactor).mean().fillna(0.0) * 100  # 因子收益数据  #
        R_date = (XRM_relative * fRM).rename(date2)  # 风险因子超额收益
        ##################################################### 风险暴露
        Risk_date = XRM_relative[:10].rename(date1)  # - halt_expo_sum # symbols包含了停牌股票，不需要减去
        # print Factor_date,Risk_date
        Risk_date = pd.concat([Factor_date, Risk_date])
        ##################################################### 收益分解
        date_speci = date2.replace('-', '')
        Spret = DataAPI.RMSpecificRetDayGet(secID=set_universe('A', date_speci),
                                            beginDate=date_speci,
                                            endDate=date_speci,
                                            field=['secID', 'spret']).set_index('secID')
        Y = Spret.reindex(index=self.expo[date1].index.tolist()).fillna(0)
        X = self.expo[date1].fillna(0)
        res_ols = sm.OLS(Y['spret'], X).fit()
        f = res_ols.params
        f = f.rename(date2)
        R_Factor = ((X_factor - Xbench_factor) * f).rename(date2)  # ALPHA因子超额收益
        RSpret = np.sum((Spret['spret'].loc[symbols].fillna(0)).T.dot(stk_weight))
        RSpret_bench = np.sum(
            Spret['spret'].loc[today_data.bench_symbols].fillna(0).T.dot(today_data.bench_symbol_weight))
        R_date = pd.concat([R_Factor, R_date])
        R_date['RSpret'] = RSpret - RSpret_bench
        R_date['RSpret_resid'] = RSpret - RSpret_bench - R_Factor.sum()
        R_date['DELTA'] = self.returns.loc[date2] * 100 - self.benchmark_return.loc[date2] * 100 - R_date[
            self.riskfactor + ['RSpret']].sum()
        return R_date, Risk_date, industry_weights, bench_industry_weights, f

    def H_report(self):  # 行业因子平均暴露
        H_mean = self.H.mean(axis=1) * 100
        Hindex_mean = self.Hindex.mean(axis=1) * 100
        H_mean = pd.concat([H_mean, Hindex_mean], axis=1)
        H_mean.columns = ["Portflio", "Index"]
        H_mean = H_mean.sort_values(by=['Portflio'], ascending=False)
        H_mean.index = [x.decode('utf8') for x in H_mean.index]
        H_mean.plot(kind='bar', alpha=0.7, color=['b'] + ['r'], figsize=(15, 7))
        plt.xticks(np.arange(len(H_mean)), H_mean.index, fontproperties=font, fontsize=16, rotation=60)
        plt.title(u"行业配置", fontproperties=font, fontsize=16)
        plt.ylabel(u"行业配置比例(%)", fontproperties=font, fontsize=16)
        for a, b in zip(np.arange(len(H_mean)), H_mean.values[:, 0]):
            plt.text(a - 0.1, b, '%.1f' % b, ha='center', va='bottom', fontproperties=font, fontsize=10)
        plt.grid()
        plt.show()

    def Risk_report(self):  # 因子暴露均值
        Risk_mean = self.Risk.mean(axis=1)
        Risk_mean.plot(kind='bar', alpha=0.7, color=['b'], figsize=(15, 7))
        plt.xlabel(u"风险因子", fontproperties=font, fontsize=16)
        plt.ylabel(u"每日因子暴露均值", fontproperties=font, fontsize=16)
        for a, b in zip(np.arange(len(Risk_mean)), Risk_mean.values):
            plt.text(a, b, '%.1f' % b, ha='center', va='bottom', fontproperties=font, fontsize=10)
        plt.grid()
        plt.show()

    def R_report(self):
        R_sum = self.R.sum(axis=1)
        print  ['RSpret'] + ['DELTA'] + self.factor + self.riskfactor[0:10]
        alpha1 = R_sum[self.factor]
        alpha2 = R_sum[self.riskfactor[0:10]]
        alpha1.index = ['alpha' + "%02d" % i for i in range(1, len(self.factor) + 1)]
        alpha1 = alpha1.sort_values(ascending=False)
        alpha2 = alpha2.sort_values(ascending=False)
        alpha = pd.concat([R_sum[['RSpret', 'DELTA']], alpha1, alpha2])  # ['R_speci','R_speci_resid','DELETA']
        color = ['b']
        alpha.plot(kind='bar', alpha=0.7, color=color, figsize=(15, 7))
        plt.xticks(np.arange(len(alpha)), alpha.index, fontproperties=font, fontsize=16, rotation=50)
        plt.title(u"风格超额收益分解", fontproperties=font, fontsize=16)
        plt.ylabel(u"风格超额收益(%)", fontproperties=font, fontsize=16)
        for a, b in zip(np.arange(len(alpha)), alpha.values):
            plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontproperties=font, fontsize=12)
        plt.grid()
        plt.show()
        print alpha

    def RH_report(self):
        R_sum = self.R.sum(axis=1)
        print self.riskfactor[10:]
        alpha = R_sum[self.riskfactor[10:]].sort_values(ascending=False)
        color = ['b']
        alpha.plot(kind='bar', alpha=0.7, color=color, figsize=(16, 7))
        plt.xticks(np.arange(len(alpha)), alpha.index, fontproperties=font, fontsize=16, rotation=50)
        plt.title(u"行业超额收益分解", fontproperties=font, fontsize=16)
        plt.ylabel(u"行业超额收益(%)", fontproperties=font, fontsize=16)
        for a, b in zip(np.arange(len(alpha)), alpha.values):
            plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontproperties=font, fontsize=12)
        plt.grid()
        plt.show()
        print alpha

    def R_summary(self):
        R_sum = self.R.sum(axis=1)
        print '选股:', self.alpha_dict.keys() + ['RSpret'] + ['DELETA']
        print '风格:', self.riskfactor[len(self.alpha_dict):10]
        print '行业:', self.riskfactor[10:]
        alpha = R_sum[self.alpha_dict.keys() + ['RSpret'] + ['DELETA']].sum()  #
        riskalpha = R_sum[list(set(self.riskfactor[0:10]) - set(self.alpha_dict.keys()))].sum()
        Industry = R_sum[self.riskfactor[10:]].sum()
        Returns = R_sum['returns']
        R_sum = pd.Series([Returns, alpha, riskalpha, Industry], index=['总超额收益', '选股', '风格', '行业'])
        R_sum.index = [x.decode('utf8') for x in R_sum.index]
        color = ['r'] + ['b'] * (len(R_sum) - 1)
        R_sum.plot(kind='bar', alpha=0.7, color=color, figsize=(10, 7))
        plt.xticks(np.arange(len(R_sum)), R_sum.index, fontproperties=font, fontsize=16, rotation=0)
        plt.title(u"超额收益分解", fontproperties=font, fontsize=16)
        plt.ylabel(u"累计超额收益(%)", fontproperties=font, fontsize=16)
        for a, b in zip(np.arange(len(R_sum)), R_sum.values):
            plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontproperties=font, fontsize=16)
        plt.grid()
        plt.show()
        print R_sum


def Buy_list(bt, benchmark, Yesterday, Buylistype, Portf_current_name=None):
    '''
    bt：回测报告，格式为pandas.DataFrame
    Yesterday：昨日股票池，%Y-%M-%D，文本格式
    Portf_current_name:现有股票池 '综合信息查询_组合证券6001_20170919.xls'
    '''
    data = bt.query("tradeDate == '%s'" % Yesterday)
    bt_date = pd.DataFrame(data['security_position'].values[0]).T
    stockset = filter_halt(bt_date.index.tolist(), Yesterday)
    portfolio_value = data['portfolio_value'].values
    weight = bt_date.loc[stockset]
    weight = weight['value'] / portfolio_value
    print weight.sum()
    if Buylistype == '导入证券':
        buy_list = DataAPI.EquGet(equTypeCD=u"A",
                                  secID=stockset,
                                  field=u"secID,exchangeCD,ticker").set_index('secID').reindex(index=stockset)
        buy_list.columns = ['市场内部编号', '证券代码']
        buy_list['目标权重'] = weight
        if Portf_current_name is None:
            pass
        else:
            Portf_current = pd.read_excel(Portf_current_name, converters={u'证券代码': str})
            print  Portf_current[u'证券代码'].tolist()
            tvdf = DataAPI.MktEqudGet(tradeDate=Yesterday,
                                      ticker=Portf_current[u'证券代码'].tolist(),
                                      field=u"secID,exchangeCD,turnoverValue").set_index('secID')
            # 剔除停牌
            print filter_halt(tvdf.index.tolist(), Yesterday)
            tvdf = tvdf.dropna(how='any')[tvdf['turnoverValue'] != 0]
            Portf_stklist = filter_halt(tvdf.index.tolist(), Yesterday)
            print Portf_stklist
            for stk in Portf_stklist:
                exchangeCD = tvdf['exchangeCD'].loc[stk]
                if stk not in buy_list.index.tolist():
                    buy_list.loc[stk, ['市场内部编号', '证券代码', '目标权重']] = [exchangeCD, stk, 0]
        buy_list = buy_list.replace({'XSHE': '2', 'XSHG': '1'})
        buy_list.to_csv(benchmark + '导入证券' + Yesterday + '.csv', encoding='GB18030')
        print len(buy_list), benchmark + '导入证券' + Yesterday + '.csv'
        print buy_list.tail()
    elif Buylistype == '期现套利':
        buy_list = DataAPI.EquGet(equTypeCD=u"A",
                                  secID=stockset,
                                  field=u"secID,ticker,secShortName,exchangeCD").set_index('secID').reindex(
            index=stockset)
        buy_list = buy_list.replace({'XSHE': u'深交所', 'XSHG': u'上交所'})
        buy_list.columns = ['证券代码', '证券名称', '交易市场']
        buy_list['数量'] = ''
        buy_list['市值'] = ''
        buy_list['市值权重'] = ''
        buy_list['设置比例'] = 100.0 * weight
        buy_list['指数权重'] = ''
        buy_list['停牌标志'] = ''
        buy_list['所属行业'] = ''
        buy_list['替代证券代码'] = '510050'
        buy_list['替代证券名称'] = '50ETF'
        buy_list['替代证券交易市场'] = '上交所'
        buy_list.to_csv(benchmark + '比例权重' + Yesterday + '.csv', encoding='GB18030')
        print len(buy_list), benchmark + '比例权重' + Yesterday + '.csv'
    print buy_list.head()
