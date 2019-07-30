# coding: utf-8

import numpy as np
import pandas as pd
import statsmodels.api as sm
from CAL.PyCAL import *
from quartz.universe import set_universe
from quartz_extensions.MFHandler.SignalProcess import standardize, neutralize, winsorize
from requests.exceptions import ConnectionError, ReadTimeout


def winsorize(ax):
    return ax[(float(ax.mean()-3*ax.std()) <= ax) & (ax <= float(ax.mean()+3*ax.std()))]

cal = Calendar('China.SSE')

TODAY = pd.to_datetime('today').strftime("%Y-%m-%d")

BETA_FACTORS = ["EARNYILD", "BTOP", "LIQUIDTY", "MOMENTUM", "GROWTH", "BETA", "SIZE", "RESVOL", "LEVERAGE", "SIZENL",
                'Bank', 'RealEstate', 'Health', 'Transportation', 'Mining', 'NonFerMetal', 'HouseApp', 'LeiService',
                'MachiEquip', 'BuildDeco', 'CommeTrade', 'CONMAT', 'Auto', 'Textile', 'FoodBever', 'Electronics',
                'Computer', 'LightIndus', 'Utilities', 'Telecom', 'AgriForest', 'CHEM', 'Media', 'IronSteel',
                'NonBankFinan', 'ELECEQP', 'AERODEF', 'Conglomerates', 'COUNTRY']

MY_RISK_FACTORS = [u'Earning Yild', u'Growth', u'Leverage', u'Liquidity', u'ResVol',
                   u'BTOP', u'SIZE', u'Momentum', u'BETA', u'NLSIZE', u'农林牧渔', u'采掘',
                   u'化工', u'钢铁', u'有色金属', u'建筑材料', u'建筑装饰', u'电气设备', u'机械设备', u'国防军工',
                   u'汽车', u'电子', u'家用电器', u'食品饮料', u'纺织服装', u'轻工制造', u'医药生物', u'公用事业',
                   u'交通运输', u'房地产', u'银行', u'商业贸易', u'休闲服务', u'计算机', u'传媒', u'通信', u'综合',
                   u'非银金融', u'COUNTRY']  # u'证券', u'保险', u'多元金融'

# 行业列表
INDUSTRY_LIST = DataAPI.IndustryGet(industryVersion=u"SW",
                                    industryLevel=u"1,2",
                                    isNew=u"1",
                                    ).query("industryID != '01030322' and industryLevel == 1 or industryID in %s" %
                                            ["0103032201", "0103032202", "0103032203"])['industryName'].values

# 优化时行业的边界
INDUSTRY_BOUND_DICT = {'银行': 0.03,
                       '保险': 0.04,
                       '证券': 0.02}

CATEGORY_DICT = {u'周期上游': ['采掘', '有色金属'],
                 u'周期中游': ['钢铁', '化工', '公用事业', '交通运输'],
                 u'周期下游': ['建筑材料', '建筑装饰', '汽车', '机械设备'],
                 u'大金融': ['银行', '非银金融', '房地产'],
                 u'消费': ['轻工制造', '商业贸易', '休闲服务', '家用电器', '纺织服装', '医药生物', '食品饮料', '农林牧渔'],
                 u'成长': ['计算机', '传媒', '通信', '电气设备', '电子', '国防军工']}

CATEGORY_DICT_R = {}
for k, v in CATEGORY_DICT.iteritems():
    for i in v:
        CATEGORY_DICT_R[i] = k


def ConnectionErrorDeco(runtimes=5, display=False):
    def deco(process):
        def wrapper(*args, **kwargs):
            for i in range(runtimes):
                try:
                    res = process(*args, **kwargs)
                except ConnectionError or ReadTimeout:
                    print "try %s %d times" % (process.__name__, i + 1)
                    continue
                if display:
                    print '%s process successfully' % process.__name__
                return res
            raise ConnectionError("次数用完")
        return wrapper
    return deco


def get_industry(stklist, tradeDate):
    Hdf = DataAPI.EquIndustryGet(industryVersionCD=u"010303",
                                 secID=stklist,
                                 intoDate=tradeDate,
                                 field=['secID', 'industryName1', 'industryName2'],
                                 pandas="1").set_index('secID')
    return Hdf.apply(lambda x: x[1] if x[0] == '非银金融' else x[0], axis=1)


def symbols_process(symbols=None, index_code=None, tradeDate=None):
    if None not in (symbols, index_code):
        return list(
            set(symbols_process(symbols=symbols) + symbols_process(index_code=index_code, tradeDate=tradeDate)))

    if isinstance(symbols, (list, tuple, np.ndarray)):
        return list(symbols)

    if index_code is not None:
        return get_index_component(index_code, tradeDate)


def date_process(dates=None, start=None, end=None, period=None, tradeDate=None, forward=0):
    if dates is not None:
        assert isinstance(dates, (list, tuple, np.ndarray))
        if forward:
            dates = [cal.advanceDate(d, '%dB' % -forward).strftime("%Y-%m-%d") for d in dates]
        return dates

    if tradeDate is not None:
        assert isinstance(tradeDate, str)
        return date_process([tradeDate], forward=forward)

    if None not in (start, end, period):
        if forward:
            start, end = [cal.advanceDate(d, '%dB' % -forward).strftime("%Y-%m-%d") for d in [start, end]]
        datelist = [d.strftime("%Y-%m-%d") for d in cal.bizDatesList(start, end)]
        return datelist[::period]

    raise ValueError('Parameter Error')


def get_index_data(index_code, tradeDate, exclude_halt=True, debug=False):
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
    if exclude_halt:
        stk_halt = last_data.query("isOpen==0").index
        wb.loc[stk_halt] = 0
    if debug:
        print 'index_ret', wb.dot(last_data['chg'] - 1)

    return wb


def get_index_component(index_code, tradeDate):
    if isinstance(index_code, (tuple, list, np.ndarray)):
        res = []
        for code in index_code:
            res.extend(get_index_component(code, tradeDate))
        return res

    try:
        res = set_universe(index_code, tradeDate)
        if not len(res):
            raise ValueError
        return res

    except ValueError:
        today = pd.to_datetime('today').strftime("%Y-%m-%d")
        if not isinstance(index_code, unicode):
            index_code = index_code.decode('utf-8')
        data = DataAPI.EquIndustryGet(industry=u"",
                                      secID=set_universe('000852.ZICN', tradeDate) + set_universe('000906.ZICN',
                                                                                                  tradeDate),
                                      industryVersionCD=u"010303",
                                      field=u"secID,secShortName,industryName1,intoDate,outDate",
                                      pandas="1").fillna(today)
        data = data.query("intoDate <= '%s' < outDate" % tradeDate)[['secID', 'industryName1']]
        return data.query("industryName1 in %s" % CATEGORY_DICT[index_code])['secID'].drop_duplicates().tolist()


def get_index_data2(index_code, tradeDate, exclude_halt=True):
    newdate1 = cal.advanceDate(tradeDate, '-50B', BizDayConvention.Preceding).strftime("%Y-%m-%d")
    df = DataAPI.IdxCloseWeightGet(secID=u"",
                                   ticker=index_code,
                                   beginDate=newdate1,
                                   endDate=tradeDate,
                                   field=['consID', 'weight', 'effDate'], pandas="1").set_index(['consID']).fillna(0)

    df = df.query("effDate == effDate.max()")
    # 得到stklist的行业，并pivoted,输出求解用stklist
    df.index.name = 'secID'

    # 找到当日停牌的股票，将权重设为0
    temp = DataAPI.MktEqudAdjGet(tradeDate=tradeDate,
                                 secID=df.index,
                                 field=u"secID,isOpen,closePrice",
                                 pandas="1").set_index(['secID']).reindex(index=df.index)  # 去除当日停牌股票
    weights = df['weight'] / 100
    if tradeDate != df.iloc[0]['effDate']:  # 调整最新的指数成分股权重
        temp2 = DataAPI.MktEqudAdjGet(tradeDate=df.iloc[0]['effDate'],
                                      secID=df.index,
                                      field=u"secID,closePrice",
                                      pandas="1").set_index(['secID']).reindex(index=df.index)
        weights = weights * temp['closePrice'] / temp2['closePrice']
        weights = weights.fillna(0.0)
        weights /= weights.sum()
        weights = weights.round(5)
    if exclude_halt:
        weights[temp['isOpen'] == 0] = 0
    return weights


def get_index_data3(index_code, tradeDate, exclude_halt=True):
    if index_code in CATEGORY_DICT or isinstance(index_code, (list, tuple, np.ndarray)):
        yesterday = cal.advanceDate(tradeDate, '-1B').strftime("%Y-%m-%d")
        symbols = get_index_component(index_code, tradeDate)
        df = DataAPI.MktEqudAdjGet(secID=symbols,
                                   tradeDate=yesterday, field=u"secID,isOpen,negMarketValue", pandas="1")
        df['weight'] = df['negMarketValue'] / df['negMarketValue'].sum()
        if exclude_halt:
            df.loc[df['isOpen'] == 0, 'weight'] = 0
        return df.set_index('secID')['weight']
    return get_index_data(index_code, tradeDate, exclude_halt)


def regular_expo(expo, tradeDate):
    if len(tradeDate) > 8:
        tradeDate = tradeDate.replace('-', '')

    return neutralize(standardize(winsorize(expo.replace([np.inf, -np.inf], np.nan).astype(float)
                                            .fillna(0.0)))
                      .fillna(0.0),
                      target_date=tradeDate).fillna(0.0)


def winsorize_expo(expo):
    return expo.apply(winsorize)


def standardize_expo(expo):
    return expo.apply(standardize)


def my_neutralize(expo, tradeDate, risk_factor):
    symbols = expo.index
    risk_expo = (RMDataBase.RMExpoGet(symbols, risk_factor, tradeDate=tradeDate)
                 .set_index('Symbol')
                 .reindex(index=symbols, columns=risk_factor)
                 .fillna(0))
    return expo.apply(lambda x: sm.OLS(x, risk_expo).fit().resid)


def fill_industry_mean(expo, tradeDate):
    expo = expo.assign(industry=get_industry(expo.index, tradeDate))
    tmp = expo.merge(right=expo.groupby('industry').mean().fillna(0), left_on='industry',
                     right_index=True, suffixes=('_o', ''))
    return expo.fillna(tmp.reindex_like(expo)).drop('industry', axis=1)


def fillna(expo, tradeDate=None, industry_mean=False):
    expo = expo.replace([-np.inf, np.inf], np.nan)
    # if expo.isnull().all().any():
    #     raise ValueError

    if expo.notnull().all().all():
        return expo

    if not industry_mean:
        return expo.fillna(0)

    return fill_industry_mean(expo, tradeDate)


def filter_halt(stock, tradeDate):
    tvdf = DataAPI.MktEqudGet(tradeDate=tradeDate, secID=stock, field=u"secID,isOpen", pandas="1")  # 去除当日停牌股票
    tvdf = tvdf.dropna(how='any').query('isOpen==1')
    stock = tvdf['secID'].tolist()
    return stock


def regular_stock(stock, tradeDate, freq=10):
    # 去除ST股
    STdf = DataAPI.SecSTGet(secID=stock, beginDate=tradeDate, endDate=tradeDate, field=['secID'])
    STlist = STdf['secID'].tolist()
    stock = [s for s in stock if s not in STlist]

    # 去除交易日停牌的
    stock = filter_halt(stock, tradeDate)

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


# def get_cumret(start, end, left_close=True, right_close=True, price_type='close', stock=None, factor=None):
#     start = start.replace('-', '')
#     end = end.replace('-', '')
#
#     if not left_close:
#         start = cal.advanceDate(start, '1B').strftime("%Y%m%d")
#
#     if not right_close:
#         end = cal.advanceDate(end, '-1B').strftime("%Y%m%d")
#
#     if price_type == 'close':
#         assert stock is not None
#         data = DataAPI.MktEqudAdjGet(tradeDate=u"",
#                                      secID=stock,
#                                      beginDate=start,
#                                      endDate=end,
#                                      field=u"secID,tradeDate,closePrice,preClosePrice",
#                                      pandas="1")
#         data.eval("ret=closePrice/preClosePrice-1")
#         df = data.pivot('tradeDate', 'secID', 'ret') + 1
#
#     elif price_type == 'spret':
#         assert stock is not None
#         data = DataAPI.RMSpecificRetDayGet(secID=stock,
#                                            beginDate=start,
#                                            endDate=end,
#                                            field=['secID', 'tradeDate', 'spret'])
#         df = data.pivot(index='tradeDate', columns='secID', values='spret') / 100 + 1
#         return 100 * 250 * (df.prod().rename('spret') - 1) / df.shape[0]
#
#     elif price_type == 'factor':
#         assert factor is not None
#         if isinstance(factor, str):
#             factor_list = factor.split(',')
#         elif isinstance(factor, (list, np.ndarray, tuple)):
#             factor_list = factor
#
#         if 'tradeDate' not in factor_list:
#             factor_list = np.append(factor_list, 'tradeDate')
#
#         df = DataAPI.RMFactorRetDayGet(tradeDate=u"",
#                                        beginDate=start,
#                                        endDate=end,
#                                        field=factor_list,
#                                        pandas="1").set_index('tradeDate') + 1
#     return 100 * (df.prod() - 1)


def schmidt_orth(arr):  # 施密特因子正交
    res = arr.copy()
    for i in range(arr.shape[1] - 1):
        temp = np.zeros(arr.shape[0])
        for j in range(i + 1):
            temp[:] = temp[:] + np.dot(res[:, i + 1], res[:, j]) / np.dot(res[:, j], res[:, j]) * res[:, j]
        res[:, i + 1] = res[:, i + 1] - temp[:]
        u = np.mean(res[:, i + 1])
        sigma = np.std(res[:, i + 1])
        if sigma == 0:
            sigma = 1
        res[:, i + 1] = (res[:, i + 1] - u) / sigma  # 正交后标准化
    return res


def _cross_diff(arr1, arr2):
    set1 = set(arr1)
    set2 = set(arr2)
    return list(set1 & set2), list(set1 - set2)
