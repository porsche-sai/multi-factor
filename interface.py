# coding: utf-8

import copy

import numpy as np
import pandas as pd
import statsmodels.api as sm
from lib.tools import *
from sqlalchemy import create_engine, MetaData, Table, and_, select, desc, Column, VARCHAR, Float, exists
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, mapper

map_dict = {u'银行': u'Bank',
            u'农林牧渔': u'AgriForest',
            u'采掘': u'Mining',
            u'化工': u'CHEM',
            u'有色金属': u'NonFerMetal',
            u'钢铁': u'IronSteel',
            u'建筑材料': u'CONMAT',
            u'建筑装饰': u'BuildDeco',
            u'电气设备': u'ELECEQP',
            u'机械设备': u'MachiEquip',
            u'国防军工': u'AERODEF',
            u'汽车': u'Auto',
            u'电子': u'Electronics',
            u'家用电器': u'HouseApp',
            u'食品饮料': u'FoodBever',
            u'纺织服装': u'Textile',
            u'轻工制造': u'LightIndus',
            u'医药生物': u'Health',
            u'公用事业': u'Utilities',
            u'交通运输': u'Transportation',
            u'房地产': u'RealEstate',
            u'商业贸易': u'CommeTrade',
            u'休闲服务': u'LeiService',
            u'计算机': u'Computer',
            u'传媒': u'Media',
            u'通信': u'Telecom',
            u'综合': u'Conglomerates',
            u'非银金融': u'NonBankFinan',
            # u'证券': u'Securities',
            # u'多元金融': u'MultiFin',
            # u'保险': u'Insurance'
            }

uqer_rm_mapper = {u'银行': u'Bank',
                  u'农林牧渔': u'AgriForest',
                  u'采掘': u'Mining',
                  u'化工': u'CHEM',
                  u'有色金属': u'NonFerMetal',
                  u'钢铁': u'IronSteel',
                  u'建筑材料': u'CONMAT',
                  u'建筑装饰': u'BuildDeco',
                  u'电气设备': u'ELECEQP',
                  u'机械设备': u'MachiEquip',
                  u'国防军工': u'AERODEF',
                  u'汽车': u'Auto',
                  u'电子': u'Electronics',
                  u'家用电器': u'HouseApp',
                  u'食品饮料': u'FoodBever',
                  u'纺织服装': u'Textile',
                  u'轻工制造': u'LightIndus',
                  u'医药生物': u'Health',
                  u'公用事业': u'Utilities',
                  u'交通运输': u'Transportation',
                  u'房地产': u'RealEstate',
                  u'商业贸易': u'CommeTrade',
                  u'休闲服务': u'LeiService',
                  u'计算机': u'Computer',
                  u'传媒': u'Media',
                  u'通信': u'Telecom',
                  u'综合': u'Conglomerates',
                  u'非银金融': u'NonBankFinan',
                  'ResVol': 'RESVOL',
                  'Momentum': 'MOMENTUM',
                  'Liquidity': 'LIQUIDTY',
                  'NLSIZE': 'SIZENL',
                  'Earning Yild': 'EARNYILD',
                  'Growth': 'GROWTH',
                  'Leverage': 'LEVERAGE'}

uqer_rm_mapper_R = {}
for k, v in uqer_rm_mapper.iteritems():
    uqer_rm_mapper_R[v] = k


def exclude_subnew(stklist, tradeDate, diff):
    listDate = DataAPI.EquGet(secID=stklist,
                              field=u"secID,listDate").set_index('secID')['listDate']
    distance = listDate.map(lambda x: cal.bizDatesNumber(x, tradeDate))
    return distance.index[distance >= diff]


def regular_returns(symbols, yesterday, today, diff):
    st_secid = DataAPI.SecSTGet(beginDate=yesterday,
                                endDate=yesterday,
                                secID=symbols,
                                ticker=u"", field=u"secID", pandas="1")['secID'].values

    div_secid = DataAPI.EquDivGet(secID=symbols,
                                  exDivDate=today.replace('-', ''),
                                  field=u"secID", pandas="1")['secID'].values

    symbols = exclude_subnew(symbols.drop((set(st_secid) | set(div_secid))), yesterday, diff)
    return symbols


# @ConnectionErrorDeco(5)
# def regression(diff=30):
#     factor_returns = pd.DataFrame()
#     spret = pd.DataFrame()
#
#     expo_dates = RMDataBase.get_date_list(RMDataBase.expo_table)
#     spret_dates = RMDataBase.get_date_list(RMDataBase.spret_table)
#     factor_ret_dates = RMDataBase.get_date_list(RMDataBase.factor_returns_table)
#
#     lack_dates = set(date_process(dates=expo_dates, forward=-1)) - (set(spret_dates) & set(factor_ret_dates))
#     lack_dates = sorted(filter(lambda x: x <= TODAY, lack_dates))
#     all_a = DataAPI.EquGet(equTypeCD=u"A", listStatusCD="L,S,DE,UN", field=u"secID", pandas="1").values.T[0]
#     # endDate = min(cal.advanceDate(end, '1B').strftime("%Y-%m-%d"), TODAY)
#     ret_data = DataAPI.MktEqudGet(secID=all_a,
#                                   beginDate=lack_dates[0],
#                                   endDate=lack_dates[-1],
#                                   field=u"secID,tradeDate,chgPct,isOpen", pandas="1")
#
#     ret = ret_data.pivot('tradeDate', 'secID', 'chgPct')
#     for day in lack_dates:
#         if day > ret.index[-1]:
#             break
#         yesterday = cal.advanceDate(day, '-1B').strftime("%Y-%m-%d")
#         # 去掉复牌的股票
#         today_ret = ret.loc[day].dropna().drop(['601313.XSHG', '601360.XSHG'], errors='ignore')
#         # 去掉涨跌幅异常以及st股票
#         symbols = today_ret[(today_ret.abs() <= 0.101)].index
#         print len(today_ret) - len(symbols)
#         st_secid = DataAPI.SecSTGet(beginDate=yesterday,
#                                     endDate=yesterday,
#                                     secID=symbols,
#                                     ticker=u"", field=u"secID", pandas="1")['secID'].values
#
#         div_secid = DataAPI.EquDivGet(secID=symbols,
#                                       exDivDate=day.replace('-', ''),
#                                       field=u"secID", pandas="1")['secID'].values
#
#         symbols = exclude_subnew(symbols.drop((set(st_secid) | set(div_secid))), yesterday, diff)
#
#         # 提取因子暴露
#         today_expo = (RMDataBase.RMExpoGet(MY_RISK_FACTORS, symbols=[], tradeDate=yesterday)
#             .set_index('Symbol')
#             .drop('Date', axis=1))
#         tmp_expo = today_expo.reindex(symbols).dropna(how='all')
#         y = today_ret.reindex(tmp_expo.index).fillna(0)
#         x = fillna(tmp_expo, tradeDate=yesterday, industry_mean=True).dropna(how='all')
#
#         weights = np.sqrt(DataAPI.MktStockFactorsOneDayProGet(tradeDate=yesterday.replace('-', ''),
#                                                               secID=y.index,
#                                                               ticker=u"",
#                                                               field=u"secID,MktValue", pandas="1").set_index('secID')[
#                               'MktValue'])
#         weights /= weights.sum()
#         # y = y.reindex(index=x.index)
#         tmp_res = sm.WLS(y, x, weights).fit()
#         # tmp_spret = tmp_res.resid
#         tmp_factor_returns = tmp_res.params
#         spret[day] = (ret.loc[day] - today_expo.dot(tmp_factor_returns)).dropna()
#         factor_returns[day] = tmp_factor_returns
#
#     spret_result = spret.T.stack().rename_axis(['Date', 'Symbol']).rename('spret').to_frame().reset_index()
#     factor_returns_result = factor_returns.T.dropna(how='all').rename_axis('Date').reset_index()
#     return spret_result, factor_returns_result


# def migrate(start, end, chunksize=500, replace=False):
#     dates = date_process(start=start, end=end, period=1)
#     group_num = len(dates) / chunksize + 1
#     date_arr = np.array_split(dates, group_num)
#
#     for arr in date_arr:
#         print arr[0]
#         print arr[-1]
#         obj = StyleExtractor(arr[0], arr[-1])
#         obj.migrate_expo(replace=replace)
#     spret, factor_ret = regression()
#     RMDataBase.to_factor_returns(factor_ret, chunksize=chunksize, replace=replace)
#     RMDataBase.to_spret(spret, chunksize=chunksize, replace=replace)
#
#
# def daily_maintain(start='2012-06-30', end=None, chunksize=100):
#     end_data = RMDataBase.find_end(RMDataBase.expo_table)
#     if end is None:
#         end = TODAY
#
#     if end_data is None:
#         migrate(start, end, chunksize=chunksize, replace=True)
#
#     else:
#         start = end_data
#         migrate(start, end, chunksize=chunksize, replace=False)


def const_cls(otable):
    table = copy.deepcopy(otable)

    class Data(declarative_base()):
        __tablename__ = table.name
        Date = Column('Date', VARCHAR(10), primary_key=True)

    for col in table.c.values():
        col.table = None
        if col.name in map_dict:
            setattr(Data, map_dict[col.name], col)
        elif col.name != 'Date':
            setattr(Data, col.name, col)
    return Data


class RMDataBase:
    sql_engine = create_engine('sqlite:///riskmodel test.db')
    db_obj = pd.io.sql.SQLDatabase(sql_engine)  # 创建数据库对象
    metadata = MetaData(bind=sql_engine)  # 生成元数据
    metadata.create_all()

    factor_returns_table = Table('factor returns',
                                 metadata,
                                 *([Column('Date', VARCHAR(10), primary_key=True)] + [Column(col, Float) for col in
                                                                                      MY_RISK_FACTORS]),
                                 extend_existing=True)
    spret_table = Table('spret',
                        metadata,
                        Column('Date', VARCHAR(10), primary_key=True),
                        Column('Symbol', VARCHAR(10), primary_key=True),
                        Column('spret', Float),
                        extend_existing=True)
    expo_table = Table('risk_expo',
                       metadata,
                       *([Column('Date', VARCHAR(10), primary_key=True),
                          Column('Symbol', VARCHAR(10), primary_key=True)] + [Column(col, Float) for col in
                                                                              MY_RISK_FACTORS]),
                       extend_existing=True)

    for table in [factor_returns_table, spret_table, expo_table]:
        if not table.exists():
            table.create()

    @classmethod
    def get_date_list(cls, table):
        return RMDataBase.db_obj.read_query(select([table.c.Date],
                                                   group_by=table.c.Date))['Date'].values

    @classmethod
    def find_start(cls, table):
        begin_data = pd.read_sql_query(select([table.c.Date],
                                              limit=1),
                                       cls.sql_engine).values
        if len(begin_data):
            return begin_data[0, 0]

    @classmethod
    def find_end(cls, table):
        end_data = pd.read_sql_query(select([table.c.Date],
                                            order_by=[desc(table.c.Date)],
                                            limit=1),
                                     cls.sql_engine).values
        if len(end_data):
            return end_data[0, 0]

    @classmethod
    def get_date_expr(cls, beginDate, endDate, tradeDate, table):
        if tradeDate is not None:
            return table.c.Date == tradeDate

        if beginDate is None:
            beginDate = cls.find_start(table)

        if endDate is None:
            endDate = cls.find_end(table)
        date_expr = table.c.Date.between(beginDate, endDate)
        return date_expr

    @classmethod
    def RMFactorReturnsGet(cls, risk_factors, beginDate=None, endDate=None, tradeDate=None):
        date_expr = cls.get_date_expr(beginDate=beginDate, endDate=endDate, tradeDate=tradeDate,
                                      table=cls.factor_returns_table)
        query = select([cls.factor_returns_table.c.get(fac) for fac in ['Date'] + risk_factors],
                       date_expr)
        return cls.db_obj.read_query(query)

    @classmethod
    def RMSpretGet(cls, symbols=[], beginDate=None, endDate=None, tradeDate=None):
        date_expr = cls.get_date_expr(beginDate=beginDate, endDate=endDate, tradeDate=tradeDate,
                                      table=cls.spret_table)

        and_condition = [date_expr]
        if len(symbols):
            and_condition.append(cls.spret_table.c.Symbol.in_(symbols))
        query = cls.spret_table.select(and_(*and_condition))
        return cls.db_obj.read_query(query).pivot('Date', 'Symbol', 'spret')

    @classmethod
    def RMExpoGet(cls, risk_factors, symbols=[], beginDate=None, endDate=None, tradeDate=None):
        date_expr = cls.get_date_expr(beginDate=beginDate, endDate=endDate, tradeDate=tradeDate,
                                      table=cls.expo_table)

        and_condition = [date_expr]
        if len(symbols):
            and_condition.append(cls.expo_table.c.Symbol.in_(symbols))
        query = select([cls.expo_table.c.get(fac) for fac in ['Date', 'Symbol'] + risk_factors],
                       and_(*and_condition))
        return cls.db_obj.read_query(query)

    @classmethod
    def _to_sql(cls, data, table, replace, chunksize):
        print "inserting %s data" % table.name
        if replace:
            table.delete().execute()
            pd.io.sql.to_sql(data, table.name, con=cls.sql_engine, if_exists='append', index=False, chunksize=chunksize)

        else:
            dt_cls = const_cls(otable=table)
            Session = sessionmaker(bind=cls.sql_engine)  # 创建会话
            session = Session()

            m = mapper(dt_cls, table, non_primary=True)  # 将类与表对象映射
            map_df = data.rename(columns=map_dict)
            for i in range(map_df.shape[0]):
                row_data = map_df.iloc[i]  # 获取每行数据
                if session.query(exists().where(
                        and_(*map(lambda x: getattr(dt_cls, x.name) == row_data[x.name], m.primary_key)))).scalar():
                    session.bulk_update_mappings(dt_cls, row_data.to_dict())
                else:
                    session.add(dt_cls(**row_data.to_dict()))
                # 每隔一定数量提交一次
                if i % chunksize == 0:
                    print i, '\r'
                    session.commit()
            session.commit()  # 最终提交
            session.close()  # 关闭会话
        print "%s data inserted" % table.name

    @classmethod
    def to_expo(cls, data, replace=False, chunksize=1e3):
        cls._to_sql(data=data, table=cls.expo_table, replace=replace, chunksize=chunksize)

    @classmethod
    def to_spret(cls, data, replace=False, chunksize=1e3):
        cls._to_sql(data=data, table=cls.spret_table, replace=replace, chunksize=chunksize)

    @classmethod
    def to_factor_returns(cls, data, replace=False, chunksize=1e3):
        cls._to_sql(data=data, table=cls.factor_returns_table, replace=replace, chunksize=chunksize)


@ConnectionErrorDeco(5, True)
class StyleExtractor:
    all_a = DataAPI.EquGet(equTypeCD=u"A", listStatusCD="L,S,DE,UN", field=u"secID", pandas="1").values.T[0]

    map_dict = {'ResVol': pd.Series([0.74, 0.1, 0.16],
                                    index=['DASTD', 'HsigmaCNE5', 'CmraCNE5']),
                'Liquidity': pd.Series([0.35, 0.35, 0.3],
                                       index=['STOM', 'STOQ', 'STOA']),
                'Earning Yild': pd.Series([0.5, 0.5],
                                          index=['CETOP', 'ETOP']),
                'Growth': pd.Series([0.5, 0.5],
                                    index=['EGRO', 'SGRO']),
                'Leverage': pd.Series([0.74, 0.1, 0.16],
                                      index=['MLEV', 'DebtsAssetRatio', 'BLEV']),
                'Momentum': pd.Series([1.0],
                                      index=['RSTR24'])}

    def __init__(self, start, end):
        self.today = TODAY
        self.start = start
        self.end = end
        dates = date_process(start=start, end=end, period=1, forward=0)
        beginDate = cal.advanceDate(start, '-251B').strftime("%Y-%m-%d")
        endDate = min(cal.advanceDate(end, '1B').strftime("%Y-%m-%d"), self.today)
        self.ret_data = DataAPI.MktEqudGet(secID=self.all_a,
                                           beginDate=beginDate,
                                           endDate=endDate,
                                           field=u"secID,tradeDate,chgPct,isOpen", pandas="1")
        factors = ['ASSI', 'EquityToAsset', 'MktValue', 'LCAP', 'DASTD', 'HsigmaCNE5', 'CmraCNE5',
                   'STOM', 'STOA', 'STOQ', 'CETOP', 'ETOP', 'EGRO', 'SGRO', 'MLEV', 'BLEV',
                   'DebtsAssetRatio', 'RSTR24']

        arr = []
        for day in dates:
            df = DataAPI.MktStockFactorsOneDayProGet(tradeDate=day,
                                                     secID=self.all_a,
                                                     field=['secID'] + factors, pandas="1").set_index('secID')
            if not len(df):
                dates.remove(day)
                continue
            df = df.sort_values('MktValue').iloc[int(df['MktValue'].count()*0.1):]
            arr.append(df)

        self.data = pd.concat(arr, keys=dates).groupby(level=[0, 1]).first()
        self.dates = dates

    def _merge(self, expo, param):
        def _mul(ax):
            if ax.all():
                return param
            mul_params = ax.mul(param)
            return mul_params / mul_params.sum()

        df = expo.dropna(how='all')
        return df.notnull().apply(_mul, axis=1).mul(df).sum(axis=1)

    def _get_single_style(self, factor):
        param = self.map_dict[factor]
        df = (self.data.reindex(columns=param.index)
              .swaplevel()
              .unstack()
              .apply(lambda x: standardize(winsorize(x)))
              .stack()
              .swaplevel()
              .sort_index())
        return self._merge(df, param).rename(factor)

    def get_ResVol(self):
        return self._get_single_style('ResVol')

    def get_Liquidity(self):
        return self._get_single_style('Liquidity')

    def get_Earning_Yild(self):
        return self._get_single_style('Earning Yild')

    def get_Growth(self):
        return self._get_single_style('Growth')

    def get_Leverage(self):
        param = self.map_dict['Leverage']
        res = self.data.eval("adjust_MLEV = 1 / (1 - MLEV)", inplace=False).eval("adjust_BLEV = BLEV+1", inplace=False)
        df = res[['adjust_MLEV', 'adjust_BLEV', 'DebtsAssetRatio']].rename(columns={'adjust_MLEV': 'MLEV',
                                                                                    'adjust_BLEV': 'BLEV'})
        a = self._merge(df, param).rename('Leverage')
        return np.log(-1/a)

    def get_btop(self):
        section = self.data
        return (np.exp(section['ASSI']) * section['EquityToAsset'] / section['MktValue']).rename('BTOP')

    def get_momentum(self):
        return self.data['RSTR24'].rename('Momentum')

    def get_beta(self):
        # beginDate = cal.advanceDate(start, '-251B').strftime("%Y-%m-%d")
        # ret_data = DataAPI.MktEqudGet(secID=self.all_a,
        #                               beginDate=beginDate,
        #                               endDate=end,
        #                               isOpen='1',
        #                               field=u"secID,tradeDate,chgPct", pandas="1")
        ret = self.ret_data.query("isOpen==1").pivot('tradeDate', 'secID', 'chgPct')
        idx_ret = ret.apply(lambda x: x[set_universe('000906.ZICN', x.name)].mean(), axis=1)
        ret['index'] = idx_ret
        lbd = 0.5 ** (1.0 / 63)
        weights = np.array([lbd ** (250 - j - 1) for j in range(250)])
        weights /= weights.sum()
        res = []
        for symbol in ret.columns[:-1]:
            # print symbol
            clear_data = ret[[symbol, 'index']].dropna()
            arr = []
            for i in range(clear_data.shape[0] - 250):
                y, x = clear_data.iloc[i:i + 250].T.values
                arr.append(sm.WLS(y, sm.add_constant(x), weights).fit().params[-1])
            res.append(pd.Series(arr, index=clear_data.index[250:], name=symbol))
        return pd.concat(res, axis=1).truncate(self.start, self.end).stack().rename('BETA')

    def get_size(self):
        beginDate = cal.advanceDate(self.start, '-20B').strftime("%Y-%m-%d")
        mkt = DataAPI.MktEqudGet(secID=self.all_a,
                                 beginDate=beginDate,
                                 endDate=self.start,
                                 field=u"secID,tradeDate,marketValue", pandas="1")
        df1 = mkt.pivot('tradeDate', 'secID', 'marketValue')
        df2 = self.data['MktValue'].unstack()
        size = df1.append(df2).groupby(level=0).first()
        return np.log(size.rolling(20).mean().truncate(self.start, self.end).dropna(how='all')).stack().rename('SIZE')

    def transfer_style(self):
        res = [self.get_Earning_Yild(),
               self.get_Growth(),
               self.get_Leverage(),
               self.get_Liquidity(),
               self.get_ResVol(),
               self.get_btop(),
               self.get_size(),
               self.get_momentum(),
               self.get_beta()]
        return pd.concat(res, axis=1).dropna(how='all').replace([-np.inf, np.inf], np.nan)

    def results(self):
        expo = self.transfer_style()

        def _regular(x):
            if x.name == 'SIZE':
                return standardize(x)
            return standardize(winsorize(x))
        arr = []
        for day in self.dates:
            print day
            tmp_data = expo.loc[day]

            regu_data = fill_industry_mean(tmp_data.apply(_regular), day).fillna(0)
            symbols = regu_data.index
            dummy = get_industry(symbols, day).apply(lambda x: pd.Series(INDUSTRY_LIST == x,
                                                                         index=INDUSTRY_LIST,
                                                                         dtype=int))
            dummy['非银金融'] = dummy[['证券', '多元金融', '保险']].sum(axis=1)
            dummy = dummy.drop(['证券', '多元金融', '保险'], axis=1)
            # regu_data['SIZE'] = standardize(winsorize(sm.OLS(regu_data['SIZE'], dummy).fit().resid))
            regu_data = regu_data.assign(Growth=standardize(winsorize(sm.OLS(regu_data['Growth'], dummy).fit().resid)),
                                         Liquidity=standardize(
                                             winsorize(sm.OLS(regu_data['Liquidity'],
                                                              regu_data[['SIZE']].join(dummy)).fit().resid)),
                                         Momentum=standardize(
                                             winsorize(sm.OLS(regu_data['Momentum'],
                                                              regu_data[['SIZE']].join(dummy)).fit().resid)),
                                         BTOP=standardize(
                                             winsorize(sm.OLS(regu_data['BTOP'],
                                                              regu_data[['SIZE']].join(dummy)).fit().resid)),
                                         ResVol=standardize(
                                             winsorize(sm.OLS(regu_data['ResVol'],
                                                              regu_data[['SIZE', 'BETA']].join(dummy)).fit().resid)),
                                         NLSIZE=standardize(
                                             winsorize(
                                                 sm.OLS(tmp_data['SIZE'].dropna().pow(3), tmp_data['SIZE'].dropna()).fit().resid)))

            arr.append(regu_data.join(dummy).fillna(0))
        res = pd.concat(arr, keys=self.dates).assign(COUNTRY=1)
        res.columns = res.columns.map(lambda x: unicode(x, 'utf-8', 'ignore'))
        return res.rename_axis(['Date', 'Symbol']).reset_index()

    # def regression(self, diff=30):
    #     # style_expo = self.results()
    #     factor_returns = pd.DataFrame()
    #     spret = pd.DataFrame()
    #     ret = self.ret_data.pivot('tradeDate', 'secID', 'chgPct').shift(-1).truncate(self.start)
    #     for day in self.dates:
    #         print day
    #         if day == self.today:
    #             break
    #         # 去掉复牌的股票
    #         today_ret = ret.loc[day].dropna().drop('601360.XSHG', errors='ignore')
    #         # 去掉涨跌幅异常以及st股票
    #         symbols = today_ret[(today_ret.abs() <= 0.101)].index
    #         print len(today_ret) - len(symbols)
    #         st_secid = DataAPI.SecSTGet(beginDate=day,
    #                                     endDate=day,
    #                                     secID=symbols,
    #                                     ticker=u"", field=u"secID", pandas="1")['secID'].values
    #
    #         div_secid = DataAPI.EquDivGet(secID=symbols,
    #                                       exDivDate=ret.index[ret.index.get_loc(day) + 1].replace('-', ''),
    #                                       field=u"secID", pandas="1")['secID'].values
    #
    #         symbols = exclude_subnew(symbols.drop((set(st_secid) | set(div_secid))), day, diff)
    #
    #         # 提取因子暴露
    #         today_expo = (RMDataBase.RMExpoGet(MY_RISK_FACTORS, symbols=symbols, tradeDate=day)
    #             .set_index('Symbol')
    #             .drop('Date', axis=1))
    #         tmp_expo = today_expo.reindex(symbols).dropna(how='all')
    #         # tmp_expo = style_expo.loc[day].reindex(symbols).dropna(how='all')
    #         # tmp_expo = fillna(tmp_expo, tradeDate=day, industry_mean=True).dropna(how='all')
    #         y = today_ret.reindex(tmp_expo.index).fillna(0)
    #         x = fillna(tmp_expo, tradeDate=day, industry_mean=True).dropna(how='all')
    #         weights = np.sqrt(DataAPI.MktEqudGet(secID=y.index,
    #                                              tradeDate=day,
    #                                              field=u"secID,marketValue", pandas="1").set_index('secID')[
    #                               'marketValue'])
    #         weights /= weights.sum()
    #         # y = y.reindex(index=x.index)
    #         tmp_res = sm.WLS(y, x, weights).fit()
    #         # tmp_spret = tmp_res.resid
    #         tmp_factor_returns = tmp_res.params
    #         spret[day] = (ret.loc[day] - today_expo.dot(tmp_factor_returns)).dropna()
    #         factor_returns[day] = tmp_factor_returns
    #
    #     spret_result = spret.T.shift().stack().rename_axis(['Date', 'Symbol']).rename('spret').to_frame().reset_index()
    #     factor_returns_result = factor_returns.T.shift().dropna(how='all').rename_axis('Date').reset_index()
    #     return spret_result, factor_returns_result

    # def migrate_expo(self, replace=False, chunksize=1e3):
    #     expo = self.results().reset_index()
    #     RMDataBase.to_expo(expo, chunksize=chunksize, replace=replace)

    # spret, factor_ret = self.regression()
    # RMDataBase.to_factor_returns(factor_ret, chunksize=chunksize, replace=replace)
    # RMDataBase.to_spret(spret, chunksize=chunksize, replace=replace)


class Maintain(RMDataBase):
    def __init__(self, start='2012-06-30', end=None, chunksize=100, replace=False):
        self.all_a = DataAPI.EquGet(equTypeCD=u"A", listStatusCD="L,S,DE,UN", field=u"secID", pandas="1").values.T[0]
        expo_tail = self.find_end(self.expo_table)
        self.expo_is_empty = expo_tail is None
        if self.expo_is_empty:
            self.start = start
        else:
            self.start = expo_tail

        if end is None:
            self.end = TODAY
        else:
            self.end = end
        self.dates = date_process(start=self.start, end=self.end, period=1)
        self.chunksize = chunksize
        self.replace = replace

    def migrate_expo(self):
        group_num = len(self.dates) / self.chunksize + 1
        date_arr = np.array_split(self.dates, group_num)
        for arr in date_arr:
            print arr[0]
            print arr[-1]
            obj = StyleExtractor(arr[0], arr[-1])
            expo_results = obj.results()
            if self.expo_is_empty:
                self.to_expo(expo_results, replace=True)
                self.expo_is_empty = False
            else:
                self.to_expo(expo_results, replace=self.replace)

    @ConnectionErrorDeco(5)
    def get_factor_returns(self, lack_dates, diff=30):
        yesterday_list = date_process(dates=lack_dates, forward=1)
        ret_data = DataAPI.MktEqudGet(secID=self.all_a,
                                      beginDate=lack_dates[0],
                                      endDate=lack_dates[-1],
                                      field=u"secID,tradeDate,chgPct,isOpen", pandas="1")

        ret = ret_data.pivot('tradeDate', 'secID', 'chgPct')
        factor_returns = pd.DataFrame()
        for yesterday, today in zip(yesterday_list, lack_dates):
            if today > ret.index[-1]:
                break

            today_ret = ret.loc[today].dropna().drop(['601313.XSHG', '601360.XSHG'], errors='ignore')
            # 去掉涨跌幅异常以及st股票
            symbols = today_ret[(today_ret.abs() <= 0.101)].index
            print len(today_ret) - len(symbols)
            symbols = regular_returns(symbols=symbols, yesterday=yesterday, today=today, diff=diff)
            # st_secid = DataAPI.SecSTGet(beginDate=yesterday,
            #                             endDate=yesterday,
            #                             secID=symbols,
            #                             ticker=u"", field=u"secID", pandas="1")['secID'].values
            #
            # div_secid = DataAPI.EquDivGet(secID=symbols,
            #                               exDivDate=today.replace('-', ''),
            #                               field=u"secID", pandas="1")['secID'].values
            #
            # symbols = exclude_subnew(symbols.drop((set(st_secid) | set(div_secid))), yesterday, diff)

            # 提取因子暴露
            today_expo = (self.RMExpoGet(MY_RISK_FACTORS, symbols=[], tradeDate=yesterday)
                          .set_index('Symbol')
                          .drop('Date', axis=1))
            tmp_expo = today_expo.reindex(symbols).dropna(how='all')
            y = today_ret.reindex(tmp_expo.index).fillna(0)
            x = fillna(tmp_expo, tradeDate=yesterday, industry_mean=True).dropna(how='all')

            weights = np.sqrt(DataAPI.MktStockFactorsOneDayProGet(tradeDate=yesterday.replace('-', ''),
                                                                  secID=y.index,
                                                                  ticker=u"",
                                                                  field=u"secID,MktValue", pandas="1").set_index(
                'secID')['MktValue'])
            weights /= weights.sum()
            weights = weights.groupby(level=0).first().reindex_like(y).fillna(0)

            # y = y.reindex(index=x.index)
            tmp_res = sm.WLS(y, x, weights).fit()
            # tmp_spret = tmp_res.resid
            tmp_factor_returns = tmp_res.params
            # spret[today] = (ret.loc[today] - today_expo.dot(tmp_factor_returns)).dropna()
            factor_returns[today] = tmp_factor_returns
        factor_returns_result = factor_returns.T.dropna(how='all').rename_axis('Date').reset_index()
        return factor_returns_result

    def get_spret(self, lack_dates):
        spret = pd.DataFrame()
        yesterday_list = date_process(dates=lack_dates, forward=1)
        ret_data = DataAPI.MktEqudGet(secID=self.all_a,
                                      beginDate=lack_dates[0],
                                      endDate=lack_dates[-1],
                                      field=u"secID,tradeDate,chgPct,isOpen", pandas="1")

        ret = ret_data.pivot('tradeDate', 'secID', 'chgPct')
        for yesterday, today in zip(yesterday_list, lack_dates):
            today_ret = ret.loc[today]
            today_expo = (self.RMExpoGet(MY_RISK_FACTORS, symbols=[], tradeDate=yesterday)
                          .set_index('Symbol')
                          .drop('Date', axis=1))
            today_factor_returns = self.RMFactorReturnsGet(MY_RISK_FACTORS, tradeDate=today).set_index('Date').iloc[0]
            spret[today] = (today_ret - today_expo.dot(today_factor_returns)).dropna()
        return spret.T.stack().rename_axis(['Date', 'Symbol']).rename('spret').to_frame().reset_index()

    def migrate_factor_returns(self, diff=30):
        expo_dates = self.get_date_list(self.expo_table)
        # spret_dates = self.get_date_list(RMDataBase.spret_table)
        factor_ret_dates = self.get_date_list(self.factor_returns_table)

        lack_dates = set(date_process(dates=expo_dates, forward=-1)) - set(factor_ret_dates)
        lack_dates = sorted(filter(lambda x: x <= TODAY, lack_dates))
        if len(lack_dates):
            group_num = len(lack_dates) / self.chunksize + 1
            date_arr = np.array_split(lack_dates, group_num)
            for arr in date_arr:
                tmp_factor_returns_results = self.get_factor_returns(lack_dates=arr, diff=diff)
                self.to_factor_returns(tmp_factor_returns_results)

    def migrate_spret(self):
        # expo_dates = self.get_date_list(self.expo_table)
        spret_dates = self.get_date_list(self.spret_table)
        factor_ret_dates = self.get_date_list(self.factor_returns_table)

        lack_dates = set(factor_ret_dates) - set(spret_dates)
        lack_dates = sorted(filter(lambda x: x <= TODAY, lack_dates))

        if len(lack_dates):
            group_num = len(lack_dates) / self.chunksize + 1
            date_arr = np.array_split(lack_dates, group_num)
            for arr in date_arr:
                tmp_spret = self.get_spret(lack_dates=arr)
                self.to_spret(tmp_spret)

    def daily_maintain(self):
        self.migrate_expo()
        self.migrate_factor_returns()
        self.migrate_spret()


class UqerRM:
    @classmethod
    def _format_date(cls, beginDate='', endDate='', tradeDate=''):
        return map(lambda x: x.replace('-', ''), [beginDate, endDate, tradeDate])

    @classmethod
    def RMExpoGet(cls, risk_factors, symbols=[], beginDate='', endDate='', tradeDate=''):
        formated_beginDate, formated_endDate, formated_tradeDate = cls._format_date(beginDate, endDate, tradeDate)
        res = DataAPI.RMExposureDayGet(secID=symbols,
                                       tradeDate=formated_tradeDate,
                                       beginDate=formated_beginDate,
                                       endDate=formated_endDate,
                                       field=[u"tradeDate", "secID"] + map(
                                           lambda fac: uqer_rm_mapper[fac] if fac in uqer_rm_mapper else fac,
                                           risk_factors),
                                       pandas="1")

        res['tradeDate'] = pd.to_datetime(res['tradeDate']).astype(str)
        return res.rename(columns=dict(uqer_rm_mapper_R,
                                       **{'secID': 'Symbol',
                                          'tradeDate': 'Date'}))

    @classmethod
    def RMFactorReturnsGet(cls, risk_factors, beginDate='', endDate='', tradeDate=''):
        formated_beginDate, formated_endDate, formated_tradeDate = cls._format_date(beginDate, endDate, tradeDate)
        res = DataAPI.RMFactorRetDayGet(tradeDate=formated_tradeDate,
                                        beginDate=formated_beginDate,
                                        endDate=formated_endDate,
                                        field=[u"tradeDate"] + map(
                                            lambda fac: uqer_rm_mapper[fac] if fac in uqer_rm_mapper else fac,
                                            risk_factors), pandas="1")
        res['tradeDate'] = pd.to_datetime(res['tradeDate']).astype(str)
        return res.rename(columns=dict(uqer_rm_mapper_R,
                                       **{'tradeDate': 'Date'}))

    @classmethod
    def RMSpretGet(cls, symbols=[], beginDate='', endDate='', tradeDate=''):
        formated_beginDate, formated_endDate, formated_tradeDate = cls._format_date(beginDate, endDate, tradeDate)
        res = DataAPI.RMSpecificRetDayGet(secID=symbols,
                                          tradeDate=formated_tradeDate,
                                          beginDate=formated_beginDate,
                                          endDate=formated_endDate,
                                          field=u"tradeDate,secID,spret", pandas="1")
        res['tradeDate'] = pd.to_datetime(res['tradeDate']).astype(str)
        return res.rename(columns={'secID': 'Symbol',
                                   'tradeDate': 'Date'})


def set_interface(mode='uqer'):
    if mode == 'uqer':
        return UqerRM

    if mode == 'self':
        return RMDataBase

    raise ValueError("未知的风险模型接口")
