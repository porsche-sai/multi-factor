# coding=utf-8

from lib.tools import *
from lib.Barra_test import Block
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.style.use('seaborn-notebook')
legendFont = font.copy()

# ************************************************
# 快速回测表现报告
# ************************************************
BENCHMARK_DICT = {'000905': u"中证500指数增强",
                  '000300': u"沪深300指数增强",
                  '多因子模型': u"多因子模型"}

STYLE_FACTORS = ["EARNYILD", "SIZE", "MOMENTUM", "RESVOL", "BTOP", "LIQUIDTY", "GROWTH", "BETA", "LEVERAGE", "SIZENL"]
INDUSTRY_FACTORS = [u'Transportation', u'RealEstate', u'Electronics', u'Mining', u'AERODEF', u'ELECEQP', u'Textile',
                    u'CHEM', u'FoodBever', u'Computer', u'Bank', u'HouseApp', u'MachiEquip', u'Utilities',
                    u'CommeTrade', u'Telecom', u'LeiService', u'Auto', u'CONMAT', u'AgriForest', u'BuildDeco',
                    u'IronSteel', u'Health', u'NonFerMetal', u'Media', u'LightIndus', u'Conglomerates']


class btBlock(Block):
    def __init__(self, bt_block, benchmark, riskfactors):
        super(btBlock, self).__init__(symbols=bt_block['security_position'].keys(), tradeDate=bt_block['tradeDate'],
                                      benchmark=benchmark)

        self.riskfactors = riskfactors
        self.weights = pd.DataFrame(bt_block['security_position']).T['value'] / bt_block['portfolio_value']
        self.industry_weights = self.weights.dot(self.dummy)
        self.alpha_expo = bt_block['alpha_expo']
        if 'solution' in bt_block.index:
            self.solution = bt_block['solution']

    def get_relative_expo(self):
        tot_risk_expo = DataAPI.RMExposureDayGet(secID=list(set(self.symbols) | set(self.bench_symbols)),
                                                 tradeDate=self.tradeDate.replace('-', ''),
                                                 field=['secID'] + self.riskfactors,
                                                 pandas="1").set_index('secID').fillna(0)
        return tot_risk_expo.T.mul(self.solution).sum(axis=1) - tot_risk_expo.T.mul(self.bench_symbol_weight).sum(
            axis=1)

    def get_alpha_expo(self, weighted=True):
        if weighted:
            assert hasattr(self, 'solution')
            res = self.alpha_expo.T.mul(self.solution).T
            return res[res != 0].dropna(how='all').fillna(0)
        return self.alpha_expo


class btChain:
    def __init__(self, bt, perf, benchmark, alpha_expo_list, risk_alpha_list=[], style_factors=STYLE_FACTORS,
                 industry_factors=INDUSTRY_FACTORS):
        bt['tradeDate'] = bt['tradeDate'].astype(str)
        self.bt = bt
        self.perf = perf
        self.risk_alpha_list = risk_alpha_list
        self.style_factors = style_factors
        self.industry_factors = industry_factors
        self.riskfactors = self.style_factors + self.industry_factors
        self.benchmark = benchmark

        self.trade_bt = bt.dropna(subset=['security_trades'])
        self.trade_bt['alpha_expo'] = alpha_expo_list.values
        self.dates = bt['tradeDate'].values
        self.trade_dates = self.trade_bt['tradeDate'].values

        self.portfolio_returns = perf['returns']
        self.bench_returns = perf['benchmark_returns']

        obj_arr = [btBlock(bt_block=bt_block[1],
                           benchmark=benchmark,
                           riskfactors=self.riskfactors) for bt_block in self.trade_bt.iterrows()]
        self.block_chain = pd.Series(obj_arr, index=self.trade_dates)

        self.tot_symbols = list(reduce(lambda x, y: x | y,
                                       self.block_chain.apply(
                                           lambda x: set(x.symbols) | set(x.bench_symbols) | set(x.alpha_expo.index))))

    def get_cumret(self, ret_data):
        """计算每期的累计收益"""
        arr = []
        for i in range(len(self.trade_dates)):
            if i + 1 == len(self.trade_dates):
                res = (ret_data.truncate(self.trade_dates[i]) + 1).prod()
            else:
                res = (ret_data.truncate(self.trade_dates[i], self.trade_dates[i + 1]).iloc[:-1] + 1).prod()

            arr.append(res.rename(self.trade_dates[i]))
        return pd.concat(arr, axis=1).T - 1

    def get_spret_data(self):
        """提取所有涉及股票的特质收益"""
        data = DataAPI.RMSpecificRetDayGet(secID=self.tot_symbols,
                                           beginDate=self.dates[0].replace('-', ''),
                                           endDate=self.dates[-1].replace('-', ''),
                                           field=['secID', 'tradeDate', 'spret'])
        spret = data.pivot('tradeDate', 'secID', 'spret') / 100
        return spret.rename(index=lambda x: '-'.join([x[:4], x[4:-2], x[-2:]]))

    def get_factor_return_data(self):
        """提取风险因子收益"""
        return DataAPI.RMFactorRetDayGet(beginDate=self.dates[0].replace('-', ''),
                                         endDate=self.dates[-1].replace('-', ''),
                                         field=['secID', 'tradeDate'] + self.riskfactors,
                                         pandas="1").set_index('tradeDate').rename(
            index=lambda x: '-'.join([x[:4], x[4:-2], x[-2:]])).sort_index()

    def get_cum_spret(self):
        """计算每期的累计特质收益"""
        if not hasattr(self, 'cum_spret'):
            spret = self.get_spret_data()
            setattr(self, 'cum_spret', self.get_cumret(spret))
        return self.cum_spret

    def get_cum_factor_returns(self):
        """计算每期的累计风格因子收益"""
        if not hasattr(self, 'cum_fac_ret'):
            factor_ret = self.get_factor_return_data()
            cum_fac_ret = self.get_cumret(factor_ret)
            setattr(self, 'cum_fac_ret', self.get_cumret(cum_fac_ret))
        return self.cum_fac_ret

    def get_alpha_expo(self, weighted):
        """提取每期的alpha暴露"""
        return self.block_chain.apply(lambda x: x.get_alpha_expo(weighted=weighted))

    def __alpha_report(self, weighted):
        """计算每期的alpha因子收益"""
        excess_cum_spret = self._excess_spret_data()
        alpha_expo = self.get_alpha_expo(weighted=weighted)
        arr = []
        for day in self.trade_dates:
            tmp_x = alpha_expo.loc[day]
            tmp_y = excess_cum_spret.loc[day].reindex(tmp_x.index)
            arr.append(sm.OLS(tmp_y, tmp_x).fit())

        return pd.Series(arr, index=self.trade_dates).apply(lambda x: x.params)

    def predict_alpha_report(self):
        """计算每一期alpha因子收益"""
        return self.__alpha_report(weighted=False)

    def real_alpha_report(self):
        return self.__alpha_report(weighted=True)

    def alpha_portfolio_returns(self):
        """计算每一期alpha为组合带来的收益"""
        alpha_expo_sum = self.get_alpha_expo(weighted=True).apply(np.sum)
        alpha_returns = self.predict_alpha_report()

        alpha_cum_ret = pd.concat([alpha_expo_sum.mul(alpha_returns),
                                   self._excess_style_data()[self.risk_alpha_list]],
                                  axis=1)
        # return (alpha_cum_ret + 1).cumprod() - 1
        return alpha_cum_ret.cumsum()

    def predict_returns_report(self, target_factors):
        """预期收益特质收益的回归结果"""
        alpha_expo = self.get_alpha_expo(weighted=False)
        cum_spret = self.get_cum_spret()
        arr = []
        for day in self.trade_dates:
            tmp_x = alpha_expo.loc[day].dot(target_factors)
            tmp_y = cum_spret.loc[day].reindex(tmp_x.index)
            arr.append(sm.OLS(tmp_y, tmp_x).fit())
        return pd.Series(arr, index=self.trade_dates)

    def _excess_style_data(self):
        cum_fac_ret = self.get_cum_factor_returns()
        relative_expo = self.block_chain.apply(lambda x: x.get_relative_expo())
        return relative_expo.mul(cum_fac_ret)

    def excess_style_returns(self):
        """计算超额的风格收益"""
        # return (self._excess_style_data() + 1).prod() - 1
        return self._excess_style_data().sum()

    def _excess_spret_data(self):
        """计算超额的特质收益"""
        cum_spret = self.get_cum_spret()

        # w = self.block_chain.apply(lambda x: x.weights)
        bw = self.block_chain.apply(lambda x: x.bench_symbol_weight)

        return cum_spret.sub(cum_spret.mul(bw).sum(axis=1), axis=0)

    def excess_spret(self):
        excess_spret_data = self._excess_spret_data()
        w = self.block_chain.apply(lambda x: x.weights)
        return (excess_spret_data.mul(w).sum(axis=1) + 1).prod() - 1

    def compare_industry(self):
        """比较行业的权重"""
        ind_w = self.block_chain.apply(lambda x: x.industry_weights).mean().rename('porfolio')
        bench_ind_w = self.block_chain.apply(lambda x: x.bench_industry_weights).mean().rename('benchmark')

        return pd.concat([ind_w, bench_ind_w], axis=1).sort_values('porfolio', ascending=False)

    def summary(self):
        report = pd.Series()
        excess_style_returns = self.excess_style_returns()
        alpha_returns = self.excess_spret() + excess_style_returns[self.risk_alpha_list].sum()
        report['选股收益'] = alpha_returns
        report['风格收益'] = excess_style_returns[self.style_factors].drop(self.risk_alpha_list).sum()
        report['行业收益'] = excess_style_returns[self.industry_factors].sum()
        return report  # * 252 / self.bt.shape[0] * 100

    def plot_alpha(self):
        report = self.alpha_portfolio_returns()  # * 252 / self.bt.shape[0] * 100
        report.rename(
            columns=lambda x: (x in self.risk_alpha_list and x)
                              or 'alpha%.2d' % (report.columns.get_loc(x) + 1),
            inplace=True)
        report.plot(figsize=(15, 7))
        plt.show()
        report.iloc[-1].plot.bar(figsize=(15, 7), color='b')
        plt.show()
