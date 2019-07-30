# coding: utf-8

import inspect
import threading
import gc
from copy import deepcopy
from lib.self_factors import get_factor_data
# from lib.interface import RMDataBase
from lib.tools import *
from lib.tools import _cross_diff


class AutoSave(threading.Thread):
    """
    使用多线程进行备份
    """

    def __init__(self, obj, attr, value):
        super(AutoSave, self).__init__()
        self.obj = obj
        self.attr = attr
        self.value = value
        self.thread_stop = False
        self.updated = False

    def run(self):
        path = '%s checkpoint' % self.obj.__class__
        while not self.thread_stop:
            if self.value is not None:
                if self.updated:
                    setattr(self.obj, self.attr, self.value)
                    self.obj.save_obj(path)
                    self.updated = False
                    # time.sleep(self.checkpoint)
        self.obj.save_obj(path)

    def update(self, value):
        self.value = value
        self.updated = True

    def stop(self):
        self.thread_stop = True


class Cache(object):
    """
    用于对象存取
    """

    @classmethod
    def from_cache(cls, path):
        """
        读取硬盘中的对象

        Parameters
        ----------
        path: str, 对象存放路径

        Returns
        -------
        object: 对象
        """
        return pd.read_pickle(path)

    def save_obj(self, path):
        """
        存放对象

        Parameters
        ----------
        path: str, 存放路径。

        Returns
        -------

        """
        pd.to_pickle(self, path)
        print '%s instance saved' % self.__class__

    def copy(self):
        return deepcopy(self)


class Section(Cache):
    """
    截面数据。获取某一日的数据
    """

    def __init__(self, symbols=None, factors=None, index_code=None, tradeDate=None, adjust_stock={}, benchmark=None):
        """

        Parameters
        ----------
        factors：list like, 因子列表
        index_code：str or list like, 股票池代码或列表
        tradeDate：str, yyyy-mm-dd, 交易日期
        adjust_stock：dict, 对股票池处理的函数
        benchmark：str, 基准代码
        """
        self.factors = list(factors)
        self.index_code = index_code
        self.tradeDate = tradeDate
        self.adjust_stock = adjust_stock
        self.benchmark = benchmark

        for k in adjust_stock:
            if inspect.isfunction(k):
                src = inspect.getsource(k)
                adjust_stock[k]['source'] = src
                adjust_stock[k.__name__] = adjust_stock.pop(k)
            else:
                assert 'source' in adjust_stock[k]
        # 获取股票池成分及权重
        stklist = symbols_process(symbols, index_code, tradeDate)

        for func, param in adjust_stock.iteritems():
            if 'date_required' in param:
                param[param['date_required']] = tradeDate
                param.pop('date_required')

            exec (param['source']) in globals()
            exec ("a = %s(%s, **%s)" % (func,
                                        stklist,
                                        {k: param[k] for k in param if k != 'source'})) in locals(), globals()
            stklist = a
            # stklist = func(stklist, **param)

        self.symbols = stklist

        # # 获取股票对应的行业
        self.symbols_industry_dict = get_industry(self.symbols, self.tradeDate)
        # 生成哑变量
        self.dummy = self.symbols_industry_dict.apply(lambda x: pd.Series(INDUSTRY_LIST == x,
                                                                          index=INDUSTRY_LIST,
                                                                          dtype=int))
        if benchmark is not None:
            self.set_benchmark(benchmark)

    def set_benchmark(self, benchmark=None, stk_weight=None, exclude_halt=False):
        """
        设置基准

        Parameters
        ----------
        benchmark：str, 基准代码

        Returns
        -------

        """
        if not isinstance(stk_weight, pd.Series):
            stk_weight = get_index_data3(benchmark, self.tradeDate, exclude_halt).rename(self.tradeDate)
        symbols = stk_weight.index
        weights = stk_weight.values

        symbols_industry_dict = get_industry(symbols, self.tradeDate)

        # 计算股票池行业权重
        tmp_industry = stk_weight.rename(index=symbols_industry_dict)
        industry_weights = (tmp_industry.groupby(level=0)
            .sum()
            .reindex(index=INDUSTRY_LIST)
            .fillna(0))
        self.benchmark = benchmark
        setattr(self, 'bench_symbols', symbols)
        setattr(self, 'bench_weights', weights)
        setattr(self, 'bench_symbol_weight', stk_weight)
        setattr(self, 'bench_symbols_industry_dict', symbols_industry_dict)
        setattr(self, 'bench_industry_weights', industry_weights)

    def _get_expo(self):
        # 获取因子对象并存为属性。
        # 若已有属性，则判断是否与因子列表一致。一致则直接返回，否则重构。
        if hasattr(self, '_expo'):
            if set(self._expo.columns) != set(self.factors):
                cross_factors, new_factors = _cross_diff(self.factors, self._expo.columns)  # 分出交集和差集
                cross_expo = self._expo[cross_factors]  # 对交集数据进行切片
                raw_expo = get_factor_data(self.symbols, self.tradeDate, ['secID'] + new_factors)
                # new_expo = (raw_expo.drop(STAY_ORIGIN, axis=1)
                #     .apply(regular_expo, args=(self.tradeDate,))
                #     .merge(raw_expo[STAY_ORIGIN],
                #            how='left',
                #            left_index=True,
                #            right_index=True)
                #     .reindex_like(raw_expo))
                # self._expo = pd.concat([cross_expo, new_expo], axis=1).reindex(columns=self.factors)  # 合并
                self._expo = pd.concat([cross_expo, raw_expo], axis=1).reindex(columns=self.factors)  # 合并

        else:
            raw_expo = get_factor_data(self.symbols, self.tradeDate, ['secID'] + self.factors)
            # _expo = (raw_expo.drop(STAY_ORIGIN, axis=1)
            #     .apply(regular_expo, args=(self.tradeDate,))
            #     .merge(raw_expo[STAY_ORIGIN],
            #            how='left',
            #            left_index=True,
            #            right_index=True)
            #     .reindex_like(raw_expo))
            _expo = raw_expo
            setattr(self, '_expo', _expo)
        return self._expo

    def get_expo(self, orth=True, keep=[], adjust_expo={}, industry_mean=False, cover=False):
        """
        获取因子截面数据

        Parameters
        ----------
        orth: bool, 是否正交化。默认 True
        keep: list like, 不参与正交的列

        Returns
        -------
        DataFrame: 因子数据
        """
        expo = self._get_expo()

        for func, params in deepcopy(adjust_expo).iteritems():
            if 'date_required' in params:
                params[params['date_required']] = self.tradeDate
                # params.pop('date_required')
                print params[params['date_required']]
            expo = func(fillna(expo, self.tradeDate, industry_mean),
                        **{k: params[k] for k in params if k != 'date_required'})

        if cover:
            setattr(self, '_expo', expo)
            setattr(self, '__cover_info__', [k.__name__ for k in adjust_expo.keys()])

        if orth:
            tmp = expo.drop(keep, axis=1)
            orthed = pd.DataFrame(schmidt_orth(tmp.values), index=tmp.index, columns=tmp.columns)
            return orthed.merge(expo[keep],
                                how='left',
                                left_index=True,
                                right_index=True).reindex_like(expo)
        return expo

    def refactor(self, factors):
        """
        对因子数据进行切片

        Parameters
        ----------
        factors: list, 因子列表

        Returns
        -------
        Section对象: 切片后对象
        """
        obj = self.copy()
        # 若有因子对象属性，则调用因子对象refactor方法进行切片
        # if hasattr(obj, '_expo'):
        #     obj._expo_obj = obj._expo_obj.refactor(factors)
        obj.factors = factors
        return obj

    def risk_model(self, beta_factors=BETA_FACTORS):
        """
        创建风险模型

        Parameters
        ----------
        beta_factors：风险因子

        Returns
        -------
        风险模型
        """
        kwargs = {'symbols': self.symbols,
                  'risk_factors': beta_factors,
                  'tradeDate': self.tradeDate}

        return RiskModel(**kwargs)

    def get_srisk(self, window, back_length):
        spret_data = RMDataBase.RMSpretGet(self.symbols, endDate=self.tradeDate)
        spret_cumret = ((spret_data + 1).rolling(window).apply(lambda x: x.prod()) - 1)

        return spret_cumret.apply(lambda x: x.dropna().iloc[:-(back_length * window):-window].std()) * np.sqrt(
            252.0 / window)

    def get_FCov(self, risk_factors, window, back_length):
        startday = cal.advanceDate(self.tradeDate, '-%dB' % (window * back_length)).strftime("%Y-%m-%d")
        factor_data = RMDataBase.RMFactorReturnsGet(risk_factors, startday, self.tradeDate)[risk_factors]
        factor_cumret = ((factor_data + 1).rolling(window).apply(lambda x: x.prod()) - 1) * (252.0 / window)

        return factor_cumret.iloc[:-(back_length * window):-window].cov()

    def relative_risk_expo(self, risk_factors):
        assert self.benchmark is not None
        risk_expo = (RMDataBase.RMExpoGet(risk_factors, self.symbols, tradeDate=self.tradeDate)
            .set_index('Symbol')
            .drop('Date', axis=1))
        bench_risk_expo = (RMDataBase.RMExpoGet(risk_factors, self.bench_symbols, tradeDate=self.tradeDate)
            .set_index('Symbol')
            .drop('Date', axis=1))
        return risk_expo - bench_risk_expo.mul(self.bench_symbol_weight, axis=0).sum()


class Base(Cache):
    """
    时间序列数据
    """

    def __init__(self, symbols=None, index_code=None, factors=None, dates=None, start=None, end=None, period=None,
                 tradeDate=None, adjust_stock={}, forward=0, benchmark=None):
        """

        Parameters
        ----------
        index_code：str or list like, 股票池代码或列表
        factors：list, 因子列表
        dates：list, 时间序列列表
        start：str, 起始日期
        end：str, 结束日期
        period：int, 间隔
        tradeDate：str, 交易日期
        adjust_stock：bool, 是否对股票池进行删选
        forward：int, 将日期序列向前移动天数
        benchmark：str, 基准代码
        """
        self.index_code = index_code
        self.dates = date_process(dates, start, end, period, tradeDate, forward)
        self.factors = list(factors)
        self.adjust_stock = adjust_stock
        self.benchmark = benchmark
        self.forward = forward

        for k in self.adjust_stock:
            if inspect.isfunction(k):
                src = inspect.getsource(k)
                self.adjust_stock[k]['source'] = src
                self.adjust_stock[k.__name__] = self.adjust_stock.pop(k)
            else:
                assert 'source' in self.adjust_stock[k]

        # 依据日期序列创建截面数据列表
        obj_arr = [Section(symbols=symbols,
                           factors=self.factors,
                           index_code=self.index_code,
                           tradeDate=day,
                           adjust_stock=deepcopy(adjust_stock),
                           benchmark=benchmark) for day in self.dates]
        # 将截面列表合并为pandas.Series
        self.section_series = pd.Series(obj_arr, index=self.dates)
        # # 股票池及权重序列
        # self.symbols_weights = self.section_series.apply(lambda x: x.symbol_weight).T
        # # 股票池对应行业序列
        # self.symbols_industry_dict = get_industry(self.symbols_weights.index)
        # # 行业权重序列
        # self.industry_weights = self.section_series.apply(lambda x: x.industry_weights).T
        # # 哑变量序列
        # self.dummy = self.symbols_industry_dict.apply(lambda x:
        #                                               pd.Series(INDUSTRY_LIST == x, index=INDUSTRY_LIST, dtype=int))

        print "Preprocess Done!"

    def set_benchmark(self, benchmark=None, stk_weight=None, exclude_halt=False):
        """
        设置基准

        Parameters
        ----------
        benchmark：str, 基准代码

        Returns
        -------

        """
        if benchmark is not None:
            self.benchmark = benchmark
            self.section_series.apply(lambda x: x.set_benchmark(benchmark=benchmark, exclude_halt=exclude_halt))
            return

        if isinstance(stk_weight, (list, tuple, np.ndarray)):
            self.benchmark = stk_weight
            assert len(stk_weight) == len(self.section_series)
            for i, symbol_weight in enumerate(stk_weight):
                self.section_series[i].set_benchmark(stk_weight=symbol_weight)
            return

        if isinstance(stk_weight, pd.Series):
            assert (stk_weight.index == self.section_series.index).all()
            return self.set_benchmark(stk_weight=stk_weight.values)

    def get_expo(self, orth=True, keep=[], adjust_expo={}, industry_mean=False, chunksize=np.inf, save_name=None,
                 cover=False):
        """
        获取因子序列数据

        Parameters
        ----------
        orth: bool, 是否正交化
        keep: list like, 不参与正交的列
        adjust_expo: dict, 对于因子暴露的处理方法
        chunksize: int, 备份间隔, 即每隔多少条数据存一次档。默认无限大
        save_name: None or str, 存档名称

        Returns
        -------
        Series: 因子序列数据
        """
        arr_num = np.floor_divide(len(self.section_series), chunksize)
        if arr_num > 1:
            section_array = np.array_split(self.section_series, arr_num)
            res = pd.Series()
            for sec in section_array:
                res = res.append(sec.apply(
                    lambda x: x.get_expo(orth=orth, keep=keep, adjust_expo=adjust_expo, industry_mean=industry_mean,
                                         cover=cover)))
                self.save_obj(save_name)
        else:
            res = self.section_series.apply(lambda x: x.get_expo(orth=orth, keep=keep, adjust_expo=adjust_expo,
                                                                 industry_mean=industry_mean, cover=cover))
        return res

    # def get_spret(self):
    #     """
    #     获取特质收益
    #
    #     Returns
    #     -------
    #     DataFrame: 特质收益数据框
    #     """
    #     startdate = self.dates[0]
    #     enddate = cal.advanceDate(self.dates[-1], '%dB' % self.forward).strftime("%Y-%m-%d")
    #     # symbols = self.section_series.apply(lambda x: x.symbol_weight).columns
    #     tmp = self.section_series.apply(lambda x: set(x.symbols))
    #     symbols = sorted(list(reduce(lambda x, y: x | y, tmp)))
    #
    #     # 优矿限制特质收益日期数为1019
    #     datelist = np.unique(date_process(start=startdate, end=enddate, period=500) + [enddate])
    #
    #     arr = []
    #     for i in range(len(datelist) - 1):
    #         data = DataAPI.RMSpecificRetDayGet(secID=symbols,
    #                                            beginDate=datelist[i].replace('-', ''),
    #                                            endDate=datelist[i + 1].replace('-', ''),
    #                                            field=['secID', 'tradeDate', 'spret'])
    #         data = data.pivot('tradeDate', 'secID', 'spret') / 100 + 1
    #         arr.append(data)
    #     spret = pd.concat(arr).drop_duplicates()
    #     return spret.rename(index=lambda x: '-'.join([x[:4], x[4:-2], x[-2:]]))

    def get_spret(self):
        startdate = self.dates[0]
        enddate = cal.advanceDate(self.dates[-1], '%dB' % self.forward).strftime("%Y-%m-%d")
        # symbols = self.section_series.apply(lambda x: x.symbol_weight).columns
        tmp = self.section_series.apply(lambda x: set(x.symbols))
        symbols = sorted(list(reduce(lambda x, y: x | y, tmp)))

        # sqlite数据库最大查询变量数为999
        groups = (len(symbols) / 900) + 1
        res = []
        for i in range(groups):
            res.append(RMDataBase.RMSpretGet(symbols[i * 900: (i + 1) * 900], beginDate=startdate, endDate=enddate))

        return pd.concat(res, axis=1)

    def factor_test(self, adjust_expo={}, t_threshold=15):
        """
        进行因子测试

        Parameters
        ----------
        adjust_expo: dict, 对因子暴露的处理方法
        t_threshold: float, t值阈值

        Returns
        -------
        FactorResults: 因子检验类
        """
        spret_data = self.get_spret()  # 取得特质收益
        # 对特质收益进行累乘，取得累计收益
        spret = ((spret_data + 1).rolling(self.forward).apply(lambda x: x.prod()) - 1) * 100 * 250 / self.forward

        # 特质收益起始日为因子暴露值对应的日期，结束日为换仓日
        # X和Y的一级索引为日期，二级索引为股票代码
        Y = spret.shift(-self.forward).stack().rename('Y').to_frame()
        expo = self.get_expo(orth=False, adjust_expo=adjust_expo)
        X = pd.concat(expo.values, keys=expo.index)
        cross_index = pd.MultiIndex.from_tuples(np.intersect1d(X.index, Y.index))
        X = X.reindex(index=cross_index).fillna(0)
        Y = Y.reindex(index=cross_index)
        # X = sm.add_constant(X)

        return FactorResults(X, Y, t_threshold, self.forward)

    def refactor(self, factors):
        """
        对因子数据进行重构

        Parameters
        ----------
        factors: list, 因子列表

        Returns
        -------
        Base对象: 重构后对象
        """
        obj = self.copy()
        obj.section_series = obj.section_series.apply(lambda x: x.refactor(factors))
        obj.factors = list(factors)
        return obj

    def redate(self, dates=None, tradeDate=None, start=None, end=None, period=None, forward=0):
        """
        对时间序列进行重构

        Parameters
        ----------
        dates: list like, 日期列表
        tradeDate: str, 单个日期
        start: str, 起始日期
        end: str, 结束日期
        period: int, 周期数
        forward: int, 向前偏移天数

        Returns
        -------
        Base对象: 重构后对象
        """
        obj = self.copy()
        datelist = date_process(dates=dates, tradeDate=tradeDate, start=start, end=end, period=period,
                                forward=forward)

        cross_date, diff_date = _cross_diff(datelist, self.dates)

        # 股票处理方法还原
        new_obj = Base(index_code=self.index_code, factors=self.factors,
                       dates=diff_date, adjust_stock=self.adjust_stock, benchmark=self.benchmark)

        obj.section_series = obj.section_series.append(new_obj.section_series).reindex(index=datelist)
        obj.dates = datelist
        return obj


class FactorResults(Cache):
    """
    因子检验类
    """

    def __init__(self, X, Y, t_threshold, forward):
        """

        Parameters
        ----------
        X: DataFrame, 历史因子暴露数据。行索引为日期（一级索引）和股票代码(二级索引), 列索引为因子名称
        Y: DataFrame, 历史特质收益数据。行索引为日期（一级索引）和股票代码(二级索引), 列索引为因子名称
        t_threshold: float, t值阈值
        """
        self.X = X
        self.Y = Y
        self.dates = self.X.index.levels[0]
        self.t_threshold = t_threshold
        self.forward = forward
        self.report = (X.apply(lambda x: sm.OLS(Y, x).fit())
            .apply(lambda x: pd.Series(np.hstack([x.params, x.tvalues, x.rsquared]),
                                       index=['fparams', 'tvalues', 'rsquared']))
            .sort_values('rsquared', ascending=False)
            .rename_axis(None))
        self.remark_cols = self.significance_filter(t_threshold=t_threshold)  # 挑选显著因子

    def significance_filter(self, t_threshold):
        """
        显著性筛选

        Parameters
        ----------
        t_threshold: float, t值阈值

        Returns
        -------
        list: 显著因子列表
        """
        remark_cols = self.report.query("abs(tvalues) > %f" % t_threshold).index
        return remark_cols

    def mono_filter(self, group_num=5, cols=None):
        """
        单调性检验

        Parameters
        ----------
        group_num: int, 分组数
        cols: list like or None, 供筛选的列名。若为None,则取显著因子

        Returns
        -------

        """
        # num_eps = 1.0 / group_num
        # XY = pd.concat([(self.X[cols].rank(pct=True)
        #                      .floordiv(num_eps) + 1).clip_upper(group_num),
        #                 self.Y], axis=1)
        # arr = []
        # for col in cols:
        #     arr.append(XY[[col, 'Y']].groupby(col).mean()
        #                .reindex(range(1, group_num + 1))
        #                .reset_index()
        #                .rename(columns={'Y': col + '_Y'}))
        #
        # cons = pd.concat(arr, axis=1)
        # # cons = cons.filter(regex='(?<!_X)$')  # 不以"_X"结尾的列
        # cons_diff = cons.diff().dropna()
        # monotony = (cons_diff > 0).all() | (cons_diff < 0).all()
        # res_df = (cons.loc[:, monotony]
        #           .filter(regex='_Y$')  # 以_Y结尾的列
        #           .rename(columns=lambda c: c[:-2])  # 去掉列名后缀
        #           .reindex(columns=self.report.index)
        #           .dropna(axis=1))
        # return res_df
        tmp = [self._mono_test(self.X[col], group_num) for col in cols]
        cons = pd.concat(tmp, axis=1)
        res = cons.diff().dropna()
        return cons.loc[:, (res >= 0).all() | (res <= 0).all()]

    def mono_max(self, least_groups=5, init_cols=None):
        group_num = least_groups
        mono_cols = self.mono_filter(group_num=group_num, cols=init_cols).columns
        res = pd.Series(group_num, index=mono_cols)
        while len(mono_cols) > 0:
            group_num += 1
            mono_cols = self.mono_filter(group_num=group_num, cols=mono_cols).columns
            res[mono_cols] = group_num
        return res

    def ic_filter(self, cols, threshold=2):
        # results = []
        # for day in self.dates:
        #     tx = self.X.loc[day]
        #     ty = self.Y.loc[day]
        #     results.append(tx.apply(lambda x: sm.OLS(ty, x).fit()))
        # df = pd.concat(results, axis=1).T
        # df.index = self.dates
        # params = df.applymap(lambda x: x.params[0])
        # ic = params.mean() / params.std() * np.sqrt(252.0 / self.forward)
        ic = self.X[cols].apply(self._ic_test)
        tmp = ic.abs().sort_values(ascending=False)
        return ic.reindex_like(tmp[tmp >= threshold])

    def corr_filter(self, corr_threshold=0.5, cols=None):
        x_corr = self.X[cols].corr()
        tmp = x_corr[(x_corr.abs() < corr_threshold) | (x_corr == 1)]
        col = tmp.columns[0]
        while True:
            t = tmp[col].dropna()
            tmp = tmp.reindex(index=t.index, columns=t.index)
            col_idx = tmp.columns.get_loc(col)
            if col_idx + 1 >= tmp.shape[1]:
                break
            col = tmp.columns[col_idx + 1]
        return tmp.columns

    def colinear_test(self, t_threshold=15, ic_threshold=2, group_num=5, cols=None):
        """
        共线性分析

        Parameters
        ----------
        t_threshold: float, t值阈值
        cols: None or list like, 需要进行共线性分析的因子名称。若为None,则使用经过单调性检验的数据。

        Returns
        -------
        list: 筛选后的因子列表
        """
        X = self.X[cols]
        Y = self.Y
        cols = self.report.loc[cols].sort_values('rsquared', ascending=False).index
        cols_selected = [cols[0]]
        cols_drop = []

        def cdt_col_filter():
            def resid_model(ax):
                ax = sm.OLS(ax, X[cols_selected]).fit().resid
                ax = sm.OLS(Y, pd.concat([X[cols_selected], ax], axis=1)).fit()
                return pd.Series(np.hstack([ax.rsquared_adj, abs(ax.tvalues.values[-1])]),
                                 index=['rsquared_adj', 'tvalues'])

            model = X.drop(cols_selected + cols_drop, axis=1).apply(resid_model).T
            cdt_col = model['rsquared_adj'].argmax()
            if model.loc[cdt_col, 'tvalues'] >= t_threshold:
                return cdt_col

        def mono_col_filter(X_col):
            tmp_mono = self._mono_test(X_col, group_num)
            res = tmp_mono.diff().dropna()
            return (res > 0).all() | (res < 0).all()

        for _ in cols[1:]:
            cdt_col = cdt_col_filter()
            if cdt_col is None:
                break
            cdt_resid = sm.OLS(X[cdt_col], X[cols_selected]).fit().resid
            new_X = (cdt_resid - cdt_resid.mean()) / cdt_resid.std()
            if not (mono_col_filter(new_X) and (np.abs(self._ic_test(new_X)) > ic_threshold)):
                cols_drop.append(cdt_col)
                continue
            cols_selected.append(cdt_col)
            X = X.assign(**{cdt_col: new_X})
        return X[cols_selected]

    def plot_factor_returns(self, cols=None, separate=False):
        res = pd.DataFrame()
        for day in self.dates[::self.forward]:
            if separate:
                res[day] = self.X.loc[day, cols].apply(lambda x: sm.OLS(self.Y.loc[day, 'Y'], x).fit().params[0])
            else:
                res[day] = sm.OLS(self.Y.loc[day, 'Y'], self.X.loc[day, cols]).fit().params
        cum_ret = (res.T / 100+1).cumprod() - 1
        for col in cols:
            cum_ret[col].plot(label=col)
            plt.legend()
            plt.show()

    def _mono_test(self, X_col, group_num):
        if X_col.name is None:
            X_col.name = 0
        num_eps = 1.0 / group_num
        XY = pd.concat([(X_col.rank(pct=True)
                         .floordiv(num_eps) + 1).clip_upper(group_num),
                        self.Y], axis=1)
        return XY.groupby(X_col.name)['Y'].mean().rename(X_col.name).rename_axis(None)

    def _ic_test(self, X_col):
        results = pd.Series()
        for day in self.dates:
            tx = X_col.loc[day]
            ty = self.Y.loc[day]
            results[day] = ty.corr(tx)
        return results.mean() / results.std() * np.sqrt(252.0 / self.forward)


class RiskModel(Cache):
    def __init__(self, symbols, risk_factors, tradeDate):
        self.symbols = symbols
        self.risk_factors = risk_factors
        self.tradeDate = tradeDate
        self.formatted_date = tradeDate.replace('-', '')

        self.FCov = DataAPI.RMCovarianceShortGet(tradeDate=self.formatted_date,
                                                 Factor=self.risk_factors,
                                                 field=['Factor'] + self.risk_factors,
                                                 pandas="1").set_index('Factor').fillna(0) \
            .reindex(index=self.risk_factors)

        self.risk_expo = DataAPI.RMExposureDayGet(secID=self.symbols,
                                                  tradeDate=self.formatted_date,
                                                  field=['secID'] + self.risk_factors,
                                                  pandas="1").set_index('secID').fillna(0)

        self.srisk = DataAPI.RMSriskShortGet(secID=self.symbols,
                                             tradeDate=self.formatted_date,
                                             field=u"secID,SRISK",
                                             pandas="1").set_index('secID').pow(2)['SRISK']

    def factor_returns(self, method='mean', **kwargs):
        if method == 'mean':
            beginDates = cal.advanceDate(self.tradeDate, "-%dB" % kwargs['offset']).strftime("%Y%m%d")
            endDates = self.formatted_date
            return DataAPI.RMFactorRetDayGet(beginDate=beginDates,
                                             endDate=endDates,
                                             field=['secID'] + self.risk_factors,
                                             pandas="1").mean().rename_axis(self.tradeDate) * 250 * 100
        if method == 'assign':
            return pd.Series(kwargs['alpha'], name=self.tradeDate).reindex(self.risk_factors).fillna(0)

        raise ValueError("method must be specified")

    def relative_expo(self, bench_weight):
        bench_obj = RiskModel(bench_weight.index, self.risk_factors, self.tradeDate)
        bench_expo = bench_obj.risk_expo
        return self.risk_expo - (bench_expo.T * bench_weight).sum(axis=1)


class RiskModelEvaluator(Cache):
    def __init__(self, individual_ret, expo, benchmark, dates=None):
        self.individual_ret = individual_ret * 100
        self.expo = expo
        self.benchmark = benchmark
        self.dates = dates

        spret = []
        factor_ret = []
        factor_cov = []
        self.symbols_weights = {}
        for day in self.dates:
            self.symbols_weights[day] = get_index_data2(self.benchmark, day)
            tmp_symbols = self.symbols_weights[day].index
            tmp_expo = self.expo.loc[day].reindex(tmp_symbols).dropna(how='all')
            tmp_ret = self.individual_ret.loc[day].reindex(tmp_expo.index)
            res = sm.OLS(tmp_ret, tmp_expo).fit()
            spret.append(res.resid)
            factor_ret.append(res.params)
            factor_cov.append(tmp_expo.mul(res.params).cov())
        self.spret = pd.concat(spret, keys=self.dates)
        self.factor_ret = pd.concat(factor_ret, axis=1, keys=self.dates).T
        self.factor_cov = pd.concat(factor_cov, keys=self.dates)

    def bias_test(self):
        bias = pd.Series()
        for i in range(len(self.dates) - 1):
            day_sigma = self.dates[i]
            day_return = self.dates[i + 1]
            symbols_weights = self.symbols_weights[day_return]
            symbols = symbols_weights.index.tolist()

            tmp_expo = self.expo.loc[day_sigma].loc[symbols]
            tmp_sret = self.spret.loc[day_sigma].loc[symbols]
            tmp_cov = self.factor_cov.loc[day_sigma]
            mid = tmp_expo.dot(tmp_cov).dot(tmp_expo.T) + np.diag(tmp_sret.pow(2))
            sigma = np.sqrt(symbols_weights.dot(mid.fillna(0)).dot(symbols_weights))
            if sigma in (np.inf, -np.inf) or np.isnan(sigma):
                continue
            r = self.individual_ret.loc[day_return, symbols].fillna(0).dot(symbols_weights)
            bias[day_return] = r / sigma
        result1 = np.sqrt(((bias - bias.mean()).pow(2).sum() / bias.size))
        result2 = (bias - bias.mean()).pow(2).rolling(window=12).mean().pow(0.5).mean()
        left = 1 - np.sqrt(2.0 / bias.size)
        right = 1 + np.sqrt(2.0 / bias.size)

        return {'result1': result1,
                'result2': result2,
                'left': left,
                'right': right,
                'conclusion1': left <= result1 <= right,
                'conclusion2': left <= result2 <= right}

