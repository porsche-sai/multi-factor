for srisk_window, srisk_backlength, FCov_window, FCov_backlength in product(srisk_window_list, srisk_back_list, FCov_window_list, FCov_back_list):
    def initialize(context):  # 初始化策略运行环境
        pass

    def handle_data(context):  # 核心策略逻辑
        account = context.get_account('fantasy_account')
        today = cal.advanceDate(context.current_date, '-1B').strftime("%Y-%m-%d")
        print today, time.ctime()

        today_data = bt_model.section_series[today]  # 取得当日数据
        # symbols = today_data.symbols  # 当日成分股
        w0 = today_data.bench_symbol_weight.reindex(index=today_data.symbols).fillna(0)  # 以基准权重作为初始权重

        # 创建风险模型
        # 计算风险因子收益。使用直接赋值的方法
        # 计算相对基准的风险因子暴露
        xrm = today_data.relative_risk_expo(MY_RISK_FACTORS).reindex(today_data.symbols).fillna(0)
        # factor_returns = pd.Series(0, index=xrm.columns)
        FCov = today_data.get_FCov(MY_RISK_FACTORS, FCov_window, FCov_backlength) * 10000
        crm = xrm.dot(FCov).dot(xrm.T)
        # crm = pd.DataFrame(0, index=today_data.symbols, columns=today_data.symbols)
        R =  expo[today].dot(target_factors)  # + xrm.dot(factor_returns)

        exp_srisk = today_data.get_srisk(srisk_window, srisk_backlength) * 100
        V = 0.3 * (crm + np.diag(exp_srisk.pow(2).reindex(crm.index).fillna(0)))
        # V = 0.3 * crm

        # 计算当前股票池停牌股票权重
        positions = pd.Series(account.get_positions())
        positions = positions.map(lambda x: x.value)
        unhalt_position = account.get_positions(exclude_halt=True)
        halt_stock = set(set(positions.index) - set(unhalt_position))
        halt_stock =list(halt_stock)

        stockholding_weight = positions / account.portfolio_value
        w_last = stockholding_weight.reindex_like(w0).fillna(0)
        # w_halt = stockholding_weight.sum() - w_last.sum()
        halt_weight = stockholding_weight.loc[halt_stock]
        w_halt = halt_weight.sum()

        ExposureX1 = RMDataBase.RMExpoGet(risk_cols, today_data.symbols, tradeDate=today).set_index('Symbol')[risk_cols].reindex(today_data.symbols).fillna(0)
        # ExposureX = style_expo.loc[today].loc[today_data.symbols, risk_cols].fillna(0)

        if len(halt_stock):
            today_expo = RMDataBase.RMExpoGet(risk_cols, 
                                 tradeDate=today).set_index('Symbol')[risk_cols]
            halt_expo = fillna(today_expo,
                               tradeDate=today,
                               industry_mean=True).reindex(halt_stock).fillna(0)
            tmp_halt_sum = halt_weight.dot(halt_expo)
            halt_expo_sum = tmp_halt_sum[risk_cols]
            # halt_industry_weight = tmp_halt_sum.iloc[10:-1]
            halt_industry_weight = halt_weight.dot(get_industry(halt_stock, today).apply(lambda x: pd.Series(INDUSTRY_LIST == x,
                                                                                                  index=INDUSTRY_LIST,
                                                                                                  dtype=int)))

        else:
            halt_expo = 0
            halt_expo_sum = 0
            halt_industry_weight = 0
        # halt_expo_sum = halt_weight.dot(halt_Exposure)
        stock_common = set(today_data.symbols) & set(today_data.bench_symbols)  # 找到共同股票
        ExposureX_bench = today_data.bench_symbol_weight.loc[stock_common].dot(ExposureX1.loc[stock_common])
        ExposureX = ExposureX1 - ExposureX_bench - halt_expo_sum # 将风格因子基准中性化
        # halt_industry_weight = halt_weight.dot(get_industry(halt_stock).apply(lambda x: pd.Series(INDUSTRY_LIST == x,
        #                                                                                           index=INDUSTRY_LIST,
        #                                                                                           dtype=int)))

        adjust_bench_industry_weights = (today_data.bench_industry_weights - halt_industry_weight).clip_lower(0)

        threshold = (adjust_bench_industry_weights * 0.1).clip_lower(0.01)
        bound = []
        for i in range(len(w0)):
            if today_data.symbols_industry_dict.iloc[i] in forbidden:
                bound.append((0, 0))
            else:
                bound.append((0, 0.015))
        lbound, rbound = zip(*bound)

        # 市值中性右边界
        style_right = [0.01] * len(risk_cols)
        # 市值中性左边界
        style_left = [-0.01] * len(risk_cols)

        # G1 = xrm['SIZE']
        style_G = ExposureX[risk_cols].T

        # 行业中性右边界
        industry_right = adjust_bench_industry_weights + threshold
        # 行业中性左边界
        industry_left = adjust_bench_industry_weights - threshold
        industry_G = today_data.dummy.T

        # 右边界
        individual_right = np.array(rbound)
        individual_left = np.array(lbound)
        individual_G = np.eye(len(rbound))

        u = np.hstack([style_right, industry_right, individual_right, 1-w_halt])
        l = np.hstack([style_left, industry_left, individual_left, 1-w_halt])

        A = sparse.csc_matrix(np.vstack([style_G, industry_G, individual_G, np.ones_like(w0)]))

        P = sparse.csc_matrix(2*V.values)
        q = -R.values

        m = osqp.OSQP()
        m.setup(P=P, q=q, A=A, l=l, u=u, verbose=False, polish=1)
        result = m.solve()
        w_opt = result.x
        print result.info.status

        tao = np.zeros_like(w_opt)
        less = (w_opt<w_last).values
        greater = (w_opt>w_last).values
        tao[greater] = 5
        tao[less] = -10
        q_2 = (-R+tao).values

        individual_right[less & (w_last < individual_right).values] = w_last.values[less & (w_last<individual_right).values]
        individual_left[greater & (w_last > individual_left).values] = w_last.values[greater & (w_last > individual_left).values]

        u_2 = np.hstack([style_right, industry_right, individual_right, 1-w_halt])
        l_2 = np.hstack([style_left, industry_left, individual_left, 1-w_halt])

        m_2 = osqp.OSQP()
        m_2.setup(P=P, q=q_2, A=A, l=l_2, u=u_2, verbose=False, polish=1)
        # m.update(q=q, l=l, u=u)
        result = m_2.solve()
        w_opt = result.x
        w_opt[w_opt < 1e-4] = 0
        solution = pd.Series(w_opt, index=today_data.symbols)

        status = DataAPI.MktEqudAdjGet(tradeDate=today,
                                       secID=list(set(today_data.symbols+account.get_positions(exclude_halt=True).keys())),
                                       field=u"secID,isOpen",
                                       pandas="1").set_index('secID')
        solution = solution.loc[status.query("isOpen == 1").index].fillna(0).sort_values()
        for k, v in solution.to_dict().items():
            account.order_pct_to(k, v)
        # 生成策略对象
    strategy = quartz.TradingStrategy(initialize, handle_data)
    # -------------------策略定义结束--------------------   

    # 开始回测
    bt, perf = quartz.quick_backtest(sim_params, strategy, data=data)
    result = riskCoeffs(perf)
    result[u'对冲后'].to_csv('spret factor_ret parameter/sw%d sb%d Fw%d Fb %d.csv'% (srisk_window, srisk_backlength, FCov_window, FCov_backlength), encoding='gbk')