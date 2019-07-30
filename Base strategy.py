# -*- coding: UTF-8 -*
# Copyright 2017/11/27. All Rights Reserved
# Author: sai
# Base strategy.py 2017/11/27 14:00
from cvxopt import matrix, solvers
import time

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

forbidden = []

riskfactor = ["EARNYILD", "BTOP", "LIQUIDTY", "MOMENTUM", "GROWTH", "BETA", "SIZE", "RESVOL", "LEVERAGE",
              "SIZENL", 'Bank', 'RealEstate', 'Health', 'Transportation', 'Mining', 'NonFerMetal',
              'HouseApp', 'LeiService', 'MachiEquip', 'BuildDeco', 'CommeTrade', 'CONMAT', 'Auto',
              'Textile', 'FoodBever', 'Electronics', 'Computer', 'LightIndus', 'Utilities', 'Telecom',
              'AgriForest', 'CHEM', 'Media', 'IronSteel', 'NonBankFinan', 'ELECEQP', 'AERODEF', 'Conglomerates']
alpha_dict = {'EARNYILD': 5.8,
              'BTOP': 5.2}

accounts = {
    'fantasy_account': AccountConfig(account_type='security',
                                     capital_base=100000000,
                                     commission=Commission(buycost=0.0001, sellcost=0.0011),
                                     slippage=Slippage(value=0.0005, unit='perValue'))
}


def initialize(context):  # 初始化策略运行环境
    pass


def handle_data(context):  # 核心策略逻辑
    account = context.get_account('fantasy_account')
    today = cal.advanceDate(context.current_date, '-1B').strftime("%Y-%m-%d")
    print today, time.ctime()

    today_data = new_bt_model.section_series[today]  # 取得当日数据
    # symbols = today_data.symbols  # 当日成分股
    w0 = today_data.bench_symbol_weight.reindex(index=today_data.symbols).fillna(0)  # 以基准权重作为初始权重

    # 创建风险模型
    risk_obj = today_data.risk_model(riskfactor)
    # 计算风险因子收益。使用直接赋值的方法
    factor_returns = risk_obj.factor_returns(method='assign', alpha=alpha_dict)
    # 计算相对基准的风险因子暴露
    xrm = risk_obj.relative_expo(bench_weight=today_data.bench_symbol_weight)
    crm = xrm.dot(risk_obj.FCov).dot(xrm.T)
    R = orth_expo[today].dot(new_target_factors) + xrm.dot(factor_returns)
    V = 0.3 * (crm + np.diag(risk_obj.srisk.reindex(index=crm.index)))

    field = "secID,EARNYILD,BTOP,MOMENTUM,RESVOL,GROWTH,BETA,LEVERAGE,LIQUIDTY,SIZENL,SIZE"
    ExposureX = DataAPI.RMExposureDayGet(secID=today_data.symbols,
                                         tradeDate=Date.strptime(today, '%Y-%m-%d').strftime('%Y%m%d'),
                                         field=field, pandas="1").set_index(['secID']).reindex(
        index=today_data.symbols).fillna(0.0)
    stock_common = filter(lambda x: x in today_data.symbols, today_data.bench_symbols)  # 找到共同股票
    ExposureX_bench = np.dot(today_data.bench_symbol_weight.loc[stock_common].T, np.array(ExposureX.loc[stock_common]))
    ExposureX = np.array(ExposureX) - ExposureX_bench  # 将风格因子基准中性化
    # 计算当前股票池停牌股票权重
    positions = pd.Series(account.get_positions())
    positions = positions.map(lambda x: x.value)
    stockholding_weight = positions / account.portfolio_value
    w_last = stockholding_weight.reindex_like(w0).fillna(0)
    w_halt = stockholding_weight.sum() - w_last.sum()

    threshold = (today_data.bench_industry_weights * 0.05).clip_lower(0.005)
    bound = []
    for i in range(len(w0)):
        if today_data.symbols_industry_dict.iloc[i] in forbidden:
            bound.append((0, 0))
        else:
            bound.append((0, 0.01))
    lbound, rbound = zip(*bound)

    # 市值中性右边界
    h1 = 0.01
    # G1 = xrm['SIZE']
    G1 = ExposureX[:, 9]
    # 市值中性左边界
    h2 = 0.01
    # G2 = -xrm['SIZE']
    G2 = -ExposureX[:, 9]
    # 行业中性右边界
    h3 = threshold + today_data.bench_industry_weights
    G3 = today_data.dummy.T
    # 行业中性左边界
    h4 = threshold - today_data.bench_industry_weights
    G4 = -today_data.dummy.T

    # 右边界
    h5 = np.array(rbound)
    G5 = np.eye(len(rbound))
    # 左边界
    h6 = np.array(lbound)
    G6 = -np.eye(len(lbound))

    h = np.hstack([h1, h2, h3, h4, h5, h6])
    G = np.vstack([G1, G2, G3, G4, G5, G6])

    # 换手率约束
    V_exchange = np.ones_like(V) * 560
    A_exchange = w_last.values * 560

    # 和约束
    A = np.ones_like(w0)
    b = min(w0.sum(), 1 - w_halt)

    P = matrix(2 * (V.values + V_exchange))
    q = matrix(-R.values - 2 * A_exchange)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A).T
    b = matrix(float(b))
    result = solvers.qp(P, q, G, h, A, b)
    w_opt = np.array(result['x'].T)[0].round(5)
    w_opt[w_opt < 1e-4] = 0
    solution = pd.Series(w_opt, index=today_data.symbols)

    status = DataAPI.MktEqudAdjGet(tradeDate=today,
                                   secID=today_data.symbols + account.security_position.keys(),
                                   field=u"secID,isOpen",
                                   pandas="1").set_index(['secID'])
    solution = solution.loc[status.query("isOpen == 1").index].fillna(0).sort_values()
    for k, v in solution.to_dict().items():
        account.order_pct_to(k, v)

if __name__ == '__main__':
    pass
