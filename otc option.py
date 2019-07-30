# -*- coding: UTF-8 -*


def getHistVolatilityC2C(secID, beginDate, endDate, kind="Stock"):
    cal = Calendar('China.SSE')
    spotBeginDate = cal.advanceDate(beginDate, '-520B', BizDayConvention.Preceding)
    begin = spotBeginDate.toISO().replace('-', '')
    end = endDate.toISO().replace('-', '')

    fields = ['tradeDate', 'preClosePrice', 'closePrice']
    if kind == 'Index':
        fields = ['tradeDate', 'preCloseIndex', 'closeIndex']
        security = DataAPI.MktIdxdGet(indexID=secID, beginDate=begin, endDate=end, field=fields).rename(
            columns={'closeIndex': 'closePrice', 'preCloseIndex': 'preClosePrice'})
    elif kind == 'Fund':
        security = DataAPI.MktFunddGet(secID=secID, beginDate=begin, endDate=end, field=fields)
    else:
        security = DataAPI.MktEqudAdjGet(secID=secID, beginDate=begin, endDate=end, field=fields)

    security['dailyReturn'] = security['closePrice'] / security['preClosePrice']  # 日回报率
    security['u'] = np.log(security['dailyReturn'])  # u2为复利形式的日回报率
    security['tradeDate'] = pd.to_datetime(security['tradeDate'])

    periods = {'hv1W': 5, 'hv2W': 10, 'hv1M': 21, 'hv2M': 41, 'hv3M': 62, 'hv4M': 83,
               'hv5M': 104, 'hv6M': 124, 'hv9M': 186, 'hv1Y': 249}
    # 利用方差模型计算波动率
    for prd in periods.keys():
        tmp = pd.rolling_std(security['u'], window=periods[prd]) * np.sqrt(249.0)
        security[prd] = np.round(tmp, 4)

    security = security[security.tradeDate >= beginDate.toISO()]
    security = security.set_index('tradeDate')
    security = security.fillna(0.0)
    return security


def extremeValueVariance(secID, end=Date.todaysDate() - 1, kind='Stock'):
    cal = Calendar('China.SSE')
    start = cal.advanceDate(end, '-250B', BizDayConvention.Preceding)

    if kind == 'Index':
        fields = ['tradeDate', 'preCloseIndex', 'highestIndex', 'lowestIndex', 'closeIndex']
        stock = DataAPI.MktIdxdGet(indexID=secID, beginDate=start.toISO().replace('-', ''),
                                   endDate=end.toISO().replace('-', ''), field=fields).rename(
            columns={'closeIndex': 'closePrice', 'preCloseIndex': 'preClosePrice', 'highestIndex': 'highestPrice',
                     'lowestIndex': 'lowestPrice'})
    elif kind == 'Fund':
        fields = ['tradeDate', 'preClosePrice', 'highestPrice', 'lowestPrice', 'closePrice']
        stock = DataAPI.MktFunddGet(secID=secID, beginDate=start.toISO().replace('-', ''),
                                    endDate=end.toISO().replace('-', ''), field=fields)
    else:
        fields = ['tradeDate', 'preClosePrice', 'highestPrice', 'lowestPrice', 'closePrice']
        stock = DataAPI.MktEqudAdjGet(secID=secID, beginDate=start.toISO().replace('-', ''),
                                      endDate=end.toISO().replace('-', ''), field=fields)

    stock['tradeDate'] = pd.to_datetime(stock['tradeDate'])
    stock = stock.set_index('tradeDate')
    stock['H2L'] = np.log(stock['highestPrice'] / stock['lowestPrice'])  # 日回报率
    stock['H2L'] = stock['H2L'].fillna(0.0)
    periods = {'hv2W': 10, 'hv1M': 21, 'hv2M': 41, 'hv3M': 62, 'hv6M': 126, 'hv9M': 186, 'hv1Y': 249}
    H2L = pd.DataFrame()
    for prd in ['hv2W', 'hv1M', 'hv2M', 'hv3M', 'hv6M', 'hv9M', 'hv1Y']:
        p = pd.DataFrame()
        p['H2L'] = pd.rolling_mean(stock['H2L'], window=periods[prd]).tail(1) * math.sqrt(249.0)
        p.index = [prd]
        p = p.round(4)
        H2L = pd.concat([H2L, p], axis=0)
    H2L = H2L.transpose() * 100
    return H2L


def vols_calc(secID, end=Date.todaysDate() - 1, kind='Stock'):
    cal = Calendar('China.SSE')
    start = cal.advanceDate(end, '-400B', BizDayConvention.Preceding)
    hist_c2c = getHistVolatilityC2C(secID, start, end, kind)
    periods = {'hv2W': 10, 'hv1M': 21, 'hv2M': 41, 'hv3M': 62, 'hv6M': 126, 'hv9M': 186, 'hv1Y': 249}
    vols = pd.DataFrame()
    window = 400  # 19个月
    for prd in ['hv2W', 'hv1M', 'hv2M', 'hv3M', 'hv6M', 'hv9M', 'hv1Y']:
        vol = pd.DataFrame()
        vol['last'] = hist_c2c[prd].tail(1)
        vol['max'] = pd.rolling_max(hist_c2c[prd], window=window - periods[prd]).tail(1)
        mean = pd.rolling_mean(hist_c2c[prd], window=window - periods[prd]).tail(1)
        std = pd.rolling_std(hist_c2c[prd], window=window - periods[prd]).tail(1)
        std.values[0] = max(std.values[0], 0.02)  # 设置最小标准差为2%
        vol['mean+2std'] = mean + 2 * std
        vol['mean+1std'] = mean + std
        vol['mean'] = mean
        vol['mean-1std'] = mean - std
        vol['mean-2std'] = mean - 2 * std
        vol['min'] = pd.rolling_min(hist_c2c[prd], window=window - periods[prd]).tail(1)
        vol.index = [prd]
        vols = pd.concat([vols, vol], axis=0)
    vols = vols.transpose() * 100
    vols = vols.round(2)
    return vols
