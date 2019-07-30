# -*- coding: UTF-8 -*
import pandas as pd


def get_buylist(bt, date):
    '''
    bt：回测报告，格式为pandas.DataFrame
    date：股票池持仓日期，%Y-%M-%D，文本格式
    '''

    Yesterday = date
    bt_last = pd.DataFrame(bt.query("tradeDate == '%s'" % Yesterday)['security_position'].values[0]).T
    buy_list = DataAPI.EquGet(equTypeCD=u"A", secID=bt_last.index,listStatusCD=u"", field=u"secID,ticker,secShortName,exchangeCD",pandas="1").set_index('secID').reindex(index=bt_last.index)
    buy_list = buy_list.replace({'XSHE': u'深交所',
                                 'XSHG': u'上交所'})
    buy_list.columns = ['证券代码','证券名称','交易市场']
    buy_list['数量'] = ''
    buy_list['市值'] = ''
    buy_list['市值权重'] = ''
    buy_list['设置比例'] = 100.0*bt_last['value']/bt_last['value'].sum()
    buy_list['指数权重'] = ''
    buy_list['停牌标志'] = ''
    buy_list['所属行业'] = ''
    buy_list['替代证券代码'] = '510500'
    buy_list['替代证券名称'] = '500ETF'
    buy_list['替代证券交易市场'] = '上交所'

    return buy_list

Yesterday = (pd.to_datetime('today', box=False).astype('datetime64[D]')-1).astype(str)
Yesterday = '2017-08-31'
buylist = get_buylist(bt, Yesterday)
# buylist.to_csv('buylist.csv', encoding='GB18030', index=False)
print len(buylist)
print buylist.head()