import numpy as np
import pandas as pd
import yfinance as yf

def calc_rsi(s, p=14):
    d = s.diff()
    g, l = d.clip(lower=0), -d.clip(upper=0)
    ag = g.ewm(com=p-1, min_periods=p).mean()
    al = l.ewm(com=p-1, min_periods=p).mean()
    return 100 - (100 / (1 + ag / al))

def calc_macd(s, f=12, sl=26, sig=9):
    ef = s.ewm(span=f, adjust=False).mean()
    es = s.ewm(span=sl, adjust=False).mean()
    m = ef - es
    return m, m.ewm(span=sig, adjust=False).mean()

def calc_bb(s, w=20):
    mid, std = s.rolling(w).mean(), s.rolling(w).std()
    return ((mid + 2 * std) - (mid - 2 * std)) / mid

def get_lags(s, lags=(1, 2, 3, 5, 10)):
    return pd.DataFrame({f'lag_{l}': s.shift(l) for l in lags})

def get_rolls(s, wins=(5, 21)):
    d = {}
    for w in wins:
        d[f'rm_{w}'] = s.shift(1).rolling(w).mean()
        d[f'rs_{w}'] = s.shift(1).rolling(w).std()
    return pd.DataFrame(d)

def make_feats(df):
    c = df['Close'].shift(1)
    rsi = calc_rsi(c).rename('rsi_14')
    m_l, m_s = calc_macd(c)
    m_l, m_s = m_l.rename('macd_l'), m_s.rename('macd_s')
    bb = calc_bb(c).rename('bb_w')
    lags = get_lags(df['log_ret'])
    rolls = get_rolls(df['log_ret'])
    vix = df[['vix_c', 'vix_5d']].shift(1)
    
    return pd.concat([lags, rolls, rsi, m_l, m_s, bb, vix, df['log_ret']], axis=1).dropna()

def load_data(start='2010-01-01', end='2024-12-31'):
    spy = yf.Ticker('SPY').history(start=start, end=end)
    vix = yf.Ticker('^VIX').history(start=start, end=end)
    spy = spy[['Close']].copy()
    if spy.index.tz is not None: spy.index = spy.index.tz_localize(None)
    spy['log_ret'] = np.log(spy['Close'] / spy['Close'].shift(1))
    vix = vix[['Close']].copy()
    if vix.index.tz is not None: vix.index = vix.index.tz_localize(None)
    vix.columns = ['vix_c']
    vix['vix_5d'] = vix['vix_c'].pct_change(5)

    return spy.join(vix, how='inner').dropna()
