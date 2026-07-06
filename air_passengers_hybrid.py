import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier

def run_hybrid():
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
    df = pd.read_csv(url, parse_dates=['Month'], index_col='Month')
    df.columns = ['pass']
    df.index.freq = 'MS'
    
    y = df['pass'].astype(float)
    ytr, yte = y.iloc[:-12], y.iloc[-12:]
    
    f = CalendarFourier(freq='Y', order=4)
    dp = DeterministicProcess(index=ytr.index, constant=True, order=1, additional_terms=[f], drop=True)
    xtr1 = dp.in_sample()
    
    m1 = LinearRegression(fit_intercept=False).fit(xtr1, ytr)
    yf = pd.Series(m1.predict(xtr1), index=xtr1.index)
    res = ytr - yf
    
    dfr = pd.DataFrame({'res': res})
    dfr['l1'], dfr['l12'] = dfr['res'].shift(1), dfr['res'].shift(12)
    dfr = dfr.dropna()
    
    m2 = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42).fit(dfr[['l1', 'l12']], dfr['res'])
    
    xte1 = dp.out_of_sample(steps=12)
    yp1 = pd.Series(m1.predict(xte1), index=xte1.index)
    
    xte2 = pd.DataFrame({'l1': [res.iloc[-1]]*12, 'l12': res.iloc[-12:].values}, index=yte.index)
    yp2 = pd.Series(m2.predict(xte2), index=yte.index)
    
    y_pred = yp1 + yp2
    print(f"MAE:  {mean_absolute_error(yte, y_pred):.1f}")
    print(f"RMSE: {mean_squared_error(yte, y_pred)**0.5:.1f}")

if __name__ == "__main__":
    run_hybrid()
