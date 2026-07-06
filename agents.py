import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from xgboost import XGBRegressor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

mom_feats = ['lag_1','lag_2','lag_3','lag_5','lag_10','rm_5','rm_21','rs_5','rsi_14','macd_s']
vol_feats = ['bb_w','rs_5','rs_21','vix_c','vix_5d']

class TrendAgt:
    def __init__(self, f_ord=4):
        self.f_ord = f_ord
        self.mod = LinearRegression(fit_intercept=False)
        self.dp = None
        
    def fit(self, tr):
        f = CalendarFourier(freq='YE', order=self.f_ord)
        self.dp = DeterministicProcess(index=tr.index, constant=True, order=1, additional_terms=[f], drop=True)
        self.mod.fit(self.dp.in_sample(), tr["log_ret"])
        
    def pred(self, te):
        return self.mod.predict(self.dp.out_of_sample(steps=len(te), forecast_index=te.index))

class MomAgt:
    def __init__(self):
        self.mod = XGBRegressor(n_estimators=200, lr=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
        
    def fit(self, tr):
        self.mod.fit(tr[[c for c in mom_feats if c in tr.columns]], tr["log_ret"])
        
    def pred(self, te):
        return self.mod.predict(te[[c for c in mom_feats if c in te.columns]])

class VolAgt:
    def __init__(self):
        self.mod = XGBRegressor(n_estimators=200, lr=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
        
    def fit(self, tr):
        self.mod.fit(tr[[c for c in vol_feats if c in tr.columns]], tr["log_ret"])
        
    def pred(self, te):
        return self.mod.predict(te[[c for c in vol_feats if c in te.columns]])

def mk_seq(s, w):
    return np.array([s[i-w:i] for i in range(w, len(s))], dtype=np.float32), np.array([s[i] for i in range(w, len(s))], dtype=np.float32)

class Net(nn.Module):
    def __init__(self, inp=1, hid=64, lays=2, drop=0.2):
        super().__init__()
        self.lstm = nn.LSTM(inp, hid, lays, batch_first=True, dropout=drop)
        self.out = nn.Linear(hid, 1)
        
    def forward(self, x):
        o, _ = self.lstm(x)
        return self.out(o[:, -1, :]).squeeze(-1)

class SeqAgt:
    def __init__(self, w=30, hid=64, ep=20, bs=64, lr=1e-3, pat=5):
        self.w, self.hid, self.ep, self.bs, self.lr, self.pat = w, hid, ep, bs, lr, pat
        self.scl = StandardScaler()
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def fit(self, tr):
        torch.manual_seed(0); np.random.seed(0)
        sr = self.scl.fit_transform(tr['log_ret'].values.reshape(-1, 1)).flatten()
        self.tail = tr['log_ret'].values[-self.w:]
        
        x, y = mk_seq(sr, self.w)
        xt, yt = torch.tensor(x).unsqueeze(-1).to(self.dev), torch.tensor(y).to(self.dev)
        
        vn = max(1, int(0.1 * len(xt)))
        xtr, xvl = xt[:-vn], xt[-vn:]
        ytr, yvl = yt[:-vn], yt[-vn:]
        ldr = DataLoader(TensorDataset(xtr, ytr), batch_size=self.bs, shuffle=False)
        
        self.net = Net(hid=self.hid).to(self.dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        crit = nn.MSELoss()
        
        bv, wt, bst = float('inf'), 0, None
        for _ in range(self.ep):
            self.net.train()
            for xb, yb in ldr:
                opt.zero_grad()
                crit(self.net(xb), yb).backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                vl = crit(self.net(xvl), yvl).item()
            if vl < bv - 1e-6:
                bv, wt, bst = vl, 0, {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                wt += 1
                if wt >= self.pat: break
        if bst: self.net.load_state_dict(bst)
        return self
        
    def pred(self, te):
        full = np.concatenate([self.tail, te['log_ret'].values])
        sf = self.scl.transform(full.reshape(-1, 1)).flatten()
        x, _ = mk_seq(sf, self.w)
        xt = torch.tensor(x).unsqueeze(-1).to(self.dev)
        
        self.net.eval()
        with torch.no_grad(): p = self.net(xt).cpu().numpy()
        return self.scl.inverse_transform(p.reshape(-1, 1)).flatten()
