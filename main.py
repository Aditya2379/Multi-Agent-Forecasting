from sklearn.metrics import mean_absolute_error
from features import load_data, make_feats
from agents import TrendAgt, MomAgt, VolAgt, SeqAgt

def run():
    print("Loading data & making features...")
    df = make_feats(load_data())
    print(f"Data ready. Shape: {df.shape}")
    
    tr, te = df[df.index.year <= 2018], df[df.index.year == 2019]
    print(f"Train rows: {len(tr)}, Test rows: {len(te)}\n")
    
    agts = {
        "Trend": TrendAgt(),
        "Momentum": MomAgt(),
        "Volatility": VolAgt(),
        "Sequence": SeqAgt(ep=10)
    }
    
    for n, a in agts.items():
        a.fit(tr)
        p = a.pred(te)
        print(f"{n:12s} MAE: {mean_absolute_error(te['log_ret'], p):.6f}")

if __name__ == "__main__":
    run()
