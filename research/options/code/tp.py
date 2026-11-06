import analyze as a, pandas as pd, sys
cur = sys.argv[1]
t = a.pick(a.load(cur))
t = t[t.dte_d.isin([2, 3]) & t.bucket.isin(["0-2%", "2-4%", "4-6%", "6-9%"])]
t = a.tp_mults(t, cur)
t.to_pickle(a.R / f"tp_{cur}.pkl")
rows = []
for (call, dte, b), s in t.groupby(["call", "dte_d", "bucket"], observed=True):
    for col in ["mult", "mult_tp2", "mult_tp3", "mult_tp5"]:
        s2 = s.assign(mult=s[col])
        row = dict(side="C" if call else "P", dte=dte, b=b, rule=col, ev=s2.mult.mean(), p2=(s2.mult >= 2).mean(), p5=(s2.mult >= 5).mean(), p10=(s2.mult >= 10).mean())
        for hz in (30, 90):
            r = a.roll(s2, call, dte, b, frac=1.0, horizon=hz)
            row[f"h2_{hz}"] = r.hit2.mean(); row[f"h5_{hz}"] = r.hit5.mean(); row[f"h10_{hz}"] = r.hit10.mean()
        rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(a.R / f"tp_{cur}.csv", index=False)
pd.set_option("display.width", 250)
print(df.round(3).to_string())
