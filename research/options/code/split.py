import analyze as a, pandas as pd, sys
cur = sys.argv[1]
t0 = a.pick(a.load(cur))
rows = []
for ticks in [0, 1, 2]:
    tt, _ = a.single_stats(t0, ticks)
    for per, (lo, hi) in {"A21-23": ("2021", "2024"), "B24-26": ("2024", "2027")}.items():
        s = tt[(tt.day >= lo) & (tt.day < hi)]
        for call in [True, False]:
            for dte in [2, 3]:
                for b in ["2-4%", "4-6%", "6-9%"]:
                    m = s[(s.call == call) & (s.dte_d == dte) & (s.bucket == b)].mult
                    row = dict(ticks=ticks, per=per, side="C" if call else "P", dte=dte, b=b, n=len(m), ev=m.mean(), p2=(m >= 2).mean(), p5=(m >= 5).mean(), p10=(m >= 10).mean())
                    for f, hz in [(1.0, 30), (0.1, 90), (0.1, 365)]:
                        r = a.roll(s, call, dte, b, frac=f, horizon=hz)
                        row[f"f{f}_{hz}_h2"] = r.hit2.mean(); row[f"f{f}_{hz}_h10"] = r.hit10.mean()
                    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(a.R / f"split_{cur}.csv", index=False)
pd.set_option("display.width", 250)
print(df[df.ticks == 1].round(3).to_string())
print(df.groupby(["ticks", "per", "side"])[["ev", "p10", "f1.0_30_h10", "f0.1_365_h10"]].mean().round(3))
