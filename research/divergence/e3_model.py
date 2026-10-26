"""E3: predict double top (fail) vs breakout at the trigger, IS-trained, OOS-tested.

Geometry alone predicts a lot: a price far below H1 and near L1 touches L1 first more often. For
a driftless random walk P(fail) = 1 - recov. So models are compared against that baseline:
  geo   logistic on geometry only (recov, dist_atr, retr)
  full  logistic on all features
  gbm   gradient boosting on all features
Then the trade: EV = p * tp - (1 - p) * stop - cost with p from the model; enter only when EV is
above a threshold; realized net per trade on OOS by score quintile.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common as C
from e2_features import FEATS

GEO = ["recov", "dist_atr", "retr"]


def fit(kind, X, y):
    if kind == "gbm":
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                           min_samples_leaf=200, l2_regularization=1.0)  # fmt: skip
    else:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500))
    return m.fit(X, y)


def main():
    D = pd.read_pickle("data/research/divergence_e1.pkl").replace([np.inf, -np.inf], np.nan)
    D = D[D.label.isin([1, 2])].copy()
    D["fail"] = (D.label == 2).astype(int)
    D = D.dropna(subset=FEATS + ["net_b0.0"])
    rows, qrows = [], []
    preds = []
    for (tf, side), g in D.groupby(["tf", "side"]):
        tr, te = g[g.t < C.T_OOS], g[g.t >= C.T_OOS]
        base_te = 1 - te.recov.clip(0, 1)
        row = {"tf": tf, "side": side, "n_is": len(tr), "n_oos": len(te),
               "auc_randomwalk": roc_auc_score(te.fail, base_te),
               "fail_rate_oos": te.fail.mean(), "rw_expected_oos": base_te.mean()}  # fmt: skip
        for kind, cols in (("geo", GEO), ("full", FEATS), ("gbm", FEATS)):
            m = fit(kind, tr[cols], tr.fail)
            p_tr = m.predict_proba(tr[cols])[:, 1]
            p = m.predict_proba(te[cols])[:, 1]
            row[f"auc_{kind}_is"] = roc_auc_score(tr.fail, p_tr)
            row[f"auc_{kind}_oos"] = roc_auc_score(te.fail, p)
            if kind == "full":
                coef = m[-1].coef_[0]
                for f, c in zip(cols, coef):
                    row[f"coef_{f}"] = c
            # EV with the model's probability; payoff from the stop / target distances
            ev = p * te.tp0 - (1 - p) * te.stop0 - 2 * C.COST
            ev_tr = p_tr * tr.tp0 - (1 - p_tr) * tr.stop0 - 2 * C.COST
            qs = np.quantile(ev_tr, [0.5, 0.8, 0.9, 0.95])
            for qn, th in zip(("top50", "top20", "top10", "top5"), qs):
                for tag, d, e in (("is", tr, ev_tr), ("oos", te, ev)):
                    m_ = (e >= th).to_numpy() if hasattr(e, "to_numpy") else e >= th
                    x = d["net_b0.0"].to_numpy()[m_]
                    mu, t, _ = C.day_t(d.t.to_numpy()[m_], x)
                    qrows.append({"tf": tf, "side": side, "model": kind, "sel": qn, "period": tag,
                                  "n": int(m_.sum()), "win": (x > 0).mean(), "bp": 1e4 * mu, "t": t})  # fmt: skip
            if kind == "gbm":
                te2 = te.copy()
                te2["p"] = p
                te2["ev"] = ev
                tr2 = tr.copy()
                tr2["p"] = p_tr
                tr2["ev"] = ev_tr
                preds.append(pd.concat([tr2, te2]))
        rows.append(row)
    S = pd.DataFrame(rows)
    Q = pd.DataFrame(qrows)
    out = C.RESULTS / "E3_model"
    out.mkdir(parents=True, exist_ok=True)
    S.to_csv(out / "auc.csv", index=False, float_format="%.4f")
    Q.to_csv(out / "selection.csv", index=False, float_format="%.4f")
    pd.concat(preds).to_pickle("data/research/divergence_e3_preds.pkl")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    print(S[[c for c in S.columns if not c.startswith("coef")]].round(3).to_string())
    print(S[["tf", "side"] + [c for c in S.columns if c.startswith("coef")]].round(2).T.to_string())
    print(Q.pivot_table(index=["tf", "side", "model", "sel"], columns="period", values=["n", "win", "bp", "t"]).round(2).to_string())


if __name__ == "__main__":
    main()
