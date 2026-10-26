"""E8 outcome mix and coin-flip benchmark: share of W bottoms / divergences / continuations, and the
neckline-first hit rate versus the random-walk rate for the same distances,
(entry - stop) / (neckline - stop)."""

import pandas as pd

import common as C
import e8_account as A


def main(K=3):
    D = A.prep(K)
    D = D[D.exit == "full"]
    rows = []
    for key, g in D.groupby(["tf", "side", "x", "stop_k"]):
        for st, gg in (("all", g), ("rule", g[g.rule])):
            for tag, pm in (("is", gg.t < C.T_OOS), ("oos", gg.t >= C.T_OOS)):
                x = gg[pm]
                hit = x.why == 3
                rw = ((x.px_in - x.stop) / (x.H1 - x.stop)).mean()
                rows.append(dict(zip(["tf", "side", "x", "stop"], key)) | {
                    "set": st, "per": tag, "n": len(x), "W_bottom": (hit & (x.broke_L0 == 0)).mean(),
                    "divergence": (hit & (x.broke_L0 == 1)).mean(), "continuation": x.why.isin([1, 5]).mean(),
                    "time": (x.why == 0).mean(), "hit_rate": hit.mean(), "coin_flip": rw,
                    "edge_pp": 100 * (hit.mean() - rw), "R": x.R.mean()})  # fmt: skip
    S = pd.DataFrame(rows)
    S.to_csv(C.RESULTS / "E8_wbottom" / f"outcomes_k{K}.csv", index=False, float_format="%.3f")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(S[(S.stop == 0.5)].round(3).to_string())


if __name__ == "__main__":
    main()
