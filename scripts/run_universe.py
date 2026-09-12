#!/usr/bin/env python
"""Run the full pipeline across every fund in the universe.

    python scripts/run_universe.py

Writes data/processed/universe_results.csv and prints the ranked tables.
"""

import sys, warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import logging; logging.disable(logging.INFO)
import pandas as pd
from src.config import load_universe
from src.data.loaders import load_prices, load_nav_file
from src.data.align import align_price_nav
from src.analysis.premium import add_premium, describe_premium
from src.data.quality import check_quality
from src.analysis.reversion import estimate_reversion

NAV = ROOT / "data" / "raw" / "nav"
def nav_path(t):
    for ext in (".xls", ".xlsx", ".csv"):
        if (NAV / f"{t}{ext}").exists():
            return NAV / f"{t}{ext}"
    raise FileNotFoundError(t)

u = load_universe()
rows = []
for fund in u.funds:
    t = fund.ticker
    px = load_prices(t, u.start, u.end_or_today)
    nav = load_nav_file(nav_path(t))
    fr, al = align_price_nav(px, nav, ticker=t, start=u.start, end=u.end_or_today)
    fr = add_premium(fr)
    fr, q = check_quality(fr, ticker=t, band_bps=fund.outlier_band_bps)
    st = describe_premium(fr, ticker=t)
    rv = estimate_reversion(fr["premium_bps"], ticker=t)
    rows.append({
        "ticker": t, "sleeve": fund.sleeve, "role": fund.role,
        "foreign": fund.foreign_session, "n": st.n,
        "mean_bps": round(st.mean_bps, 2), "sd_bps": round(st.sd_bps, 2),
        "min_bps": round(st.min_bps, 1), "max_bps": round(st.max_bps, 1),
        "pct_prem": round(100*st.frac_premium, 1),
        "b": round(rv.fit.b, 4), "b_se": round(rv.fit.b_se, 4),
        "adf_p": f"{rv.adf_pvalue:.1e}", "p_b0": f"{rv.fit.b_pvalue:.1e}",
        "resolution": rv.resolution.value, "half_life": rv.display,
        "stale": q.n_stale_nav, "extreme": q.n_extreme_premium,
    })
df = pd.DataFrame(rows)
out = ROOT / "data" / "processed"
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "universe_results.csv", index=False)

pd.set_option("display.width", 250)
print("\n=== PREMIUM CHARACTERISTICS ===")
print(df[["ticker","sleeve","n","mean_bps","sd_bps","min_bps","max_bps","pct_prem"]].to_string(index=False))
print("\n=== REVERSION ===")
print(df[["ticker","sleeve","b","b_se","adf_p","p_b0","resolution","half_life"]].to_string(index=False))
print("\n=== QUALITY (flagged, retained) ===")
print(df[["ticker","n","stale","extreme"]].to_string(index=False))
