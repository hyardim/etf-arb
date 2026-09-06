# ETF Premium / Discount & Arbitrage Dynamics

Measures the deviation between an ETF's market price and its net asset value (NAV), estimates how
fast that deviation decays, and compares decay speed across ETFs whose underlying baskets differ in
how easy they are to trade.

**Central hypothesis:** arbitrage efficiency is a function of the tradability of the underlying. An
S&P 500 ETF should snap back near-instantly; a high-yield credit or emerging market ETF should not.

> **Status:** in progress. Built incrementally — see the commit history.

## What this project does not claim

It does **not** model the creation/redemption arbitrage mechanism itself. That requires
primary-market data which is not observable outside an authorised participant: creation unit sizes,
AP fees, and settlement timing. This project measures the **observable residual** of that mechanism
— the price-to-NAV deviation and its decay — and is explicit about the difference.

## Planned deliverables

- Premium/discount time series for 8 ETFs spanning several asset classes
- Half-life of mean reversion per ETF, with confidence intervals and explicit resolution labels
- A cross-sectional result relating half-life to underlying liquidity
- Isolation of stale-NAV price discovery in time-zone-mismatched funds as distinct from mispricing

## Universe

| Ticker | Sleeve | Role | Issuer |
|---|---|---|---|
| `IVV` | S&P 500 | Fastest reversion | iShares |
| `ITOT` | US total market | Broad equity control | iShares |
| `EWJ` | Japan equity | Time-zone / stale-NAV case | iShares |
| `LQD` | Investment grade credit | OTC underlying | iShares |
| `HYG` | High yield credit | Price-leads-NAV case | iShares |
| `EMB` | EM USD debt | Hardest to arbitrage | iShares |
| `EEM` | EM equity | EM + time-zone compound | iShares |
| `SPY` | S&P 500 | Cross-issuer control | SSGA |

Seven of eight funds come from a single issuer so that fund structure and NAV convention are held
roughly constant and **underlying liquidity is the variable that moves**. `SPY` tracks the same
index as `IVV` from a different issuer under a different legal structure (UIT rather than open-end),
serving as a control on the NAV convention rather than as a data point.

Sample window: 2018-01-01 to present.

## Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Repository layout

```
configs/universe.yaml     tickers, sleeves, issuers, sample window
data/raw/nav/             hand-downloaded issuer NAV history (committed)
data/raw/prices/          yfinance cache (gitignored, regenerates)
data/processed/           aligned panel (gitignored, regenerates)
docs/NAV_SOURCES.md       per-fund provenance: URL, click path, date pulled
src/data/                 loaders, calendar alignment, quality checks
src/analysis/             premium construction, reversion, cross-section
tests/                    including estimator recovery against simulated OU
notebooks/                data quality and results
```

## Method notes

Sections referenced below are from the build guide this project follows.

- **Unadjusted prices only.** NAV as published is not dividend-adjusted, so comparing an adjusted
  price series against it manufactures a drift that grows with every dividend and masquerades as a
  persistent premium.
- **Strict inner join on date, never forward-fill NAV.** Forward-filling creates an artificial
  premium on the gap day and an artificial reversal the next — exactly the pattern a mean-reversion
  estimator is built to detect.
- **Estimator validated against simulated processes of known half-life** before being pointed at
  real data.
- **Half-lives are labelled, never bare.** Each is reported as `identified`, `censored_fast`
  (reversion completes faster than daily sampling can resolve), or `unit_root` (did not mean-revert
  over the sample).
