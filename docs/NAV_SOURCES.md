# NAV provenance

Every NAV series in this project comes from the issuer's own published daily
file, downloaded by hand. This file records exactly where each one came from.

Provenance is a legitimate interview question: being able to say "NAV came
directly from the issuer's published daily file, and here is the URL and the
date I pulled it" closes off a whole line of scepticism about whether the
premium series is real or a data artefact.

## Why these are downloaded by hand

Scripted download was attempted during design and rejected. It does not work
over plain HTTP for any of the four issuers tested:

| Issuer | Result |
|---|---|
| iShares | `200` + `Content-Type: text/csv`, body is the **product page HTML** |
| SSGA | `301` → `404` HTML |
| Vanguard | `200`, client-rendered shell; data loads via JavaScript |
| Invesco | `406 Not Acceptable`, empty body |

The iShares case is the instructive one. Its served HTML contains **no NAV
download link at all** — the control is rendered client-side. It does contain
a URL that looks like one, as the value of a hidden `videoSearchUrl` input in
the site search form. Fetching that URL, even with a browser user-agent,
session cookie and correct referer, returns the product page with
`Content-Type: text/csv`.

So a scraper gets a web page while every ordinary success signal — status
code, content type, file extension — reports success. `load_nav_csv` in
[`src/data/loaders.py`](../src/data/loaders.py) therefore validates the bytes
themselves and rejects anything whose first non-whitespace character is `<`.

## Download procedure

For each fund:

1. Open the fund's product page on the issuer site (URLs in the table below).
2. Go to the **Performance** section and find the historical NAV export
   (iShares labels this a downloadable NAV history; the exact wording moves
   around, so go by the section rather than a remembered button label).
3. **Set the date range to cover 2018-01-01 → today before exporting.** A date
   filter left at its default is the most common cause of a truncated file.
   The loader enforces a 1,000-row floor specifically to catch this.
4. Save the file the browser produces as `data/raw/nav/<TICKER>.csv`.
   Do not open and re-save it in Excel — that rewrites date formats and can
   silently coerce NAV to a rounded display value.
5. Verify it loads, then record the row count and range in the table below:

```bash
.venv/bin/python -c "
from src.data.loaders import load_nav_csv
s = load_nav_csv('data/raw/nav/HYG.csv')
print(len(s), s.index.min().date(), s.index.max().date())
"
```

If the loader rejects the file, the error message names the specific cause —
markup, unrecognised header, truncation, duplicate dates, or a column matched
to the wrong field. Fix the download rather than loosening the loader.

## Sources

Expect roughly 2,180 rows for a full 2018-01-01 → 2026-09 daily series.

| Ticker | Issuer | Product page | Date pulled | Rows | First | Last |
|---|---|---|---|---|---|---|
| `HYG` | iShares | https://www.ishares.com/us/products/239565/ | _pending_ | | | |
| `IVV` | iShares | https://www.ishares.com/us/products/239726/ | _pending_ | | | |
| `ITOT` | iShares | https://www.ishares.com/us/products/239724/ | _pending_ | | | |
| `EWJ` | iShares | https://www.ishares.com/us/products/239665/ | _pending_ | | | |
| `LQD` | iShares | https://www.ishares.com/us/products/239566/ | _pending_ | | | |
| `EMB` | iShares | https://www.ishares.com/us/products/239572/ | _pending_ | | | |
| `EEM` | iShares | https://www.ishares.com/us/products/239637/ | _pending_ | | | |
| `SPY` | SSGA | https://www.ssga.com/us/en/intermediary/etfs/spdr-sp-500-etf-trust-spy | _pending_ | | | |

> Product-page IDs above are recorded from the iShares URL scheme and should be
> confirmed on arrival — verify the page you land on names the expected fund
> before exporting.

**Start with `HYG` only.** The build works one fund end to end through the
whole pipeline before the other seven are added, because every data-quality
problem shows up on the first fund and fixing it once is far cheaper than
fixing it eight times.
