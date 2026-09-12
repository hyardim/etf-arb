# NAV provenance

Every NAV series in this project comes from the issuer's own published daily
file. Most are downloaded by hand; SSGA's can be fetched directly. This file
records exactly where each one came from.

Provenance is a legitimate interview question: being able to say "NAV came
directly from the issuer's published daily file, and here is the URL and the
date I pulled it" closes off a whole line of scepticism about whether the
premium series is real or a data artefact.

## Why most of these are downloaded by hand

Scripted download was attempted during design. It works for one issuer of the
four tested and fails for the other three:

| Issuer | Result |
|---|---|
| iShares | `200` + `Content-Type: text/csv`, body is the **product page HTML** |
| SSGA | **works** — see correction below |
| Vanguard | `200`, client-rendered shell; data loads via JavaScript |
| Invesco | `406 Not Acceptable`, empty body |

> **Correction.** SSGA was originally recorded here as `301 → 404`. That was my
> error, not theirs: I guessed the filename `nav-history-spy.xlsx`. The real
> path is `navhist-us-en-spy.xlsx`, and it serves a genuine OOXML workbook over
> plain HTTP with no browser required. SSGA funds can therefore be fetched
> programmatically. iShares, Vanguard and Invesco were re-confirmed as blocked.

The iShares case is the instructive one. Its served HTML contains **no NAV
download link at all** — the control is rendered client-side. It does contain
a URL that looks like one, as the value of a hidden `videoSearchUrl` input in
the site search form. Fetching that URL, even with a browser user-agent,
session cookie and correct referer, returns the product page with
`Content-Type: text/csv`.

So a scraper gets a web page while every ordinary success signal — status
code, content type, file extension — reports success. `load_nav_file` in
[`src/data/loaders.py`](../src/data/loaders.py) therefore validates the bytes
themselves and rejects HTML outright.

The SSGA correction above is worth keeping visible rather than quietly editing
away: a negative result from a scraper is only ever evidence about the URL you
actually tried. "Blocked" and "I guessed the wrong filename" produce identical
symptoms, and the second is the more likely of the two.

## Download procedure

For each fund:

1. Open the fund's product page on the issuer site (URLs in the table below).
2. Go to the **Performance** section and find the historical NAV export
   (iShares labels this a downloadable NAV history; the exact wording moves
   around, so go by the section rather than a remembered button label).
3. **Set the date range to cover 2018-01-01 → today before exporting.** A date
   filter left at its default is the most common cause of a truncated file.
   The loader enforces a 1,000-row floor specifically to catch this.
4. Save the file the browser produces as `data/raw/nav/<TICKER>.<ext>`, keeping
   whatever extension it arrives with. Do not open and re-save it in Excel —
   that rewrites date formats and can silently coerce NAV to a rounded display
   value.
5. Verify it loads, then record the row count and range in the table below:

```bash
.venv/bin/python -c "
from src.data.loaders import load_nav_file
s = load_nav_file('data/raw/nav/HYG.xls')
print(len(s), s.index.min().date(), s.index.max().date())
"
```

Or run the whole pipeline for that fund:

```bash
.venv/bin/python scripts/summary.py HYG
```

If the loader rejects the file, the error message names the specific cause —
markup, unrecognised header, truncation, duplicate dates, or a column matched
to the wrong field. Fix the download rather than loosening the loader.

## Sources

Expect roughly 2,180 rows for a full 2018-01-01 → 2026-09 daily series.

| Ticker | Issuer | Product page | Date pulled | Rows | First | Last |
|---|---|---|---|---|---|---|
| `HYG` | iShares | https://www.ishares.com/us/products/239565/ — "Data Download" | 2026-09-12 | 4,893 | 2007-04-04 | 2026-09-11 |
| `IVV` | iShares | https://www.ishares.com/us/products/239726/ — "Data Download" | 2026-09-12 | 6,623 | 2000-05-15 | 2026-09-11 |
| `ITOT` | iShares | https://www.ishares.com/us/products/239724/ — "Data Download" | 2026-09-12 | 5,722 | 2004-01-20 | 2026-09-11 |
| `EWJ` | iShares | https://www.ishares.com/us/products/239665/ — "Data Download" | 2026-09-12 | 7,677 | 1996-03-12 | 2026-09-11 |
| `LQD` | iShares | https://www.ishares.com/us/products/239566/ — "Data Download" | 2026-09-12 | 6,077 | 2002-07-22 | 2026-09-11 |
| `EMB` | iShares | https://www.ishares.com/us/products/239572/ — "Data Download" | 2026-09-12 | 4,717 | 2007-12-17 | 2026-09-11 |
| `EEM` | iShares | https://www.ishares.com/us/products/239637/ — "Data Download" | 2026-09-12 | 5,928 | 2003-04-07 | 2026-09-11 |
| `SPY` | SSGA | [navhist-us-en-spy.xlsx](https://www.ssga.com/library-content/products/fund-data/etfs/us/navhist-us-en-spy.xlsx) (direct, scriptable) | 2026-09-12 | 5,733 | 2003-12-01 | 2026-09-10 |

> Every file was verified by CONTENT, not filename. Each downloaded NAV series
> was checked against its candidate ticker's actual market price before being
> renamed into place: all six iShares files matched at correlation >= 0.999 with
> level gaps under 0.25%. The browser names these exports after the fund
> ("iShares-Core-SP-500-ETF_fund.xls"), never the ticker, so a mix-up is easy
> and silent.

`HYG` was built end to end first, before any other fund was added, because
every data-quality problem shows up on the first fund and fixing it once is far
cheaper than fixing it eight times. That order is why the loader already
handled SpreadsheetML, invalid XML and header sniffing by the time the second
file arrived in a different format entirely.


## File formats encountered

Three so far, and the extension predicted the contents in none of them
reliably. `load_nav_file` detects format from leading bytes:

| Issuer | Extension | Actual format |
|---|---|---|
| iShares | `.xls` | SpreadsheetML 2003 (XML text), with invalid bare `&` in hyperlink attributes |
| SSGA | `.xlsx` | Genuine OOXML (ZIP, `PK\x03\x04`) |
| — | `.csv` | Has been served as an HTML product page |
