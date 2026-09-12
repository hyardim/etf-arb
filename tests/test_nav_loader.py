"""Tests for NAV file ingestion.

The NAV loader's job is to refuse bad input, so these tests are mostly about
proving refusals happen. The markup test in particular reproduces a real
observed failure rather than a hypothetical one -- see the module comment in
src/data/loaders.py.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.loaders import NavDataError, load_nav_file

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

# iShares exports carry a preamble of fund metadata before the real header,
# and its depth varies by issuer and drifts over time -- hence header sniffing
# rather than a hardcoded skiprows.
_PREAMBLE = (
    "iShares iBoxx $ High Yield Corporate Bond ETF\n"
    "Fund Holdings as of,\"Sep 04, 2026\"\n"
    "Inception Date,\"Apr 04, 2007\"\n"
    "\n"
)


def _nav_csv(tmp_path, rows: int = 1200, *, preamble: str = _PREAMBLE, name="HYG.csv"):
    dates = pd.bdate_range("2018-01-02", periods=rows)
    body = "\n".join(
        f'"{d:%b %d, %Y}","{80.0 + (i % 50) * 0.01:.2f}"' for i, d in enumerate(dates)
    )
    path = tmp_path / name
    path.write_text(f'{preamble}"As Of","NAV per Share"\n{body}\n')
    return path


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------


def test_loads_valid_file(tmp_path):
    nav = load_nav_file(_nav_csv(tmp_path))
    assert nav.name == "nav"
    assert nav.index.name == "date"
    assert len(nav) == 1200
    assert nav.index.is_monotonic_increasing
    assert nav.index.tz is None
    assert (nav > 0).all()


def test_skips_variable_preamble(tmp_path):
    """Header row is found by content, so preamble depth does not matter."""
    deep = "junk line\n" * 25
    nav = load_nav_file(_nav_csv(tmp_path, preamble=deep))
    assert len(nav) == 1200


def test_strips_currency_formatting(tmp_path):
    path = tmp_path / "HYG.csv"
    dates = pd.bdate_range("2018-01-02", periods=1100)
    body = "\n".join(f'"{d:%b %d, %Y}","$1,080.55"' for d in dates)
    path.write_text(f'"As Of","NAV per Share"\n{body}\n')
    assert load_nav_file(path).iloc[0] == pytest.approx(1080.55)


@pytest.mark.parametrize(
    "date_hdr,nav_hdr",
    [("As Of", "NAV per Share"), ("Date", "NAV"), ("Trade Date", "Net Asset Value")],
)
def test_accepts_known_header_aliases(tmp_path, date_hdr, nav_hdr):
    """Issuers word these differently and change them over time."""
    dates = pd.bdate_range("2018-01-02", periods=1100)
    body = "\n".join(f'"{d:%Y-%m-%d}","85.00"' for d in dates)
    path = tmp_path / "F.csv"
    path.write_text(f'"{date_hdr}","{nav_hdr}"\n{body}\n')
    assert len(load_nav_file(path)) == 1100


def test_trailing_disclaimer_rows_are_dropped(tmp_path):
    """Exports commonly end with unparseable footer text."""
    path = _nav_csv(tmp_path)
    with path.open("a") as fh:
        fh.write("\nPast performance is no guarantee of future results.\n")
    assert len(load_nav_file(path)) == 1200


# --------------------------------------------------------------------------
# The markup guard -- reproduces an observed failure
# --------------------------------------------------------------------------


def test_html_body_is_rejected(tmp_path):
    """A CSV-looking URL in the iShares page HTML answers with HTTP 200,
    Content-Type text/csv, and the product page in the body. Status, content
    type and file extension all report success, so the bytes are the only
    honest signal."""
    path = tmp_path / "HYG.csv"
    path.write_text(
        '<!DOCTYPE html>\n<html xmlns="http://www.w3.org/1999/xhtml" lang="en-US">\n'
        "<head>\n<title>iShares iBoxx $ High Yield Corporate Bond ETF | HYG</title>\n"
        "</head>\n<body>...</body></html>\n"
    )
    with pytest.raises(NavDataError, match="HTML page, not NAV data"):
        load_nav_file(path)


def test_html_with_leading_whitespace_rejected(tmp_path):
    path = tmp_path / "HYG.csv"
    path.write_text("\n\n   <!DOCTYPE html>\n<html></html>\n")
    with pytest.raises(NavDataError, match="HTML page, not NAV data"):
        load_nav_file(path)


def test_html_with_bom_rejected(tmp_path):
    path = tmp_path / "HYG.csv"
    path.write_bytes(b"\xef\xbb\xbf<!DOCTYPE html><html></html>")
    with pytest.raises(NavDataError, match="HTML page, not NAV data"):
        load_nav_file(path)


# --------------------------------------------------------------------------
# Structural refusals
# --------------------------------------------------------------------------


def test_missing_file_points_at_provenance_doc(tmp_path):
    with pytest.raises(NavDataError, match="NAV_SOURCES"):
        load_nav_file(tmp_path / "absent.csv")


def test_empty_file_rejected(tmp_path):
    path = tmp_path / "HYG.csv"
    path.write_text("")
    with pytest.raises(NavDataError, match="empty"):
        load_nav_file(path)


def test_unrecognised_header_rejected(tmp_path):
    """Wrong export type (holdings rather than NAV history) must not be
    guessed at positionally."""
    path = tmp_path / "HYG.csv"
    path.write_text('"Ticker","Name","Weight (%)"\n"T","AT&T","0.51"\n' * 400)
    with pytest.raises(NavDataError, match="no header row"):
        load_nav_file(path)


def test_truncated_file_rejected(tmp_path):
    """Usually a date filter left applied on the issuer page."""
    with pytest.raises(NavDataError, match="only 300 rows"):
        load_nav_file(_nav_csv(tmp_path, rows=300))


def test_duplicate_dates_rejected(tmp_path):
    """Overlapping exports must be reconciled by hand -- silently collapsing
    them makes the surviving value arbitrary."""
    path = _nav_csv(tmp_path, rows=1100)
    lines = path.read_text().splitlines()
    lines.append(lines[-1])
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(NavDataError, match="duplicate dates"):
        load_nav_file(path)


def test_non_positive_nav_rejected(tmp_path):
    path = _nav_csv(tmp_path, rows=1100)
    lines = path.read_text().splitlines()
    lines[6] = lines[6].rsplit(",", 1)[0] + ',"0.00"'
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(NavDataError, match="non-positive"):
        load_nav_file(path)


def test_wrong_column_magnitude_rejected(tmp_path):
    """Catches a NAV column matched to shares outstanding or net assets."""
    dates = pd.bdate_range("2018-01-02", periods=1100)
    body = "\n".join(f'"{d:%Y-%m-%d}","19850000000"' for d in dates)
    path = tmp_path / "HYG.csv"
    path.write_text(f'"As Of","NAV"\n{body}\n')
    with pytest.raises(NavDataError, match="outside"):
        load_nav_file(path)


def test_ambiguous_columns_rejected(tmp_path):
    dates = pd.bdate_range("2018-01-02", periods=1100)
    body = "\n".join(f'"{d:%Y-%m-%d}","85.00","85.01"' for d in dates)
    path = tmp_path / "HYG.csv"
    path.write_text(f'"As Of","NAV","NAV per Share"\n{body}\n')
    with pytest.raises(NavDataError, match="ambiguous"):
        load_nav_file(path)


def test_mostly_unparseable_rows_rejected(tmp_path):
    """A low parse ratio means the matched columns are the wrong ones."""
    dates = pd.bdate_range("2018-01-02", periods=1100)
    body = "\n".join(f'"{d:%Y-%m-%d}","n/a"' for d in dates)
    path = tmp_path / "HYG.csv"
    path.write_text(f'"As Of","NAV per Share"\n{body}\n')
    with pytest.raises(NavDataError, match="parsed to a"):
        load_nav_file(path)
