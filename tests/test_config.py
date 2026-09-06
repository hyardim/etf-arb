"""Tests for universe config loading.

The emphasis is on rejection: a malformed config must raise rather than
produce a half-valid Universe that fails confusingly further down the
pipeline.
"""

from __future__ import annotations

import textwrap

import pytest

from src.config import ConfigError, load_universe

# --------------------------------------------------------------------------
# The real config
# --------------------------------------------------------------------------


def test_real_config_loads():
    u = load_universe()
    assert len(u) == 8
    assert u.start == "2018-01-01"
    assert u.end is None
    assert u.end_or_today > u.start


def test_real_config_universe_design():
    """The design decisions in the plan must actually be encoded in the file."""
    u = load_universe()

    # SPY is the cross-issuer control, not a data point.
    assert u["SPY"].is_control
    assert u["SPY"].issuer == "ssga"
    assert len(u.primary) == 7

    # IVV and SPY track the same index -- that is what makes SPY a control.
    assert u["SPY"].sleeve == u["IVV"].sleeve

    # The time-zone cases are flagged, because they are analysed separately.
    assert u["EWJ"].foreign_session
    assert u["EEM"].foreign_session
    assert not u["HYG"].foreign_session

    # Credit and EM get wider inspection bands than equity.
    assert u["HYG"].outlier_band_bps > u["IVV"].outlier_band_bps


def test_lookup_and_unknown_ticker():
    u = load_universe()
    assert u["HYG"].sleeve == "credit_hy"
    with pytest.raises(KeyError, match="NOPE"):
        _ = u["NOPE"]


# --------------------------------------------------------------------------
# Rejection cases
# --------------------------------------------------------------------------

_VALID_FUND = """\
  - ticker: HYG
    name: iShares iBoxx $ High Yield Corporate Bond ETF
    sleeve: credit_hy
    issuer: ishares
    foreign_session: false
    outlier_band_bps: 800
    role: primary
"""


def _write(tmp_path, body: str):
    path = tmp_path / "universe.yaml"
    path.write_text(textwrap.dedent(body))
    return path


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_universe(tmp_path / "absent.yaml")


def test_missing_required_field(tmp_path):
    path = _write(
        tmp_path,
        """
        sample:
          start: "2018-01-01"
          end: null
        funds:
          - ticker: HYG
            name: HYG
            sleeve: credit_hy
        """,
    )
    with pytest.raises(ConfigError, match="missing fields"):
        load_universe(path)


def test_unexpected_field_is_rejected(tmp_path):
    """A misspelled key must fail, not be silently ignored -- otherwise the
    intended setting never takes effect and nothing says so."""
    path = _write(tmp_path, f"sample:\n  start: '2018-01-01'\nfunds:\n{_VALID_FUND}    forein_session: true\n")
    with pytest.raises(ConfigError, match="unexpected fields"):
        load_universe(path)


def test_duplicate_tickers_rejected(tmp_path):
    path = _write(tmp_path, f"sample:\n  start: '2018-01-01'\nfunds:\n{_VALID_FUND}{_VALID_FUND}")
    with pytest.raises(ConfigError, match="duplicate tickers"):
        load_universe(path)


def test_bad_role_rejected(tmp_path):
    body = _VALID_FUND.replace("role: primary", "role: benchmark")
    path = _write(tmp_path, f"sample:\n  start: '2018-01-01'\nfunds:\n{body}")
    with pytest.raises(ConfigError, match="role"):
        load_universe(path)


@pytest.mark.parametrize("band", ["0", "-100", "'wide'"])
def test_non_positive_band_rejected(tmp_path, band):
    body = _VALID_FUND.replace("outlier_band_bps: 800", f"outlier_band_bps: {band}")
    path = _write(tmp_path, f"sample:\n  start: '2018-01-01'\nfunds:\n{body}")
    with pytest.raises(ConfigError, match="outlier_band_bps"):
        load_universe(path)


def test_non_bool_foreign_session_rejected(tmp_path):
    """YAML will happily parse 'yes'/'no' strings; require a real bool so a
    quoted value cannot silently read as truthy."""
    body = _VALID_FUND.replace("foreign_session: false", "foreign_session: 'no'")
    path = _write(tmp_path, f"sample:\n  start: '2018-01-01'\nfunds:\n{body}")
    with pytest.raises(ConfigError, match="foreign_session"):
        load_universe(path)


def test_bad_date_rejected(tmp_path):
    path = _write(tmp_path, f"sample:\n  start: '01/01/2018'\nfunds:\n{_VALID_FUND}")
    with pytest.raises(ConfigError, match="ISO date"):
        load_universe(path)


def test_end_before_start_rejected(tmp_path):
    path = _write(
        tmp_path,
        f"sample:\n  start: '2018-01-01'\n  end: '2017-01-01'\nfunds:\n{_VALID_FUND}",
    )
    with pytest.raises(ConfigError, match="must be after"):
        load_universe(path)


def test_empty_funds_rejected(tmp_path):
    path = _write(tmp_path, "sample:\n  start: '2018-01-01'\nfunds: []\n")
    with pytest.raises(ConfigError, match="non-empty"):
        load_universe(path)
