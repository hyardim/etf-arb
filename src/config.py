"""Universe configuration loading and validation.

The universe definition is data, not code, so that the analysis design is
visible in one file rather than scattered through the pipeline. This module
loads it and refuses to return a half-valid object -- a typo in a ticker or a
missing sleeve should fail here, loudly, rather than surface later as an empty
price series or a mislabelled cross-sectional row.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "universe.yaml"

VALID_ROLES = frozenset({"primary", "control"})

_REQUIRED_FUND_FIELDS = (
    "ticker",
    "name",
    "sleeve",
    "issuer",
    "foreign_session",
    "outlier_band_bps",
    "role",
)


class ConfigError(ValueError):
    """Raised when the universe config is malformed."""


@dataclass(frozen=True)
class Fund:
    """One ETF in the universe."""

    ticker: str
    name: str
    sleeve: str
    issuer: str
    foreign_session: bool
    """True when the underlying basket trades in a session that does not
    overlap the US listing, making NAV a stale-close observation."""
    outlier_band_bps: float
    """Absolute premium beyond which an observation is FLAGGED for hand
    inspection. Flagging is not deletion -- see src/data/quality.py."""
    role: str

    @property
    def is_control(self) -> bool:
        return self.role == "control"


@dataclass(frozen=True)
class Universe:
    """The full set of funds plus the shared sample window."""

    start: str
    end: str | None
    funds: tuple[Fund, ...]

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(f.ticker for f in self.funds)

    @property
    def primary(self) -> tuple[Fund, ...]:
        return tuple(f for f in self.funds if not f.is_control)

    @property
    def end_or_today(self) -> str:
        """Resolved end date; the config stores null to mean 'today'."""
        return self.end or dt.date.today().isoformat()

    def __getitem__(self, ticker: str) -> Fund:
        for fund in self.funds:
            if fund.ticker == ticker:
                return fund
        raise KeyError(f"{ticker!r} is not in the universe: {list(self.tickers)}")

    def __len__(self) -> int:
        return len(self.funds)


def _parse_date(value: str, field: str) -> str:
    try:
        return dt.date.fromisoformat(str(value)).isoformat()
    except ValueError as exc:
        raise ConfigError(f"sample.{field} is not an ISO date: {value!r}") from exc


def _build_fund(raw: dict, index: int) -> Fund:
    missing = [f for f in _REQUIRED_FUND_FIELDS if f not in raw]
    if missing:
        label = raw.get("ticker", f"index {index}")
        raise ConfigError(f"fund {label} is missing fields: {missing}")

    unexpected = set(raw) - set(_REQUIRED_FUND_FIELDS)
    if unexpected:
        # Fail rather than ignore: a silently dropped key is usually a typo in
        # a field name, and ignoring it means the intended setting never
        # takes effect.
        raise ConfigError(f"fund {raw['ticker']} has unexpected fields: {sorted(unexpected)}")

    if raw["role"] not in VALID_ROLES:
        raise ConfigError(
            f"fund {raw['ticker']} has role {raw['role']!r}; expected one of {sorted(VALID_ROLES)}"
        )
    if not isinstance(raw["foreign_session"], bool):
        raise ConfigError(
            f"fund {raw['ticker']} foreign_session must be a bool, got {raw['foreign_session']!r}"
        )

    band = raw["outlier_band_bps"]
    if not isinstance(band, (int, float)) or isinstance(band, bool) or band <= 0:
        raise ConfigError(
            f"fund {raw['ticker']} outlier_band_bps must be a positive number, got {band!r}"
        )

    return Fund(
        ticker=str(raw["ticker"]),
        name=str(raw["name"]),
        sleeve=str(raw["sleeve"]),
        issuer=str(raw["issuer"]),
        foreign_session=raw["foreign_session"],
        outlier_band_bps=float(band),
        role=str(raw["role"]),
    )


def load_universe(path: Path | str = DEFAULT_CONFIG) -> Universe:
    """Load and validate the universe config.

    Raises:
        ConfigError: if the file is malformed in any way. There is no
            partial-success path; a bad config stops the pipeline here.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"universe config not found: {path}")

    with path.open() as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ConfigError(f"{path} did not parse to a mapping")
    for key in ("sample", "funds"):
        if key not in raw:
            raise ConfigError(f"{path} is missing top-level key {key!r}")

    sample = raw["sample"]
    if not isinstance(sample, dict) or "start" not in sample:
        raise ConfigError("sample must be a mapping containing 'start'")

    start = _parse_date(sample["start"], "start")
    end = _parse_date(sample["end"], "end") if sample.get("end") is not None else None
    if end is not None and end <= start:
        raise ConfigError(f"sample.end ({end}) must be after sample.start ({start})")

    raw_funds = raw["funds"]
    if not isinstance(raw_funds, list) or not raw_funds:
        raise ConfigError("funds must be a non-empty list")

    funds = tuple(_build_fund(f, i) for i, f in enumerate(raw_funds))

    tickers = [f.ticker for f in funds]
    duplicates = sorted({t for t in tickers if tickers.count(t) > 1})
    if duplicates:
        raise ConfigError(f"duplicate tickers in universe: {duplicates}")

    return Universe(start=start, end=end, funds=funds)
