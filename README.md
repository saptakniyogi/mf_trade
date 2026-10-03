# Mutual fund data enrichment fix

This package combines the AMFI/mftool performance fix with resilient mfdata enrichment.

## What changed

- Requires `mftool>=3.4,<4`.
- Keeps mftool/AMFI as the primary NAV and daily performance source.
- Retries AMFI performance across recent weekdays and never caches empty holiday responses.
- Uses mfdata bulk scheme lookup as a fast path, but a bulk timeout no longer disables enrichment.
- Falls back to independent mfdata scheme requests for every selected scheme missing after bulk lookup.
- Uses a 15-second mfdata timeout with one bounded retry for transient HTTP failures.
- Reads scheme-level AUM, returns, ratios and `family_id` from mfdata.
- Parses both nested and flat mfdata ratio formats, including PE/PB/PS, Sharpe, Sortino and standard deviation.
- Fetches family holdings once per unique family and derives sector allocation from the returned holdings.
- Enriches up to 20 unique fund families by default, configurable with `MF_DATA_FAMILY_ENRICHMENT_LIMIT`.
- Uses concurrent requests for independent scheme and family enrichment so one slow fund does not block the complete run.
- Caches successful mfdata responses for 24 hours by default.
- Does not invent market-cap weights. They remain missing when the provider does not supply a market-cap classification.
- Preserves explicit missing-data warnings for genuinely unavailable fields.

## Expected enrichment logs

If the bulk endpoint times out, the run should continue with messages similar to:

`mfdata POST failed for /api/v1/schemes/bulk: ...`

followed by:

`mfdata individual fallback: 20/20 schemes missing after bulk; fetching independently with 6 workers.`

On successful enrichment:

`mfdata enrichment completed: scheme_details=20/20 holdings=20/20 valuation=20/20`

The exact counts depend on provider coverage and the selected universe.

## Installation

```bash
pip install -r requirements.txt
```

## Optional Kaggle NAV backup

The app can use Kaggle's `tharunreddy2911/mutual-fund-historic-nav-data` as a secondary NAV-history provider when mftool/AMFI history is still missing. It is lazy-loaded through `kagglehub`, cached locally, and used only for NAV-derived metrics such as 1Y/3Y/5Y CAGR, volatility, max drawdown, Sharpe and Sortino. It does not provide AUM, holdings, sectors or valuation. `kagglehub.dataset_download()` supports downloading the public dataset and caches resources locally.

Kaggle is not treated as an authoritative live NAV source. By default, a matched history must be no more than 45 days old before its metrics are used. Override with `MF_KAGGLE_NAV_MAX_STALENESS_DAYS` if you deliberately want to accept older historical data.

The dataset's historical parquet stores `Scheme_Code` as the pandas index, so it appears to pandas as `Date` and `NAV` data columns only. The loader explicitly detects and restores that index as the scheme-code key before matching the requested funds.

Environment controls:

```text
MF_KAGGLE_NAV_ENABLED=true
MF_KAGGLE_NAV_DATASET=tharunreddy2911/mutual-fund-historic-nav-data
MF_KAGGLE_NAV_MAX_STALENESS_DAYS=45
MF_KAGGLE_NAV_CACHE_TTL_HOURS=168
MF_KAGGLE_NAV_CHUNK_SIZE=100000
```

If Kaggle asks for authentication, set `KAGGLE_API_TOKEN` or configure the Kaggle credentials supported by `kagglehub`. Public resources normally do not require authentication unless Kaggle requests consent/authentication.
