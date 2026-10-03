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

## Primary mutual-fund NAV data source

The application now uses the AMFI-derived TigZig NAV data service as the primary current and historical NAV source. TigZig publishes a daily scheme snapshot containing scheme identity, classification, latest NAV and quarterly average AUM, and exposes full historical NAV by scheme code. The service is open and does not require an API key.

The primary refresh is lazy and runs at most once per day. If the refresh fails, the last successful local snapshot remains available. Historical scheme responses are cached locally for 24 hours.

Data-source order is:

1. TigZig / AMFI NAV snapshot and historical NAV
2. mftool / AMFI direct data and performance reports
3. Kaggle NAV data as a tertiary archival fallback
4. mfdata and local factsheets for fields not supplied by NAV sources, such as holdings, sectors and valuation

Relevant settings:

- `MF_TIGZIG_NAV_ENABLED=true`
- `MF_TIGZIG_NAV_BASE_URL=https://api.tigzig.com/mf/v1`
- `MF_TIGZIG_NAV_CACHE_TTL_HOURS=24`
- `MF_TIGZIG_NAV_CACHE_DIR=tigzig_nav_cache`
- `MF_TIGZIG_NAV_BULK_SIZE=50`
- `MF_KAGGLE_NAV_ENABLED=true`
- `MF_KAGGLE_NAV_DATASET=tharunreddy2911/mutual-fund-historic-nav-data`

TigZig is an AMFI-derived provider rather than the AMFI website itself. The application therefore keeps mftool/AMFI as a fallback and does not treat third-party NAV data as an exclusive dependency.

## Provider fail-fast behavior

Optional `mfdata.in` enrichment is fail-fast at the bulk-provider boundary. If the bulk endpoint times out or is unavailable, the application skips the individual scheme fan-out for that run so an external provider outage cannot stall the analysis for several minutes. Higher-priority AMFI-derived NAV data remains available.
