# Stock Structure Analyzer

Upload a CSV and get a TradingView-style chart with swings, BOS, CHoCH, fair value gaps,
plus an experimental next-candle probability.

CSV columns: `published_date, open, high, low, close, per_change, traded_quantity, traded_amount, status`
(only the first five are required).

## Run
    pip install -r requirements.txt
    uvicorn main:app --reload
Open http://127.0.0.1:8000 and upload `sample_data.csv` (or your own file).
The chart library loads from a CDN, so the browser needs internet access.

## Notes
- BOS/CHoCH/FVG detection is rule-based (analysis.py). Swing length is adjustable in the UI.
- The prediction is a small gradient-boosting model. It reports accuracy on the last 20% of your data
  next to a naive baseline. If it does not beat the baseline, treat it as noise. Not financial advice.
