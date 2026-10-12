"""Pattern detection (swings, BOS, CHoCH, fair value gaps) and a small experimental predictor."""
import io
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

REQUIRED = ["published_date", "open", "high", "low", "close"]
NUMERIC = ["open", "high", "low", "close", "per_change", "traded_quantity", "traded_amount"]


def load_csv(raw: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise ValueError(f"Could not read CSV: {e}")
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    for c in ["per_change", "traded_quantity", "traded_amount"]:
        if c not in df.columns:
            df[c] = np.nan
    if "status" not in df.columns:
        df["status"] = ""
    df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")
    for c in NUMERIC:
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace(",", "").str.replace("%", "").str.strip(),
            errors="coerce",
        )
    df = df.dropna(subset=["published_date", "open", "high", "low", "close"])
    df = df.drop_duplicates("published_date").sort_values("published_date").reset_index(drop=True)
    if len(df) < 20:
        raise ValueError("Need at least 20 valid rows.")
    df["per_change"] = df["per_change"].fillna(df["close"].pct_change() * 100)
    return df


def find_swings(high, low, n):
    """Fractal swings: a high/low that is the extreme of n bars on each side.
    A swing is only 'known' n bars later (confirm), so nothing looks into the future."""
    swings = []
    for i in range(n, len(high) - n):
        if high[i] >= high[i - n : i + n + 1].max() and high[i] > high[i - n : i].max():
            swings.append({"i": i, "kind": "high", "price": float(high[i]), "confirm": i + n})
        if low[i] <= low[i - n : i + n + 1].min() and low[i] < low[i - n : i].min():
            swings.append({"i": i, "kind": "low", "price": float(low[i]), "confirm": i + n})
    return swings


def find_structure(close, swings):
    """BOS = close breaks a swing in the direction of the current trend.
    CHoCH = close breaks a swing against the current trend (trend flips)."""
    by_confirm = {}
    for s in swings:
        by_confirm.setdefault(s["confirm"], []).append(s)
    n = len(close)
    events, trend_arr, since_arr = [], np.zeros(n), np.zeros(n)
    trend, hi, lo, last_break = 0, None, None, 0
    for i in range(n):
        for s in by_confirm.get(i, []):
            if s["kind"] == "high":
                hi = s
            else:
                lo = s
        if hi is not None and close[i] > hi["price"]:
            kind = "CHoCH" if trend == -1 else "BOS"
            events.append({"type": kind, "direction": "bull", "from_i": hi["i"], "to_i": i, "price": hi["price"]})
            trend, hi, last_break = 1, None, i
        elif lo is not None and close[i] < lo["price"]:
            kind = "CHoCH" if trend == 1 else "BOS"
            events.append({"type": kind, "direction": "bear", "from_i": lo["i"], "to_i": i, "price": lo["price"]})
            trend, lo, last_break = -1, None, i
        trend_arr[i] = trend
        since_arr[i] = i - last_break
    return events, trend_arr, since_arr


def find_fvgs(o, h, l, c, min_pct=0.001):
    """Three-candle imbalance. Bullish: low[i] > high[i-2]. Bearish: high[i] < low[i-2]."""
    n = len(c)
    out = []
    for i in range(2, n):
        if l[i] > h[i - 2]:
            direction, top, bottom = "bull", float(l[i]), float(h[i - 2])
        elif h[i] < l[i - 2]:
            direction, top, bottom = "bear", float(l[i - 2]), float(h[i])
        else:
            continue
        if (top - bottom) / c[i] < min_pct:
            continue
        fill = None
        for j in range(i + 1, n):
            if (direction == "bull" and l[j] <= bottom) or (direction == "bear" and h[j] >= top):
                fill = j
                break
        out.append({"direction": direction, "top": top, "bottom": bottom,
                    "start_i": i - 2, "formed_i": i, "end_i": fill if fill is not None else n - 1,
                    "filled": fill is not None, "fill_i": fill})
    return out


def open_fvg_counts(n, fvgs):
    bull, bear = np.zeros(n + 1), np.zeros(n + 1)
    for f in fvgs:
        arr = bull if f["direction"] == "bull" else bear
        arr[f["formed_i"]] += 1
        if f["fill_i"] is not None:
            arr[f["fill_i"]] -= 1  # gap is no longer open on the bar that fills it
    return np.cumsum(bull)[:n], np.cumsum(bear)[:n]


def build_features(df, trend, since, open_bull, open_bear):
    c = df["close"]
    f = pd.DataFrame(index=df.index)
    f["ret1"], f["ret3"], f["ret5"] = c.pct_change(), c.pct_change(3), c.pct_change(5)
    f["range_pct"] = (df["high"] - df["low"]) / c
    f["body_pct"] = (c - df["open"]) / df["open"]
    q = df["traded_quantity"]
    f["vol_ratio"] = q / q.rolling(20).mean()
    d = c.diff()
    up, dn = d.clip(lower=0).rolling(14).mean(), (-d.clip(upper=0)).rolling(14).mean()
    f["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    f["dist_ma20"] = c / c.rolling(20).mean() - 1
    f["trend"], f["bars_since_break"] = trend, since
    f["open_bull_fvg"], f["open_bear_fvg"] = open_bull, open_bear
    return f.replace([np.inf, -np.inf], np.nan)


def predict_next(df, feats):
    n = len(df)
    if n < 120:
        return {"available": False, "reason": "Need at least 120 rows to train a model."}
    y = (df["close"].shift(-1) > df["close"]).astype(int)
    idx = feats.dropna(subset=["ret5", "dist_ma20", "rsi"]).index
    train_idx = idx[idx < n - 1]
    split = int(len(train_idx) * 0.8)
    Xtr, ytr = feats.loc[train_idx[:split]], y.loc[train_idx[:split]]
    Xte, yte = feats.loc[train_idx[split:]], y.loc[train_idx[split:]]
    params = dict(max_depth=3, learning_rate=0.05, max_iter=150, random_state=0)
    model = HistGradientBoostingClassifier(**params).fit(Xtr, ytr)
    acc = float((model.predict(Xte) == yte).mean())
    majority = int(ytr.mean() >= 0.5)
    baseline = float((yte == majority).mean())
    final = HistGradientBoostingClassifier(**params).fit(feats.loc[train_idx], y.loc[train_idx])
    prob_up = float(final.predict_proba(feats.iloc[[-1]])[0][1])
    return {"available": True, "prob_up": prob_up, "holdout_accuracy": acc, "baseline_accuracy": baseline,
            "n_train": int(len(Xtr)), "n_test": int(len(Xte))}


def _num(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x) else float(x)


def analyze(df: pd.DataFrame, swing: int = 3):
    swing = max(1, min(int(swing), 15))
    o, h, l, c = (df[k].to_numpy(float) for k in ["open", "high", "low", "close"])
    dates = df["published_date"].dt.strftime("%Y-%m-%d").tolist()

    swings = find_swings(h, l, swing)
    events, trend, since = find_structure(c, swings)
    fvgs = find_fvgs(o, h, l, c)
    ob, os_ = open_fvg_counts(len(df), fvgs)
    feats = build_features(df, trend, since, ob, os_)
    try:
        prediction = predict_next(df, feats)
    except Exception as e:  # keep the chart working even if the model fails
        prediction = {"available": False, "reason": f"Model error: {e}"}

    rows = [{"time": dates[i], "open": o[i], "high": h[i], "low": l[i], "close": c[i],
             "volume": _num(df["traded_quantity"][i]), "per_change": _num(df["per_change"][i]),
             "amount": _num(df["traded_amount"][i]), "status": str(df["status"][i])} for i in range(len(df))]
    state = {1: "bullish", -1: "bearish", 0: "neutral"}[int(trend[-1])]
    return {
        "rows": rows,
        "swings": [{"time": dates[s["i"]], "kind": s["kind"], "price": s["price"]} for s in swings],
        "events": [{"type": e["type"], "direction": e["direction"], "price": e["price"],
                    "from_time": dates[e["from_i"]], "to_time": dates[e["to_i"]]} for e in events],
        "fvgs": [{"direction": f["direction"], "top": f["top"], "bottom": f["bottom"], "filled": f["filled"],
                  "start_time": dates[f["start_i"]], "end_time": dates[f["end_i"]]} for f in fvgs],
        "structure": state,
        "prediction": prediction,
    }
