"""Pandas port of the Lab 2 feature job (glue-scripts/feature_engineer.py).

The Glue job is PySpark and cannot be unit tested without a cluster. These
functions compute the same 11 features from a transaction-grain DataFrame so
tests/test_features.py can pin their behaviour on small hand-built inputs, and
tests/test_data.py can recompute the pipeline's output to check for leakage.

They must match the Glue job exactly:

  * Only purchases on or before the snapshot date T count (the observation
    window). Anything later is the holdout the label comes from.
  * Windows are measured backwards from T in whole days, and "in the N-day
    window" means days_before < N, so a purchase exactly N days before T is out.
  * category_diversity_score excludes 'unknown' (the transform's imputed value)
    and divides by N_CATEGORIES.
  * avg_basket_size_6m is ITEMS per order over 180 days, not dollars, and is 0
    when there were no orders in the window.
  * Windows are anchored to T, never to today().

tests/test_data.py checks the port against the Glue output on the full dataset.
"""

from __future__ import annotations

import pandas as pd

# Same constants as glue-scripts/feature_engineer.py.
FEATURE_CUTOFF = pd.Timestamp("2026-04-01")
N_CATEGORIES = 8.0

# A customer with no purchases on or before T has no recency to measure. Rather
# than NaN (which XGBoost would treat as informative missingness) they get a
# large recency, so they look maximally dormant, and zeros everywhere else.
NO_PURCHASE_RECENCY = 999


def _history(transactions: pd.DataFrame, snapshot_date) -> pd.DataFrame:
    """Rows on or before T, with a whole-day `days_before` column (T minus date)."""
    t = pd.Timestamp(snapshot_date).normalize()
    df = transactions.copy()
    df["purchase_date"] = pd.to_datetime(df["purchase_date"]).dt.normalize()
    df["days_before"] = (t - df["purchase_date"]).dt.days
    return df[df["days_before"] >= 0]


def _ids(customer_ids) -> list | None:
    if customer_ids is None:
        return None
    if isinstance(customer_ids, pd.DataFrame):
        customer_ids = customer_ids["customer_id"]
    return list(pd.Series(customer_ids).drop_duplicates())


def _assemble(columns: dict, hist: pd.DataFrame, customer_ids, fill: dict) -> pd.DataFrame:
    """One row per customer. Customers with no history get `fill` values.

    The roster is `customer_ids` when given, else every customer seen in hist.
    """
    ids = _ids(customer_ids)
    if ids is None:
        ids = sorted(hist["customer_id"].unique())
    out = pd.DataFrame({name: s.reindex(ids) for name, s in columns.items()}, index=ids)
    out = out.fillna(value=fill).astype("float64")
    out.index.name = "customer_id"
    return out.reset_index()


def compute_recency_features(transactions, snapshot_date, customer_ids=None) -> pd.DataFrame:
    """days_since_last_purchase and customer_tenure_days, in days before T."""
    hist = _history(transactions, snapshot_date)
    g = hist.groupby("customer_id")["days_before"]
    return _assemble(
        {"days_since_last_purchase": g.min(), "customer_tenure_days": g.max()},
        hist, customer_ids,
        fill={"days_since_last_purchase": NO_PURCHASE_RECENCY, "customer_tenure_days": 0},
    )


def compute_frequency_features(transactions, snapshot_date, customer_ids=None) -> pd.DataFrame:
    """Orders in the 30, 90 and 180 days before T (each row is one order)."""
    hist = _history(transactions, snapshot_date)
    cols = {
        f"purchase_frequency_{n}d": (hist["days_before"] < n).groupby(hist["customer_id"]).sum()
        for n in (30, 90, 180)
    }
    return _assemble(cols, hist, customer_ids, fill={name: 0 for name in cols})


def compute_monetary_features(transactions, snapshot_date, customer_ids=None) -> pd.DataFrame:
    """avg_order_value, total_spend_90d, total_lifetime_value, avg_basket_size_6m."""
    hist = _history(transactions, snapshot_date)
    by_cust = hist.groupby("customer_id")

    in_90 = hist["days_before"] < 90
    in_180 = hist["days_before"] < 180
    spend_90 = hist["order_value"].where(in_90, 0).groupby(hist["customer_id"]).sum()
    orders_180 = in_180.groupby(hist["customer_id"]).sum()
    items_180 = hist["num_items"].where(in_180, 0).groupby(hist["customer_id"]).sum()
    # Guard the denominator: no orders in the window means a basket size of 0.
    basket = (items_180 / orders_180.where(orders_180 > 0)).fillna(0)

    cols = {
        "avg_order_value": by_cust["order_value"].mean(),
        "total_spend_90d": spend_90,
        "total_lifetime_value": by_cust["order_value"].sum(),
        "avg_basket_size_6m": basket,
    }
    return _assemble(cols, hist, customer_ids, fill={name: 0 for name in cols})


def compute_behavioral_features(transactions, snapshot_date=None, customer_ids=None) -> pd.DataFrame:
    """category_diversity_score and online_to_store_ratio.

    With snapshot_date the history filter applies, as in the Glue job. Without
    it every row passed in is used, which is what a caller who has already
    filtered wants.
    """
    if snapshot_date is None:
        hist = transactions.copy()
    else:
        hist = _history(transactions, snapshot_date)

    real = hist[hist["product_category"] != "unknown"]
    diversity = real.groupby("customer_id")["product_category"].nunique() / N_CATEGORIES
    online = (hist["channel"] == "online").groupby(hist["customer_id"]).mean()

    cols = {"category_diversity_score": diversity, "online_to_store_ratio": online}
    return _assemble(cols, hist, customer_ids, fill={name: 0 for name in cols})


def build_churn_feature_set(customers: pd.DataFrame, transactions: pd.DataFrame,
                            snapshot_date) -> pd.DataFrame:
    """The 11 model features for every customer in `customers`.

    Customers with no purchases on or before T are kept and get the sentinel
    values above, so the output never silently loses a row. Any other columns
    on `customers` (loyalty_tier, churn_label, ...) are carried through.
    """
    roster = customers[["customer_id"]]
    parts = [
        compute_recency_features(transactions, snapshot_date, roster),
        compute_frequency_features(transactions, snapshot_date, roster),
        compute_monetary_features(transactions, snapshot_date, roster),
        compute_behavioral_features(transactions, snapshot_date, roster),
    ]
    out = customers.copy()
    for part in parts:
        out = out.merge(part, on="customer_id", how="left")
    return out
