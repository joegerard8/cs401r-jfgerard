"""
NorthStar Retail — Feature Engineering Unit Tests
CS 401R Lab 4 Starter Kit

These tests validate your Lab 2 feature engineering pipeline — the code that
transforms raw transaction and customer data into the 11 model features
used by the churn prediction model.

Run locally:
    pytest tests/test_features.py -v

Design principle: Unit test the feature logic in isolation using small,
controlled DataFrames — do not depend on the full synthetic dataset.
"""

from datetime import datetime, timedelta

import os

import numpy as np
import pandas as pd
import pytest

# ── Import your feature engineering module ─────────────────────────────────────
# Lab 2 implements feature engineering in PySpark (glue-scripts/feature_engineer.py),
# which is awkward to unit test. For Lab 4 you extract the pure computation logic into
# testable functions in data/feature_engineering.py that operate on pandas DataFrames,
# then have the Glue job and these tests share that logic.
#
# Your pandas functions must compute EXACTLY what your Glue job computes --
# same windows (purchase_date > T - N days), same definitions. In particular,
# avg_basket_size_6m is ITEMS per order over 180 days (Lab 2 schema), not
# dollars. Check the port against your Glue output on the full dataset.
#
# That refactor IS the assignment here: code you cannot test in isolation is code you
# cannot gate a pipeline on.
try:
    from data.feature_engineering import (
        compute_recency_features,
        compute_frequency_features,
        compute_monetary_features,
        compute_behavioral_features,
        build_churn_feature_set,
    )
    FEATURES_MODULE_AVAILABLE = True
    _IMPORT_ERROR = None
except ImportError as e:
    FEATURES_MODULE_AVAILABLE = False
    _IMPORT_ERROR = e

# A missing module is a FAILURE, not a skip. Silently skipping means `pytest tests/`
# goes green having tested nothing, and a CI gate that cannot fail is not a gate.
# Set ALLOW_MISSING_FEATURES_MODULE=1 only while you are still building it.
if not FEATURES_MODULE_AVAILABLE and os.environ.get("ALLOW_MISSING_FEATURES_MODULE") != "1":
    pytest.fail(
        f"Could not import data.feature_engineering ({_IMPORT_ERROR!r}). Lab 4 Task 1 "
        "requires the Lab 2 feature logic extracted into testable pandas functions in "
        "data/feature_engineering.py. If the file exists, check that pytest.ini is at the "
        "repo root and that the module itself imports cleanly. Set "
        "ALLOW_MISSING_FEATURES_MODULE=1 to defer this while developing.",
        pytrace=False,
    )

pytestmark = pytest.mark.skipif(
    not FEATURES_MODULE_AVAILABLE,
    reason="feature module unavailable and ALLOW_MISSING_FEATURES_MODULE=1 was set"
)

# Must match FEATURE_CUTOFF in the Lab 2 feature job. Windows are measured
# backwards from here, never from today().
SNAPSHOT_DATE = datetime(2026, 4, 1)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_customers():
    """Customer roster. Matches the customer_id format Lab 2 generates."""
    return pd.DataFrame({
        "customer_id": ["CUST-10000001", "CUST-10000002", "CUST-10000003"],
        "loyalty_tier": ["Bronze", "Silver", "Platinum"],
        "churn_label": [1, 0, 0],
    })


@pytest.fixture
def sample_transactions():
    """Small transaction set at PROCESSED grain.

    Columns match Lab 2's transform output exactly:
        transaction_id, customer_id, purchase_date, order_value,
        num_items, payment_method, channel, store_id, product_category

    Three deliberately different customers:
      CUST-10000001  lapsing  - last purchase 60d before T, 2 buys, one category, store only
      CUST-10000002  active   - last purchase 5d before T, 8 buys, two categories, mixed channel
      CUST-10000003  high value - last purchase 30d before T, large baskets, four categories
    """
    T = SNAPSHOT_DATE
    rows = []

    def tx(cid, days_ago, value, items, channel, category, n):
        return {
            "transaction_id": f"TXN-{cid[-4:]}{n:08d}",
            "customer_id": cid,
            "purchase_date": (T - timedelta(days=days_ago)).isoformat(),
            "order_value": value,
            "num_items": items,
            "payment_method": "credit_card",
            "channel": channel,
            "store_id": "ONLINE" if channel == "online" else "STORE-042",
            "product_category": category,
        }

    # Customer 1 - lapsing: 2 purchases, both inside 90d, single category, store only
    rows.append(tx("CUST-10000001", 60, 89.99, 2, "store", "Apparel", 1))
    rows.append(tx("CUST-10000001", 85, 45.00, 1, "store", "Apparel", 2))

    # Customer 2 - active: 8 purchases inside 90d, alternating channel, two categories
    for i in range(8):
        rows.append(tx("CUST-10000002", 5 + i * 10, 55.00 + i * 5, 2,
                       "online" if i % 2 == 0 else "store",
                       "Footwear" if i % 2 == 0 else "Outdoor", 10 + i))

    # Customer 3 - high value: large baskets across four categories
    for i, cat in enumerate(["Electronics", "Home", "Beauty", "Grocery"]):
        rows.append(tx("CUST-10000003", 30 + i * 25, 365.00, 5,
                       "online" if i % 2 == 0 else "store", cat, 20 + i))

    return pd.DataFrame(rows)


class TestRecencyFeatures:

    def test_days_since_last_purchase_correct(self, sample_customers, sample_transactions):
        features = compute_recency_features(sample_transactions, SNAPSHOT_DATE)
        cust1 = features[features["customer_id"] == "CUST-10000001"].iloc[0]
        # Last purchase 60 days ago
        assert 58 <= cust1["days_since_last_purchase"] <= 62, (
            f"Expected ~60, got {cust1['days_since_last_purchase']}"
        )

    def test_recent_buyer_low_recency(self, sample_customers, sample_transactions):
        features = compute_recency_features(sample_transactions, SNAPSHOT_DATE)
        cust2 = features[features["customer_id"] == "CUST-10000002"].iloc[0]
        assert cust2["days_since_last_purchase"] <= 10, (
            f"Recent buyer should have low recency, got {cust2['days_since_last_purchase']}"
        )

    def test_no_negative_recency(self, sample_transactions):
        features = compute_recency_features(sample_transactions, SNAPSHOT_DATE)
        assert (features["days_since_last_purchase"] >= 0).all(), (
            "days_since_last_purchase should never be negative"
        )

    def test_customer_with_no_transactions_returns_sentinel(self, sample_customers):
        """Customer with no transactions should get a sentinel value (e.g., 999)."""
        empty_txns = pd.DataFrame(columns=["customer_id", "purchase_date", "order_value"])
        customer_ids = pd.DataFrame({"customer_id": ["CUST-NODATA"]})

        features = compute_recency_features(empty_txns, SNAPSHOT_DATE, customer_ids)
        assert len(features) == 1
        assert features.iloc[0]["customer_id"] == "CUST-NODATA"
        assert features.iloc[0]["days_since_last_purchase"] == 999
        assert features.iloc[0]["customer_tenure_days"] == 0


# ── Frequency Feature Tests ────────────────────────────────────────────────────

class TestFrequencyFeatures:

    def test_frequency_90d_correct(self, sample_transactions):
        features = compute_frequency_features(sample_transactions, SNAPSHOT_DATE)
        cust2 = features[features["customer_id"] == "CUST-10000002"].iloc[0]
        # Customer 2 has 8 transactions, all in last 80 days (within 90-day window)
        assert cust2["purchase_frequency_90d"] == 8, (
            f"Expected 8 purchases in 90 days, got {cust2['purchase_frequency_90d']}"
        )

    def test_frequency_90d_excludes_older_transactions(self, sample_transactions):
        """Customer 1 has 2 purchases — one at 60 days (in window), one at 85 days (in window)."""
        features = compute_frequency_features(sample_transactions, SNAPSHOT_DATE)
        cust1 = features[features["customer_id"] == "CUST-10000001"].iloc[0]
        assert cust1["purchase_frequency_90d"] == 2

    def test_frequency_is_non_negative(self, sample_transactions):
        features = compute_frequency_features(sample_transactions, SNAPSHOT_DATE)
        assert (features["purchase_frequency_90d"] >= 0).all()
        assert (features["purchase_frequency_180d"] >= 0).all()

    def test_frequency_180d_gte_frequency_90d(self, sample_transactions):
        """180-day frequency must always be >= 90-day frequency."""
        features = compute_frequency_features(sample_transactions, SNAPSHOT_DATE)
        violations = features[features["purchase_frequency_180d"] < features["purchase_frequency_90d"]]
        assert len(violations) == 0, (
            f"{len(violations)} customers have 180d_freq < 90d_freq (impossible)"
        )


# ── Monetary Feature Tests ─────────────────────────────────────────────────────

class TestMonetaryFeatures:

    def test_avg_basket_size_correct(self, sample_transactions):
        """avg_basket_size_6m = items bought in the last 180 days / orders in
        the last 180 days (Lab 2: "Mean items per order, last 180 days").
        Customer 3's four orders each have num_items=5, so 5.0. The dollar
        figure, 365, is avg_order_value."""
        features = compute_monetary_features(sample_transactions, SNAPSHOT_DATE)
        cust3 = features[features["customer_id"] == "CUST-10000003"].iloc[0]
        assert cust3["avg_basket_size_6m"] == pytest.approx(5.0), (
            f"Expected 5.0 items per order, got {cust3['avg_basket_size_6m']}"
        )
        assert cust3["avg_order_value"] == pytest.approx(365.0)

    def test_total_spend_90d_is_sum_not_average(self, sample_transactions):
        features = compute_monetary_features(sample_transactions, SNAPSHOT_DATE)
        cust2 = features[features["customer_id"] == "CUST-10000002"].iloc[0]
        # Customer 2: 8 transactions of $55-$90 in 90 days
        assert cust2["total_spend_90d"] > 400, (
            f"total_spend_90d should be sum, not average. Got {cust2['total_spend_90d']}"
        )

    def test_monetary_features_non_negative(self, sample_transactions):
        features = compute_monetary_features(sample_transactions, SNAPSHOT_DATE)
        assert (features["avg_basket_size_6m"] >= 0).all()
        assert (features["total_spend_90d"] >= 0).all()


# ── Behavioral Feature Tests ───────────────────────────────────────────────────

class TestBehavioralFeatures:

    def test_category_diversity_score_range(self, sample_transactions):
        """Category diversity should be between 0 and 1."""
        features = compute_behavioral_features(sample_transactions)
        assert (features["category_diversity_score"] >= 0).all()
        assert (features["category_diversity_score"] <= 1).all()

    def test_single_category_buyer_has_low_diversity(self, sample_transactions):
        features = compute_behavioral_features(sample_transactions)
        cust1 = features[features["customer_id"] == "CUST-10000001"].iloc[0]
        # Customer 1 only bought Apparel
        assert cust1["category_diversity_score"] < 0.3, (
            f"Single-category buyer should have low diversity, got {cust1['category_diversity_score']}"
        )

    def test_multi_category_buyer_has_high_diversity(self, sample_transactions):
        features = compute_behavioral_features(sample_transactions)
        cust3 = features[features["customer_id"] == "CUST-10000003"].iloc[0]
        # Customer 3 bought four of the eight categories: 4 / 8 = 0.5
        assert cust3["category_diversity_score"] > 0.3, (
            f"Multi-category buyer should have higher diversity"
        )


    def test_online_to_store_ratio_non_negative(self, sample_transactions):
        features = compute_behavioral_features(sample_transactions)
        assert (features["online_to_store_ratio"] >= 0).all()


# ── Engagement Feature Tests ───────────────────────────────────────────────────


class TestFullFeatureSet:

    REQUIRED_FEATURES = [
        "days_since_last_purchase",
        "purchase_frequency_90d",
        "purchase_frequency_180d",
        "avg_basket_size_6m",
        "total_spend_90d",
        "category_diversity_score",
        "online_to_store_ratio",
        "customer_tenure_days",
    ]

    def test_all_required_features_present(self, sample_customers, sample_transactions):
        """build_churn_feature_set must produce every feature listed above."""
        features = build_churn_feature_set(
            customers=sample_customers,
            transactions=sample_transactions,
            snapshot_date=SNAPSHOT_DATE,
        )
        missing = [f for f in self.REQUIRED_FEATURES if f not in features.columns]
        assert not missing, f"Missing features: {missing}"

    def test_no_null_features_for_known_customers(self, sample_customers, sample_transactions):
        """No feature is NaN for a customer with at least one transaction.

        Null strategy: a customer with no purchases on or before T gets 999 days
        of recency, 0 tenure and 0 everywhere else (see feature_engineering.py).
        Customers who do have purchases never need filling.
        """
        features = build_churn_feature_set(
            customers=sample_customers,
            transactions=sample_transactions,
            snapshot_date=SNAPSHOT_DATE,
        )
        # Check customers with transactions
        active_customer_ids = sample_transactions["customer_id"].unique()
        active_features = features[features["customer_id"].isin(active_customer_ids)]
        null_counts = active_features[self.REQUIRED_FEATURES].isna().sum()
        assert len(active_features) == 3
        assert null_counts.sum() == 0, f"Null features for active customers: {null_counts[null_counts > 0].to_dict()}"

    def test_output_has_customer_id_column(self, sample_customers, sample_transactions):
        features = build_churn_feature_set(
            customers=sample_customers,
            transactions=sample_transactions,
            snapshot_date=SNAPSHOT_DATE,
        )
        assert "customer_id" in features.columns, "Output must include customer_id for joining"


# ── Boundary and edge cases ────────────────────────────────────────────────────
#
# Small hand-built inputs, one property each. The fixture customers above cover
# the normal case; these pin the places the Glue job's definitions are easy to
# get wrong (window edges, the cutoff itself, empty and single-row customers).

def _txn(cid, days_before_T, value=50.0, items=2, channel="store", category="Apparel", n=1):
    """One processed-grain row, `days_before_T` days before the snapshot (negative = after)."""
    return {
        "transaction_id": f"TXN-{cid[-4:]}{n:08d}",
        "customer_id": cid,
        "purchase_date": (SNAPSHOT_DATE - timedelta(days=days_before_T)).isoformat(),
        "order_value": value,
        "num_items": items,
        "payment_method": "credit_card",
        "channel": channel,
        "store_id": "ONLINE" if channel == "online" else "STORE-042",
        "product_category": category,
    }


def _features_for(rows, customers=None):
    txns = pd.DataFrame(rows)
    if customers is None:
        customers = pd.DataFrame({"customer_id": sorted(txns["customer_id"].unique())})
    return build_churn_feature_set(customers, txns, SNAPSHOT_DATE).set_index("customer_id")


class TestBoundaryAndEdgeCases:

    def test_customer_with_zero_purchases_gets_sentinels_not_nan(self):
        """Boundary: a customer on the roster with no transactions at all."""
        customers = pd.DataFrame({"customer_id": ["CUST-10000001", "CUST-NOBUY"]})
        f = _features_for([_txn("CUST-10000001", 10)], customers)
        row = f.loc["CUST-NOBUY"]
        assert row["days_since_last_purchase"] == 999
        assert row["purchase_frequency_180d"] == 0
        assert row["total_lifetime_value"] == 0
        assert row["avg_basket_size_6m"] == 0
        assert not row.isna().any(), f"NaN features for a zero-purchase customer: {row[row.isna()].index.tolist()}"

    def test_single_transaction_customer(self):
        """Edge: one order. Recency equals tenure and every average is that order."""
        f = _features_for([_txn("CUST-10000001", 40, value=120.0, items=3, channel="online",
                                category="Beauty")])
        row = f.loc["CUST-10000001"]
        assert row["days_since_last_purchase"] == 40
        assert row["customer_tenure_days"] == 40
        assert row["purchase_frequency_90d"] == 1
        assert row["avg_order_value"] == pytest.approx(120.0)
        assert row["total_lifetime_value"] == pytest.approx(120.0)
        assert row["avg_basket_size_6m"] == pytest.approx(3.0)
        assert row["category_diversity_score"] == pytest.approx(1 / 8)
        assert row["online_to_store_ratio"] == pytest.approx(1.0)

    def test_window_edge_is_exclusive(self):
        """Boundary: a purchase exactly 90 days before T is outside the 90-day
        window (days_before < 90) but inside the 180-day one; 89 days is inside."""
        f = _features_for([
            _txn("CUST-10000001", 90, n=1),
            _txn("CUST-10000002", 89, n=2),
        ])
        assert f.loc["CUST-10000001", "purchase_frequency_90d"] == 0
        assert f.loc["CUST-10000001", "purchase_frequency_180d"] == 1
        assert f.loc["CUST-10000002", "purchase_frequency_90d"] == 1

    def test_purchase_on_cutoff_day_counts(self):
        """Boundary: a purchase on T itself is in the observation window."""
        f = _features_for([_txn("CUST-10000001", 0)])
        row = f.loc["CUST-10000001"]
        assert row["days_since_last_purchase"] == 0
        assert row["purchase_frequency_30d"] == 1

    def test_post_cutoff_purchases_are_ignored(self):
        """Leakage guard: rows after T belong to the holdout the label comes from
        and must not move any feature."""
        before = _features_for([_txn("CUST-10000001", 20, n=1)])
        after = _features_for([_txn("CUST-10000001", 20, n=1),
                               _txn("CUST-10000001", -10, value=999.0, n=2),
                               _txn("CUST-10000001", -40, value=999.0, n=3)])
        pd.testing.assert_frame_equal(before, after)

    def test_unknown_category_does_not_count_as_diversity(self):
        """The transform imputes 'unknown'; counting it would push the score
        above what the customer really bought (Lab 2 note)."""
        f = _features_for([
            _txn("CUST-10000001", 10, category="unknown", n=1),
            _txn("CUST-10000002", 10, category="unknown", n=2),
            _txn("CUST-10000002", 20, category="Apparel", n=3),
        ])
        assert f.loc["CUST-10000001", "category_diversity_score"] == 0
        assert f.loc["CUST-10000002", "category_diversity_score"] == pytest.approx(1 / 8)

    def test_basket_size_is_zero_when_no_orders_in_last_180_days(self):
        """Edge: the only purchase is older than 180 days, so the basket
        denominator is zero. Must be 0, not NaN or an error."""
        f = _features_for([_txn("CUST-10000001", 200, items=4)])
        row = f.loc["CUST-10000001"]
        assert row["avg_basket_size_6m"] == 0
        assert row["purchase_frequency_180d"] == 0
        assert row["total_lifetime_value"] == pytest.approx(50.0)

    def test_online_ratio_is_fraction_of_orders(self):
        f = _features_for([
            _txn("CUST-10000001", 5, channel="online", n=1),
            _txn("CUST-10000001", 15, channel="store", n=2),
            _txn("CUST-10000001", 25, channel="store", n=3),
            _txn("CUST-10000001", 35, channel="online", n=4),
        ])
        assert f.loc["CUST-10000001", "online_to_store_ratio"] == pytest.approx(0.5)

    def test_frequencies_nest_for_every_customer(self):
        """30d <= 90d <= 180d holds for any input, including a spread-out one."""
        rows = [_txn("CUST-10000001", d, n=i) for i, d in enumerate([0, 29, 30, 89, 90, 179, 180, 300])]
        f = _features_for(rows).loc["CUST-10000001"]
        assert (f["purchase_frequency_30d"], f["purchase_frequency_90d"],
                f["purchase_frequency_180d"]) == (2, 4, 6)

    def test_does_not_mutate_input_transactions(self):
        txns = pd.DataFrame([_txn("CUST-10000001", 10)])
        snapshot = txns.copy(deep=True)
        build_churn_feature_set(pd.DataFrame({"customer_id": ["CUST-10000001"]}), txns, SNAPSHOT_DATE)
        pd.testing.assert_frame_equal(txns, snapshot)


# DELIBERATE FAILURE for the Lab 4 failure demo. Reverted in the next commit.
def test_deliberate_ci_failure_demo():
    assert 1 + 1 == 3, "deliberate failure: demonstrating that PRE_BUILD halts the pipeline"
