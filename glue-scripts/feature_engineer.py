"""
Glue ETL: processed/customers/ -> features/customers/ + SageMaker Feature Store

TEMPORAL SPLIT - read this before changing anything
---------------------------------------------------
    |<---- observation window ---->|<--- holdout --->|
                              FEATURE_CUTOFF (T)   SNAPSHOT

  * Every feature is computed ONLY from purchases on or before T.
  * churn_label is derived ONLY from the holdout: 1 if the customer made
    no purchase in (T, SNAPSHOT], else 0.

Mixing the two windows is the classic leakage failure. If recency were
computed over the full range, it would encode the answer directly and the
model would post a near-perfect AUC that collapses in production.

Note that roughly a third of churners are still buying right up to T. A
model using recency alone misses them; that is precisely why the rest of
the feature set exists.

This is also where the grain changes. Input is transaction level (many rows
per customer); output is customer level (exactly one row per customer). Every
feature is an aggregate over that customer's purchase history.

Job arguments (wired by Terraform in modules/glue):
  --input_path         s3:// processed Parquet
  --output_path        s3:// features destination
  --feature_group_name SageMaker Feature Group to ingest into
  --region             AWS region for the Feature Store runtime client
"""

import sys
import time

import boto3
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import functions as F

# Loyalty tier thresholds on total lifetime value (USD).
TIER_BRONZE_MAX = 500.0
TIER_SILVER_MAX = 2000.0
TIER_GOLD_MAX = 5000.0

# Churn risk bands. The score is a rule-based proxy, not a model - Lab 3
# replaces it with a trained prediction. It exists so the feature vector has
# a plausible label-shaped column to work with in the meantime.
RISK_HIGH_LO, RISK_HIGH_HI = 0.7, 1.0
RISK_MED_LO, RISK_MED_HI = 0.4, 0.7
RISK_LOW_LO, RISK_LOW_HI = 0.0, 0.4

# Timeline anchors. These MUST match the generator that produced the raw data.
FEATURE_CUTOFF = "2026-04-01"   # T - features use data on or before this date
SNAPSHOT = "2026-06-30"         # end of the holdout window

N_CATEGORIES = 8.0              # denominator for category_diversity_score


def split_windows(df):
    """Split into the observation window (<= T) and the holdout ((T, SNAPSHOT]).

    Return (history, holdout). Use the FEATURE_CUTOFF and SNAPSHOT constants
    above - do NOT use today() or max(purchase_date).

    Everything downstream depends on this being right. Features come only
    from history; the label comes only from holdout.
    """
    t = F.to_date(F.lit(FEATURE_CUTOFF))
    snap = F.to_date(F.lit(SNAPSHOT))
    history = df.filter(F.col("purchase_date") <= t)
    holdout = df.filter((F.col("purchase_date") > t) & (F.col("purchase_date") <= snap))
    return history, holdout


def compute_rfm_features(history):
    """One row per customer_id, every window measured backwards from T.

    Produce these columns:
      days_since_last_purchase   T minus the customer's last purchase
      customer_tenure_days       T minus the customer's first purchase
      purchase_frequency_30d     orders in the 30 days before T
      purchase_frequency_90d     orders in the 90 days before T
      purchase_frequency_180d    orders in the 180 days before T
      avg_order_value            mean order_value across all history
      total_spend_90d            sum of order_value in the 90 days before T
      total_lifetime_value       sum of all order_value
      avg_basket_size_6m         mean num_items per order, last 180 days
      category_diversity_score   distinct product_category / N_CATEGORIES,
                                 EXCLUDING 'unknown' - the transform imputes
                                 that for missing categories, and counting it
                                 as a real category pushes the score above 1.0
      online_to_store_ratio      fraction of orders with channel == 'online'

    Useful: F.datediff, F.date_sub, and conditional aggregates built with
    F.sum(F.when(...).otherwise(0)). Cast everything to double - Feature
    Store expects Fractional.

    Watch the divide-by-zero in avg_basket_size_6m: a customer with no
    orders in the last 180 days needs a guarded denominator.
    """
    t = F.to_date(F.lit(FEATURE_CUTOFF))
    days_before = F.datediff(t, F.col("purchase_date"))

    def in_window(days):
        return days_before < days

    orders_6m = F.sum(F.when(in_window(180), 1).otherwise(0))
    items_6m = F.sum(F.when(in_window(180), F.col("num_items")).otherwise(0))

    agg = history.groupBy("customer_id").agg(
        F.datediff(t, F.max("purchase_date")).alias("days_since_last_purchase"),
        F.datediff(t, F.min("purchase_date")).alias("customer_tenure_days"),
        F.sum(F.when(in_window(30), 1).otherwise(0)).alias("purchase_frequency_30d"),
        F.sum(F.when(in_window(90), 1).otherwise(0)).alias("purchase_frequency_90d"),
        orders_6m.alias("purchase_frequency_180d"),
        F.avg("order_value").alias("avg_order_value"),
        F.sum(F.when(in_window(90), F.col("order_value")).otherwise(0)).alias("total_spend_90d"),
        F.sum("order_value").alias("total_lifetime_value"),
        F.when(orders_6m > 0, items_6m / orders_6m).otherwise(0).alias("avg_basket_size_6m"),
        (F.countDistinct(F.when(F.col("product_category") != "unknown", F.col("product_category")))
         / N_CATEGORIES).alias("category_diversity_score"),
        F.avg(F.when(F.col("channel") == "online", 1).otherwise(0)).alias("online_to_store_ratio"),
    )

    for col in agg.columns:
        if col != "customer_id":
            agg = agg.withColumn(col, F.col(col).cast("double"))
    return agg


def assign_loyalty_tier(df):
    """Bucket total_lifetime_value into a loyalty_tier string column.

        Bronze   : < 500
        Silver   : 500 - 2000
        Gold     : 2000 - 5000
        Platinum : >= 5000

    Thresholds are the TIER_* constants. All four tiers must appear in your
    output; if one is empty, your thresholds or your LTV aggregation is wrong.
    """
    ltv = F.col("total_lifetime_value")
    return df.withColumn(
        "loyalty_tier",
        F.when(ltv < TIER_BRONZE_MAX, "Bronze")
         .when(ltv < TIER_SILVER_MAX, "Silver")
         .when(ltv < TIER_GOLD_MAX, "Gold")
         .otherwise("Platinum"),
    )


def compute_churn_proxy(df):
    """Rule-based churn_risk_score in [0, 1]. This is NOT the label.

        High   (0.7-1.0): days_since_last_purchase > 60 AND
                          purchase_frequency_30d == 0
        Medium (0.4-0.7): days_since_last_purchase > 30
        Low    (0.0-0.4): otherwise

    Scale within each band rather than emitting three constants, and clamp
    the result to [0, 1].

    This heuristic is deliberately kept as the baseline your Lab 3 model has
    to beat. A trained model that cannot outperform three lines of rules has
    not earned its deployment.
    """
    d = F.col("days_since_last_purchase")
    high = RISK_HIGH_LO + (RISK_HIGH_HI - RISK_HIGH_LO) * F.least((d - 60) / 120.0, F.lit(1.0))
    med = RISK_MED_LO + (RISK_MED_HI - RISK_MED_LO) * (d - 30) / 30.0
    low = RISK_LOW_LO + (RISK_LOW_HI - RISK_LOW_LO) * d / 30.0
    score = (F.when((d > 60) & (F.col("purchase_frequency_30d") == 0), high)
              .when(d > 30, med)
              .otherwise(low))
    return df.withColumn(
        "churn_risk_score",
        F.round(F.greatest(F.lit(0.0), F.least(F.lit(1.0), score)), 4),
    )


def attach_churn_label(features, holdout):
    """churn_label = 1 if the customer made NO purchase in the holdout window.

    Left-join the distinct customers seen in holdout onto features; anyone
    absent from holdout churned.

    Do not compute this from any feature. The label must come from the
    holdout window and nowhere else - that separation is what makes the
    resulting model honest.
    """
    active = holdout.select("customer_id").distinct().withColumn("_active", F.lit(1))
    return (features.join(active, "customer_id", "left")
                    .withColumn("churn_label",
                                F.when(F.col("_active").isNull(), 1).otherwise(0).cast("int"))
                    .drop("_active"))


def ingest_to_feature_store(rows, feature_group_name, region, event_time):
    """PutRecord each customer into the online store.

    Runs on the driver against a collected list. That is acceptable here
    because the output is one row per customer (~2k records) - at production
    scale this would be a foreachPartition with a client per partition.
    """
    client = boto3.client("sagemaker-featurestore-runtime", region_name=region)
    ingested = 0
    for r in rows:
        record = [
            {"FeatureName": "customer_id", "ValueAsString": str(r["customer_id"])},
            # event_time is Fractional: send epoch seconds as a numeric string.
            # An ISO 8601 timestamp here is rejected by PutRecord with a
            # ValidationError naming event_time, and the job fails.
            {"FeatureName": "event_time", "ValueAsString": str(event_time)},
        ] + [
            {"FeatureName": name, "ValueAsString": str(r[name])}
            for name in [
                "days_since_last_purchase", "customer_tenure_days",
                "purchase_frequency_30d", "purchase_frequency_90d",
                "purchase_frequency_180d", "avg_order_value", "total_spend_90d",
                "total_lifetime_value", "avg_basket_size_6m",
                "category_diversity_score", "online_to_store_ratio",
                "loyalty_tier", "churn_risk_score", "churn_label",
            ]
        ]
        client.put_record(FeatureGroupName=feature_group_name, Record=record)
        ingested += 1
    return ingested


def main():
    args = getResolvedOptions(
        sys.argv,
        ["JOB_NAME", "input_path", "output_path", "feature_group_name", "region"],
    )

    sc = SparkContext()
    glue_context = GlueContext(sc)
    spark = glue_context.spark_session
    job = Job(glue_context)
    job.init(args["JOB_NAME"], args)

    df = spark.read.parquet(args["input_path"])
    print(f"[features] read {df.count()} processed transaction rows")

    history, holdout = split_windows(df)
    print(f"[features] observation window (<= {FEATURE_CUTOFF}): {history.count()} rows")
    print(f"[features] holdout window ({FEATURE_CUTOFF} to {SNAPSHOT}): {holdout.count()} rows")

    features = compute_rfm_features(history)
    features = assign_loyalty_tier(features)
    features = compute_churn_proxy(features)
    features = attach_churn_label(features, holdout)

    for col in ["avg_order_value", "total_lifetime_value", "total_spend_90d",
                "avg_basket_size_6m", "category_diversity_score", "online_to_store_ratio"]:
        features = features.withColumn(col, F.round(F.col(col), 4))

    n_customers = features.count()
    print(f"[features] computed features for {n_customers} customers")

    # Producer-side quality gates. Failing here beats shipping a broken
    # feature set that Lab 3 trains on.
    for col in ["days_since_last_purchase", "customer_tenure_days",
                "purchase_frequency_30d", "purchase_frequency_90d",
                "purchase_frequency_180d", "avg_order_value", "total_spend_90d",
                "total_lifetime_value", "avg_basket_size_6m",
                "category_diversity_score", "online_to_store_ratio",
                "loyalty_tier", "churn_risk_score", "churn_label"]:
        nulls = features.filter(F.col(col).isNull()).count()
        assert nulls == 0, f"{col} has {nulls} null values"

    out_of_range = features.filter(
        (F.col("churn_risk_score") < 0) | (F.col("churn_risk_score") > 1)
    ).count()
    assert out_of_range == 0, f"churn_risk_score out of [0,1] for {out_of_range} rows"

    churn_rate = features.agg(F.avg("churn_label")).collect()[0][0]
    print(f"[features] churn_label rate: {churn_rate:.1%}")
    assert 0.05 < churn_rate < 0.50, \
        f"churn rate {churn_rate:.1%} is implausible - check the window split"

    tiers = {r["loyalty_tier"] for r in features.select("loyalty_tier").distinct().collect()}
    print(f"[features] tier distribution present: {sorted(tiers)}")
    assert tiers == {"Bronze", "Silver", "Gold", "Platinum"}, \
        f"tier distribution is degenerate: {sorted(tiers)}"

    event_time = float(int(time.time()))
    features = features.withColumn("event_time", F.lit(event_time))

    (features.coalesce(2)
             .write
             .mode("overwrite")
             .parquet(args["output_path"]))
    print(f"[features] wrote {n_customers} rows to {args['output_path']}")

    rows = [r.asDict() for r in features.collect()]
    ingested = ingest_to_feature_store(
        rows, args["feature_group_name"], args["region"], event_time
    )
    print(f"[features] ingested {ingested} records into "
          f"{args['feature_group_name']} at event_time {event_time}")

    job.commit()


if __name__ == "__main__":
    main()
