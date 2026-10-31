"""
NorthStar Retail — Fairness (slice) Check
CS 401R Lab 4, Task 1

Manual slice evaluation by loyalty tier, read from the metrics the training job
emitted. churn_model.train() puts the per-tier results in metrics["slices"], so
this file needs no model and no data, only EVAL_METRICS_PATH (which buildspec.yml
sets, and which --eval-metrics-path sets locally):

    pytest tests/test_fairness.py -v -rw --eval-metrics-path /tmp/out/evaluation_metrics.json

This check FLAGS, it does not BLOCK. If recall at the top-10% cutoff differs by
more than 10 percentage points between the best-served and worst-served tier,
the test still passes and a warning lands in the warnings summary of the build
log (buildspec.yml runs with -rw), where whoever approves the package sees it.
Expect the flag to fire on this data. The point is that it is visible, not that
it is quiet. What the tests DO fail on is a slice report that is missing or
malformed, because then nobody could see the flag.

Recall is measured at the global top-10% cutoff (the retention team contacts one
list of the 10% highest-risk customers), so the question per tier is what share
of that tier's churners make the list.
"""

import json
import math
import os
import warnings
from pathlib import Path

import pytest

TIERS = ["Bronze", "Silver", "Gold", "Platinum"]
RECALL_GAP_LIMIT = 0.10      # 10 percentage points
MIN_POSITIVES = 30           # below this a tier's recall is too noisy to conclude from
REQUIRED_KEYS = ["n", "positives", "recall_top10", "too_small_to_conclude"]


def recall_gap(slices):
    """(gap, highest_tier, lowest_tier) over tiers that have a recall to compare.

    A tier with no positive customers has recall NaN and takes no part.
    """
    recalls = {t: s["recall_top10"] for t, s in slices.items()
               if s.get("recall_top10") is not None and not math.isnan(s["recall_top10"])}
    if len(recalls) < 2:
        raise ValueError(f"Need recall for at least two tiers to compare, got {recalls}")
    hi = max(recalls, key=recalls.get)
    lo = min(recalls, key=recalls.get)
    return recalls[hi] - recalls[lo], hi, lo


@pytest.fixture(scope="module")
def eval_metrics():
    path = os.environ.get("EVAL_METRICS_PATH", "/tmp/eval_metrics.json")
    if not Path(path).exists():
        pytest.fail(f"Evaluation metrics not found at {path}. The fairness check reads the "
                    f"per-tier results the training job emitted.", pytrace=False)
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def slices(eval_metrics):
    s = eval_metrics.get("slices")
    if not s:
        pytest.fail("No 'slices' in the evaluation metrics. churn_model.train() must put "
                    "per-tier results in its metrics dict.", pytrace=False)
    return s


class TestSliceReport:

    def test_every_tier_is_reported_with_required_fields(self, slices):
        missing_tiers = [t for t in TIERS if t not in slices]
        assert not missing_tiers, f"Tiers missing from slice report: {missing_tiers}"
        for tier in TIERS:
            absent = [k for k in REQUIRED_KEYS if k not in slices[tier]]
            assert not absent, f"{tier} slice is missing {absent}"

    def test_slices_account_for_the_whole_validation_set(self, eval_metrics, slices):
        """A slice report that silently drops customers hides who it fails."""
        total = sum(slices[t]["n"] for t in TIERS)
        assert total == eval_metrics["n_val_samples"], (
            f"Tier counts sum to {total}, validation set has {eval_metrics['n_val_samples']}"
        )
        positives = sum(slices[t]["positives"] for t in TIERS)
        expected = round(eval_metrics["positive_rate_val"] * eval_metrics["n_val_samples"])
        assert abs(positives - expected) <= 1, (
            f"Tier positives sum to {positives}, validation set has about {expected}"
        )

    def test_recall_is_a_valid_rate_for_every_tier_with_churners(self, slices):
        for tier in TIERS:
            s = slices[tier]
            if s["positives"] == 0:
                continue
            assert 0.0 <= s["recall_top10"] <= 1.0, f"{tier} recall {s['recall_top10']} not in [0, 1]"

    def test_small_tier_marker_matches_the_positive_count(self, slices):
        for tier in TIERS:
            s = slices[tier]
            assert s["too_small_to_conclude"] == (s["positives"] < MIN_POSITIVES), (
                f"{tier}: too_small_to_conclude={s['too_small_to_conclude']} but "
                f"positives={s['positives']} (cutoff {MIN_POSITIVES})"
            )


class TestRecallGapFlag:

    def test_recall_gap_is_flagged_not_blocked(self, slices):
        """Warn when the gap exceeds 10pp. Never fail because of its size."""
        gap, hi, lo = recall_gap(slices)
        if gap > RECALL_GAP_LIMIT:
            warnings.warn(
                f"FAIRNESS FLAG: recall@top10% gap {gap * 100:.1f}pp exceeds "
                f"{RECALL_GAP_LIMIT * 100:.0f}pp. Highest {hi} "
                f"({slices[hi]['recall_top10']:.3f}), lowest {lo} "
                f"({slices[lo]['recall_top10']:.3f}). The approver must review this "
                f"before promoting the package."
            )
        assert math.isfinite(gap) and gap >= 0

    def test_underpowered_tiers_are_called_out(self, slices):
        """A tier with too few churners cannot support a fairness claim either
        way. Say so in the log instead of leaving the reader to infer it."""
        small = [t for t in TIERS if slices[t]["too_small_to_conclude"]]
        if small:
            detail = ", ".join(f"{t} ({slices[t]['positives']} positives)" for t in small)
            warnings.warn(f"FAIRNESS NOTE: fewer than {MIN_POSITIVES} churners in {detail}; "
                          f"their recall is too noisy to conclude from.")
        assert all(slices[t]["positives"] >= 0 for t in TIERS)


class TestRecallGapHelper:

    def test_gap_is_highest_minus_lowest(self):
        s = {"Bronze": {"recall_top10": 0.47}, "Silver": {"recall_top10": 0.18},
             "Gold": {"recall_top10": 0.12}, "Platinum": {"recall_top10": 0.10}}
        gap, hi, lo = recall_gap(s)
        assert (hi, lo) == ("Bronze", "Platinum")
        assert gap == pytest.approx(0.37)

    def test_tier_without_churners_is_ignored(self):
        s = {"Bronze": {"recall_top10": 0.40}, "Silver": {"recall_top10": 0.30},
             "Gold": {"recall_top10": float("nan")}, "Platinum": {"recall_top10": 0.35}}
        gap, hi, lo = recall_gap(s)
        assert (hi, lo) == ("Bronze", "Silver")
        assert gap == pytest.approx(0.10)
