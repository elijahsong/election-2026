import math
import os

import numpy as np
import pandas as pd
import pytest

import nc_2020


def test_lower_bounds_parse_census_bin_labels():
    assert [nc_2020._lower_bound(b) for b in ("Under 5 years", "85 years and over")] == [0, 85]
    assert [nc_2020._lower_bound(b) for b in ("Less than $10,000", "$10,000 to $14,999", "$200,000 or more")] == [
        0, 10_000, 200_000]


def test_linear_median_is_exact_for_flat_bins():
    assert nc_2020.interpolated_median([[10, 10, 10, 10]], [0, 10, 20, 30]).tolist() == [20.0]
    assert nc_2020.interpolated_median([[0, 30, 10, 0]], [0, 10, 20, 30]).tolist() == pytest.approx([16.667], abs=1e-3)


def test_pareto_median_recovers_a_pareto_distribution():
    # survival S(x) = (x / 1000) ** -1.5 above 1000; exact bin counts out of 1e6 households
    edges = [0, 1000, 1500, 2000, 3000, 5000]
    survival = [1.0] + [(x / 1000) ** -1.5 for x in edges[1:]]
    counts = [1e6 * (survival[i] - survival[i + 1]) for i in range(len(edges) - 1)] + [1e6 * survival[-1]]
    median = nc_2020.interpolated_median([counts], edges, pareto=True)[0]
    assert median == pytest.approx(1000 * 2 ** (1 / 1.5))
    assert nc_2020.interpolated_median([counts], edges)[0] > median  # linear overshoots on a falling density


def test_median_in_open_top_bin_raises():
    with pytest.raises(ValueError, match="open-ended"):
        nc_2020.interpolated_median([[1, 1, 5]], [0, 10, 20])


def test_empty_distribution_raises():
    with pytest.raises(ValueError, match="empty"):
        nc_2020.interpolated_median([[0, 0, 0]], [0, 10, 20])


def test_pareto_falls_back_to_linear_without_a_tail_or_lower_edge():
    # median bin [10, 20) has nobody above it, and a median in the bottom bin has lower edge 0
    for counts in ([[1, 5, 0]], [[6, 1, 1]]):
        linear = nc_2020.interpolated_median(counts, [0, 10, 20])
        assert nc_2020.interpolated_median(counts, [0, 10, 20], pareto=True).tolist() == linear.tolist()


def test_unit_weights_split_by_population_with_area_fallback():
    shares = pd.DataFrame({
        # block group ...0001 is split 3:1 by population; block group ...0002 is unpopulated, split by area
        "block": ["370010001001000", "370010001001001", "370010001002000", "370010001002000"],
        "district": [1, 2, 1, 2],
        "area": [5.0, 5.0, 1.0, 3.0],
        "share": [1.0, 1.0, 0.25, 0.75],
    })
    population = pd.Series([300, 100, 0], index=["370010001001000", "370010001001001", "370010001002000"])
    w = nc_2020.unit_weights(shares, population, 12).set_index(["unit", "district"])["weight"]
    assert w.to_dict() == {("370010001001", 1): 0.75, ("370010001001", 2): 0.25,
                           ("370010001002", 1): 0.25, ("370010001002", 2): 0.75}


def test_aggregate_conserves_totals_and_rejects_unit_mismatch():
    weights = pd.DataFrame({"unit": ["a", "a", "b"], "district": [1, 2, 2], "weight": [0.75, 0.25, 1.0]})
    values = pd.DataFrame({"x": [100.0, 40.0]}, index=["a", "b"])
    out = nc_2020.aggregate(values, weights)
    assert out["x"].to_dict() == {1: 75.0, 2: 65.0} and math.isclose(out["x"].sum(), values["x"].sum())
    with pytest.raises(ValueError, match="differ"):
        nc_2020.aggregate(values.drop(index="b"), weights)


@pytest.mark.skipif(not os.getenv("CENSUS_API"), reason="needs a Census API key")
def test_live_rebuild_matches_published_116th_districts():
    diff = nc_2020.sanity_check(os.environ["CENSUS_API"]).abs()
    shares = ["white", "black", "hispanic", "college", "moved_within_past_yr", "foreign_born", "unemployment", "poverty"]
    assert diff[shares].to_numpy().max() < 0.25
    assert diff["median_income"].max() < 1_000 and diff["median_age"].max() < 0.5
    assert diff["avg_household_size"].max() < 0.02
    assert np.isfinite(diff.to_numpy()).all()
