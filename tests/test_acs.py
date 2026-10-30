import io
import os
import urllib.error

import pandas as pd
import pytest
from conftest import COUNTS

import acs
import nc_2020


def test_vintage_labels():
    assert acs.acs_vintage_label(2020) == "2016-2020_5yr"
    assert acs.acs_vintage_label(2018) == "2018_1yr"
    assert acs.acs_vintage_label(2006) == "2007_1yr"


@pytest.mark.parametrize("year", [2026, 2000, 2019])
def test_unsupported_years_raise(year):
    with pytest.raises(ValueError):
        acs.build_acs_year(year, COUNTS)


def test_reference_counts_are_valid():
    for year in COUNTS:
        acs.normalize_counts(COUNTS, int(year))


def test_build_2018(fake_api, tmp_path):
    fake_api(2018)
    df = acs.build_acs_year(2018, COUNTS, tmp_path / "acs.csv")
    assert list(df.columns) == acs.COLUMNS
    assert len(df) == 435 and df["state"].nunique() == 50
    assert not df["state"].isin(["DC", "PR"]).any()
    assert df.loc[df["state"] == "WY", "district"].tolist() == [0]
    assert df.loc[df["state"] == "CA", "district"].tolist() == list(range(1, 54))

    row = df.iloc[0]
    assert row["acs_vintage"] == "2018_1yr"
    assert (row["white"], row["black"], row["hispanic"]) == (60.0, 13.0, 20.0)
    assert row["college"] == 33.33
    assert row["moved_within_past_yr"] == 13.04
    assert (row["foreign_born"], row["unemployment"], row["poverty"]) == (10.0, 5.0, 12.0)
    assert (row["median_income"], row["median_age"], row["avg_household_size"]) == (65_432, 38.4, 2.55)
    assert row["housing_burden"] == 44.44

    on_disk = pd.read_csv(tmp_path / "acs.csv")
    assert list(on_disk.columns) == acs.COLUMNS and len(on_disk) == 435


def test_pre_2012_uses_c23001(fake_api, tmp_path):
    fake_api(2010)
    df = acs.build_acs_year(2010, COUNTS, tmp_path / "acs.csv")
    # (10 * 1500 + 4 * 375) / (10 * 30000 + 4 * 7500) = 16500 / 330000
    assert (df["unemployment"] == 5.0).all()
    assert acs.ACS_FIELDS[2010]["unemployment"] is acs.UNEMPLOYMENT_C23001
    assert acs.ACS_FIELDS[2012]["unemployment"] is acs.UNEMPLOYMENT_B23025


def tables(year: int) -> set[str]:
    return {v.split("_")[0] for v in acs.field_variables(year)}


def test_one_year_reads_collapsed_tables_and_five_year_detailed():
    assert {"C03002", "C15002", "C05002", "C17001"} <= tables(2006) and not tables(2006) & {"B03002", "B15002"}
    assert {"B03002", "B15002", "B05002", "B17001"} <= tables(2020) and not any(t.startswith("C") for t in tables(2020))


def test_withheld_table_is_reported_as_null(fake_api, tmp_path):
    for row in fake_api(2006)["rows"][:2]:
        row["C03002_001E"] = None
    with pytest.raises(ValueError, match=r"AL-1: C03002_001E=null"):
        acs.build_acs_year(2006, COUNTS, tmp_path / "acs.csv")


def test_2020_replaces_north_carolina_with_rebuild(fake_api, monkeypatch, tmp_path):
    fake_api(2020)
    rebuilt = acs.transform(acs.fetch_acs(2020, "test-key"), 2020).head(13)  # 13 valid-looking rows to relabel
    rebuilt = rebuilt.assign(state="NC", district=range(1, 14), white=12.34)
    monkeypatch.setattr(nc_2020, "rebuild", lambda key: rebuilt)
    df = acs.build_acs_year(2020, COUNTS, tmp_path / "acs.csv")
    nc = df[df["state"] == "NC"]
    assert len(df) == 435 and nc["district"].tolist() == list(range(1, 14))
    assert (nc["white"] == 12.34).all()  # the rebuild, not Census's 116th-map rows
    assert (df["acs_vintage"] == "2016-2020_5yr").all()


def test_atomic_write_keeps_readable_permissions(tmp_path):
    path = tmp_path / "out.csv"
    acs.write_csv_atomic(pd.DataFrame({"a": [1]}), path)
    umask = os.umask(0)
    os.umask(umask)
    assert path.stat().st_mode & 0o777 == 0o666 & ~umask


@pytest.mark.parametrize("bad", ["-666666666", None, "N"])
def test_missing_estimate_raises(fake_api, tmp_path, bad):
    fake_api(2018)["rows"][5]["B19013_001E"] = bad
    with pytest.raises(ValueError, match="missing or annotated"):
        acs.build_acs_year(2018, COUNTS, tmp_path / "acs.csv")


def test_missing_district_raises(fake_api, tmp_path):
    state = fake_api(2018)
    state["rows"] = [r for r in state["rows"] if not (r["state"] == "06" and r["congressional district"] == "53")]
    with pytest.raises(ValueError, match="CA: expected"):
        acs.build_acs_year(2018, COUNTS, tmp_path / "acs.csv")


def test_extra_district_raises(fake_api, tmp_path):
    state = fake_api(2018)
    state["rows"].append({**state["rows"][0], "congressional district": "99"})
    with pytest.raises(ValueError, match="AL: expected"):
        acs.build_acs_year(2018, COUNTS, tmp_path / "acs.csv")


def test_bad_counts_raise():
    counts = {"2018": {**COUNTS["2018"], "CA": 52}}
    with pytest.raises(ValueError, match="total 434"):
        acs.normalize_counts(counts, 2018)
    counts = {"2018": {**COUNTS["2018"], "DC": 1}}
    with pytest.raises(ValueError, match="exactly the 50 states"):
        acs.normalize_counts(counts, 2018)


def test_wrong_congress_raises(fake_api, tmp_path):
    state = fake_api(2018)
    state["rows"][0]["NAME"] = state["rows"][0]["NAME"].replace("116th", "115th")
    with pytest.raises(ValueError, match="expects 116th-Congress"):
        acs.build_acs_year(2018, COUNTS, tmp_path / "acs.csv")


def test_label_mismatch_raises(fake_api, tmp_path):
    fake_api(2018)["labels"]["C05002_008E"] = "Total!!Native"
    with pytest.raises(ValueError, match="C05002_008E"):
        acs.build_acs_year(2018, COUNTS, tmp_path / "acs.csv")


def test_update_is_idempotent_and_atomic(fake_api, tmp_path):
    path = tmp_path / "acs.csv"
    fake_api(2022)
    acs.build_acs_year(2022, COUNTS, path)
    fake_api(2018)
    acs.build_acs_year(2018, COUNTS, path)
    first = path.read_bytes()

    acs.build_acs_year(2018, COUNTS, path)
    assert path.read_bytes() == first
    df = pd.read_csv(path)
    assert df["year"].value_counts().to_dict() == {2018: 435, 2022: 435}
    assert df.equals(df.sort_values(["year", "state", "district"], ignore_index=True))

    fake_api(2018)["rows"][0]["C03002_001E"] = "-999999999"
    with pytest.raises(ValueError):
        acs.build_acs_year(2018, COUNTS, path)
    assert path.read_bytes() == first
    assert [p.name for p in tmp_path.iterdir()] == ["acs.csv"]


def test_missing_key_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("CENSUS_API", raising=False)
    with pytest.raises(RuntimeError, match="CENSUS_API"):
        acs.build_acs_year(2018, COUNTS, tmp_path / "acs.csv")


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def http_error(code):
    return urllib.error.HTTPError("https://api.census.gov", code, "error", {}, io.BytesIO(b"error body"))


def test_get_json_retries_transient_errors(monkeypatch):
    calls = []

    def urlopen(url, timeout):
        calls.append(url)
        if len(calls) < 3:
            raise http_error(503) if len(calls) == 1 else TimeoutError()
        return FakeResponse(b'[["NAME"]]')

    monkeypatch.setattr(acs.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(acs.time, "sleep", lambda s: None)
    assert acs._get_json("https://api.census.gov/x?key=secret") == [["NAME"]]
    assert len(calls) == 3


def test_get_json_does_not_retry_client_errors_and_redacts_key(monkeypatch):
    calls = []

    def urlopen(url, timeout):
        calls.append(url)
        raise http_error(400)

    monkeypatch.setattr(acs.urllib.request, "urlopen", urlopen)
    with pytest.raises(RuntimeError, match="HTTP 400") as err:
        acs._get_json("https://api.census.gov/x?key=secret")
    assert len(calls) == 1 and "secret" not in str(err.value)


def test_get_json_rejects_html(monkeypatch):
    monkeypatch.setattr(acs.urllib.request, "urlopen", lambda url, timeout: FakeResponse(b"<html>Missing Key</html>"))
    with pytest.raises(RuntimeError, match="non-JSON"):
        acs._get_json("https://api.census.gov/x")


@pytest.mark.skipif(not os.getenv("CENSUS_API"), reason="needs a Census API key")
def test_live_2024(tmp_path):
    df = acs.build_acs_year(2024, COUNTS, tmp_path / "acs.csv")
    assert len(df) == 435
