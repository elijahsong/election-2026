from pathlib import Path

import pandas as pd
import pytest
from conftest import COUNTS

import incumbency


def cage_rows(year: int, edits: dict | None = None) -> pd.DataFrame:
    """A synthetic CAGE House general election: every district has a running R incumbent and a D challenger.

    At-large seats are numbered 1, as in CAGE. edits maps (state, district) to replacement candidate rows.
    """
    edits = edits or {}
    rows = []
    for state, n in COUNTS[str(year)].items():
        for d in range(1, n + 1):
            race = edits.get((state, d), [("R", 1, f"INC {state}-{d}"), ("D", 0, f"CHAL {state}-{d}")])
            rows += [{"year": float(year), "state": state, "office": "H", "dist": float(d), "type": "G",
                      "party": p, "name_snyder": name, "inc": float(inc)} for p, inc, name in race]
    # rows the filters must drop: a governor race and a special election
    rows.append({"year": float(year), "state": "AK", "office": "G", "dist": None, "type": "G", "party": "D",
                 "name_snyder": "GOV", "inc": None})
    rows.append({"year": float(year), "state": "AK", "office": "H", "dist": 1.0, "type": "S", "party": "D",
                 "name_snyder": "SPECIAL", "inc": 1.0})
    return pd.DataFrame(rows)


@pytest.fixture
def one_year(monkeypatch):
    monkeypatch.setattr(incumbency, "HISTORICAL_YEARS", [2018])
    monkeypatch.setattr(incumbency, "HOUSE_OVERRIDES", {})


def classify_2018(edits):
    out = incumbency.build_house_historical(cage_rows(2018, edits), COUNTS)
    return out.set_index(["state", "district"])


def test_historical_codes_and_at_large(one_year):
    out = classify_2018({
        ("CA", 1): [("R", 0, "A"), ("D", 0, "B")],  # open
        ("CA", 2): [("D", 1, "INC"), ("R", 0, "C")],  # Democratic incumbent
        ("CA", 3): [("D", 1, "X"), ("R", 2, "Y")],  # incumbents of both parties
        ("CA", 4): [("R", 0, "N"), ("Other", 1, "W-I")],  # non-major-party incumbent
        ("CA", 5): [("R", 1, "P"), ("R", 1, "Q")],  # two incumbents of one party
    })
    assert len(out) == 435 and ("AK", 0) in out.index and ("AK", 1) not in out.index
    assert out.loc[("AK", 0), "inc_signed"] == -1  # the special-election row is ignored
    assert out.loc[("CA", 1), ["inc", "inc_signed", "status"]].tolist() == [0, 0, "open"]
    assert out.loc[("CA", 2), ["inc", "inc_signed", "incumbent_party"]].tolist() == [1, 1, "D"]
    assert out.loc[("CA", 3), "inc"] == 1 and pd.isna(out.loc[("CA", 3), "inc_signed"])
    assert out.loc[("CA", 3), "status"] == "review: incumbents of both parties"
    assert out.loc[("CA", 4), "status"] == "review: non-major-party incumbent"
    assert out.loc[("CA", 5), ["inc_signed", "incumbent_names"]].tolist() == [-1, "P; Q"]


def test_override_applies(one_year, monkeypatch):
    monkeypatch.setattr(incumbency, "HOUSE_OVERRIDES", {
        (2018, "CA", 3): (1, 1, "X", "Y wasn't an incumbent"),
        (2018, "CA", 4): (0, 0, "", "two-party clash coded as open"),
    })
    edits = {("CA", 3): [("D", 1, "X"), ("R", 1, "Y")], ("CA", 4): [("D", 1, "P"), ("R", 1, "Q")]}
    house = incumbency.build_house_historical(cage_rows(2018, edits), COUNTS)
    out = incumbency.apply_overrides(house, incumbency.HOUSE_OVERRIDES, incumbency.HOUSE_KEY)
    out = out.set_index(["state", "district"])
    assert out.loc[("CA", 3), ["inc", "inc_signed", "incumbent_party", "incumbent_names"]].tolist() == [1, 1, "D", "X"]
    assert out.loc[("CA", 4), ["inc", "inc_signed", "incumbent_party", "incumbent_names"]].tolist() == [0, 0, "", ""]
    assert out.loc[("CA", 3), "status"].startswith("override")


def test_override_must_match_a_row(one_year, monkeypatch):
    monkeypatch.setattr(incumbency, "HOUSE_OVERRIDES", {(2018, "CA", 99): (0, 0, "", "no such district")})
    with pytest.raises(ValueError, match="matches 0 rows"):
        incumbency.apply_overrides(incumbency.build_house_historical(cage_rows(2018), COUNTS),
                                   incumbency.HOUSE_OVERRIDES, incumbency.HOUSE_KEY)


def test_missing_district_raises(one_year):
    cage = cage_rows(2018)
    cage = cage[~((cage["state"] == "TX") & (cage["dist"] == 36))]
    with pytest.raises(ValueError, match=r"missing \[\('TX', 36\)\]"):
        incumbency.build_house_historical(cage, COUNTS)


def cook_csv(tmp_path, edits: dict) -> pd.DataFrame:
    """A Cook 2026 file in its real layout (five header names, six fields per row), parsed by read_cook_2026."""
    lines = ["Dist,Incumbent,Incumbent Party,2025 PVI,2026 PVI"]
    for state, n in COUNTS["2026"].items():
        for d in [0] if n == 1 else range(1, n + 1):
            name, party = edits.get((state, d), (f"Member {state}{d}", "R"))
            lines.append(f"{state}-{'AL' if d == 0 else f'{d:02d}'},{name},{party},R+5,R+5,1")
    path = tmp_path / "cook.csv"
    path.write_text("\n".join(lines) + "\n")
    return incumbency.read_cook_2026(path)


def test_2026_from_cook(tmp_path):
    cook = cook_csv(tmp_path, {("AK", 0): ("OPEN (Begich)", "R"), ("CA", 6): ("Kevin Kiley", "I"), ("CA", 7): ("Doris", "D")})
    out = incumbency.build_house_2026(cook, COUNTS).set_index(["state", "district"])
    assert len(out) == 435
    assert out.loc[("AK", 0), ["inc", "inc_signed", "incumbent_party", "status"]].tolist() == [0, 0, "", "open"]
    assert out.loc[("CA", 7), ["inc", "inc_signed"]].tolist() == [1, 1]
    assert out.loc[("CA", 6), "inc"] == 1 and pd.isna(out.loc[("CA", 6), "inc_signed"])


# Synthetic Senate classes: every state sits in two of the three, with 33 / 33 / 34 seats
STATES = sorted(incumbency.acs.STATE_FIPS.values())
CLASSES = {1: STATES[:33], 2: STATES[:16] + STATES[33:], 3: STATES[16:]}
SENATE_YEAR_CLASS = {y: incumbency.regular_class(y) for y in incumbency.HISTORICAL_YEARS}


def senate_cage(extra: list[dict] | None = None) -> pd.DataFrame:
    """Every regular Senate race 2006-2024, each with a running D incumbent, plus any extra candidate rows."""
    nextup = {1: 2030, 2: 2026, 3: 2028}  # any year with the right class works
    rows = [{"year": float(y), "state": st, "office": "S", "type": "G", "nextup": float(nextup[c]), "party": "D",
             "name_snyder": f"SEN {st}", "inc": 1.0} for y, c in SENATE_YEAR_CLASS.items() for st in CLASSES[c]]
    return pd.DataFrame(rows + (extra or []))


def senate_2026_csv(tmp_path, rows: list[tuple]) -> Path:
    path = tmp_path / "senate_2026.csv"
    lines = ["state,class,special,inc,party"] + [",".join(map(str, r)) for r in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def regular_2026(tmp_path, specials=()) -> Path:
    rows = [(st, 2, 0, 1, "R") for st in CLASSES[2]]
    rows[0] = (rows[0][0], 2, 0, 0, "")  # one open seat
    return senate_2026_csv(tmp_path, rows + list(specials))


def special(year, state, nextup, party="R", inc=1.0):
    return {"year": float(year), "state": state, "office": "S", "type": "S", "nextup": float(nextup), "party": party,
            "name_snyder": f"SPECIAL {state}", "inc": inc}


@pytest.fixture
def no_senate_fixes(monkeypatch):
    monkeypatch.setattr(incumbency, "SENATE_OVERRIDES", {})
    monkeypatch.setattr(incumbency, "NEXTUP_FIXES", {})
    monkeypatch.setattr(incumbency, "NON_NOVEMBER_SPECIALS", set())


def test_senate_classes_specials_and_2026(no_senate_fixes, tmp_path):
    class3_state = CLASSES[3][-1]
    extra = [special(2010, CLASSES[1][0], 2012),  # a November special for a class I seat
             special(2022, class3_state, 2022)]  # same-day special for a seat's last weeks: dropped
    races_2026 = incumbency.read_senate_2026(regular_2026(tmp_path, [(class3_state, 3, 1, 1, "R")]))
    out = incumbency.build_senate(senate_cage(extra), races_2026).set_index(["year", "state", "class"])
    assert out.loc[(2010, CLASSES[1][0], 1), ["special", "inc_signed"]].tolist() == [1, -1]
    assert out.loc[(2022, class3_state, 3), "special"] == 0  # only the full-term race remains
    assert out.loc[(2026, CLASSES[2][0], 2), ["inc", "inc_signed", "status"]].tolist() == [0, 0, "open"]
    assert out.loc[(2026, CLASSES[2][1], 2), ["inc", "inc_signed", "incumbent_party"]].tolist() == [1, -1, "R"]
    assert out.loc[(2026, class3_state, 3), "special"] == 1
    assert len(out) == sum(len(CLASSES[c]) for c in SENATE_YEAR_CLASS.values()) + 1 + 33 + 1


def test_senate_2026_party_must_match_inc(tmp_path):
    with pytest.raises(ValueError, match="party must be given exactly when inc is 1"):
        incumbency.read_senate_2026(senate_2026_csv(tmp_path, [("AK", 2, 0, 1, ""), ("AL", 2, 0, 0, "")]))


def test_senate_2026_must_be_class_ii(no_senate_fixes, tmp_path):
    rows = [(st, 2, 0, 1, "R") for st in CLASSES[2][1:]]  # one class II state missing
    with pytest.raises(ValueError, match="exactly the class II states"):
        incumbency.build_senate(senate_cage(), incumbency.read_senate_2026(senate_2026_csv(tmp_path, rows)))


def test_senate_class_membership_must_be_consistent(no_senate_fixes, tmp_path):
    cage = senate_cage()
    moved = (cage["year"] == 2012) & (cage["state"] == CLASSES[1][0])
    cage.loc[moved, "state"] = CLASSES[3][-1]  # a class I race in a state that isn't class I
    with pytest.raises(ValueError, match="Class 1 states differ"):
        incumbency.build_senate(cage, incumbency.read_senate_2026(regular_2026(tmp_path)))


def test_appointed_incumbent_is_open(no_senate_fixes, tmp_path):
    state = CLASSES[1][1]
    extra = [special(2010, state, 2012, party="D", inc=3.0)]  # an appointed senator running in a special
    out = incumbency.build_senate(senate_cage(extra), incumbency.read_senate_2026(regular_2026(tmp_path)))
    row = out.set_index(["year", "state", "class"]).loc[(2010, state, 1)]
    assert row[["special", "inc", "inc_signed", "incumbent_party"]].tolist() == [1, 0, 0, ""]
    assert row["status"] == f"open: appointed incumbent SPECIAL {state}"
