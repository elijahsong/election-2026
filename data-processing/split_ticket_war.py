"""Split Ticket's Wins Above Replacement (WAR), House and Senate 2014-2024, matched to the repo's races and nominees.

    python data-processing/split_ticket_war.py      (after fec_finance.py)
        -> data/processed/split_ticket_war.csv          one row per WAR race, both chambers
        -> data/processed/split_ticket_war_people.csv   FEC IDs that belong to one person (ID links)
        -> data/interim/split_ticket_war_review.csv     every check that needed a decision
        -> data/processed/house/war_f.csv               House Stage 1's lagged WAR (src/congress/war.py), one row
                                                         per district-year 2006-2026; house_features.py joins it
                                                         into modeling.csv

war_f.csv: year, state, district, war_lag and war_lag_available. war_lag is congress.war's lag (0.75 / 0.25 of the
nominees' two most recent earlier House WAR years, centered, D positive), with the neutral 0 where no nominee has an
earlier WAR race (war_lag_available False), so no district is ever dropped from Stage 1 for lack of WAR. Production
House Model A uses it since 2026-11-03 (notebooks/house/stage1/01_reg_war.ipynb).

Source: data/raw/split-ticket-war/data-Eq.csv (sha256 pinned in RAW_SHA256), Split Ticket's Full WAR Database
(https://split-ticket.org/full-wins-above-replacement-war-database/, "last refresh May 3, 2026"). Columns Year,
Chamber, Geography, Democrat, Republican, WAR ("D+1.3" / "R+0.2"), Sortable (the same number, R positive).

What WAR is (Split Ticket's description; quotes and links in data-processing/README.md):
- The residual of a regression, fit on one cycle's two-way races, of the congressional margin minus the presidential
  margin on incumbency, lagged presidential partisanship and swing, a spending ratio and demographics. It is a
  property of the race (both candidates together), already net of incumbency and money.
- "D+3": the Democrat ran 3 points of margin ahead of that baseline. war_dem is D positive (+3 for "D+3", -3 for
  "R+3"), and equals -Sortable (checked).
- One vintage: 2018-2022 were recomputed with the 2024 method in January 2025 (NY-17 2022 was R+2.7 in the December
  2022 release, R+1.7 here). The 2014 rows are in this file but on no page of the site.

Matching (anything that needed a decision is in the review file):
- Races: House "ST-NN" / "ST-AL" (district 0). Senate: a state name, "Special" / "Spec" marking a special; the class
  comes from senate_fec.csv's (year, state, special). A special that was the state's only race that year is
  unlabeled (AZ 2020, HI 2014) and takes that race. OFF_CYCLE rows (AL's December 2017 special) have no race here.
- Nominees: the race's D and R slots in house_fec.csv / senate_fec.csv (FEC candidate IDs from fec_finance.py), each
  verified by name with fec_finance.match_name, or by NAME_OVERRIDES (checked by hand). Surname-only and fuzzy
  matches are accepted (same race, same party slot) and listed in the review file.
- WAR's "Democrat" column is the main anti-Republican candidate. Where that is an independent or third-party
  candidate rather than our D nominee (Orman KS 2014, Harrington AR 2020, McMullin UT 2022, Mund ND-AL 2022, Osborn NE
  2024), dem_slot is "not_dem_nominee" and neither side is usable: the WAR compares the Republican with an
  independent, not a Democrat. King and Sanders fill our D slot, so their races are D-vs-R here; Galvin (AK-AL 2018,
  2020) and Gross (AK 2020) were the Democratic nominees.
- Usability is per side. {side}_usable: that nominee's name is verified and has an FEC ID, and the race is D vs R.
  A candidate's WAR history uses only their usable races (src/congress/war.py), so an incumbent keeps a race whose
  opponent filed no FEC report, and a Republican keeps a Louisiana jungle race whose Democrats WAR combined.

Identity across years: one person, one person_id.
- person_id is the FEC candidate ID, except that IDs of one person are linked (split_ticket_war_people.csv): within a
  chamber and state, nominee slots (2006-2026, house_fec.csv / senate_fec.csv) with the same surname and first given
  name under different IDs (a candidate who re-registered, e.g. Max Rose, NY-11 2018-2020 and 2022), where at least
  one of the IDs has a usable WAR race. person_id is the group's earliest ID. Every group was checked by hand on
  2026-11-03 (listed in the review file): the same person each time, except the DISTINCT_PEOPLE IDs (a father and son
  twice, two James O'Donnells), which are never linked.
- Also reviewed, none changing a match: an ID whose surname differs across years (spellings, a married name: Watson
  Coleman, Dean Cunnane), an ID in the D slot one year and the R slot another (Van Drew, NJ-2: his lagged WAR flips
  sign), and House IDs in different districts across years (redistricting; kept).
"""

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "src")]
from fec_finance import match_name, name_parts

RAW = ROOT / "data/raw/split-ticket-war/data-Eq.csv"
RAW_SHA256 = "b9fbac80866a324feecec80df4fd35f61726d72e6122f024dbc5331213693c86"
HOUSE_FEC = ROOT / "data/processed/house_fec.csv"
SENATE_FEC = ROOT / "data/processed/senate_fec.csv"
OUT = ROOT / "data/processed/split_ticket_war.csv"
PEOPLE = ROOT / "data/processed/split_ticket_war_people.csv"
REVIEW = ROOT / "data/interim/split_ticket_war_review.csv"
WAR_F = ROOT / "data/processed/house/war_f.csv"
SOURCE = "Split Ticket Full WAR Database (data-Eq.csv, refresh 2026-05-03)"

STATE_CODES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
    "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
    "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA",
    "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}
# WAR rows for elections outside the repo's race data (no November race): kept in the output, never usable
OFF_CYCLE = {(2018, "Alabama"): "Jones vs Moore, the December 12, 2017 special, filed under 2018"}
# Name pairs match_name can't verify, checked by hand: (year, chamber, geography, side) -> why they're one person
NAME_OVERRIDES = {
    (2016, "House", "CT-05", "dem"): "Elizabeth Esty; MEDSL spells ETSY",
    (2016, "House", "OH-05", "dem"): "James Neu Jr.; WAR has the name reversed (Neu James)",
    (2016, "House", "AL-06", "dem"): "David Putman; MEDSL spells PUTNAM",
    (2022, "House", "WY-AL", "dem"): "Lynnette Grey Bull; MEDSL writes GREYBULL",
    (2024, "Senate", "Mississippi", "dem"): "Ty Pinkins; WAR misspells Pickens",
    (2016, "House", "NC-01", "gop"): "H. Powell Dew Jr.; WAR writes Dew Powell",
    (2020, "House", "GA-04", "gop"): "Johsie Cruz Ezammudeen; WAR writes Joshie Cruz",
}
# FEC IDs never linked to another ID with the same surname and first name: different people, checked by hand
DISTINCT_PEOPLE = {
    "H0NJ10026": "Donald M. Payne (Sr.), NJ-10 to 2012; his son Donald M. Payne Jr. is H2NJ10154",
    "H2CA42023": "Duncan Hunter (Sr.), CA-52 2006; his son Duncan D. Hunter is H8CA52052",
    "H4NY27118": "James D. O'Donnell, the NY-27 Democrat of 2014, not James O'Donnell, the NY-18 Republican of 2018",
}
GOOD_MATCHES = {"full", "variant", "surname", "fuzzy", "override"}
SLOT_COLUMNS = ["dem_name", "dem_candidate_id", "gop_name", "gop_candidate_id", "ind_name"]


def read_raw(path: Path = RAW) -> pd.DataFrame:
    """The raw file as is, plus war_dem (D positive) parsed from the WAR label and checked against Sortable."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != RAW_SHA256:
        raise ValueError(f"{path} has sha256 {digest}, not the pinned {RAW_SHA256}: a new download? Re-check it")
    raw = pd.read_csv(path, encoding="utf-8-sig")
    parts = raw["WAR"].str.extract(r"^([DR])\+(\d+(?:\.\d+)?)$")
    if parts.isna().any().any():
        raise ValueError(f"Unparsed WAR labels: {raw.loc[parts[0].isna(), 'WAR'].tolist()}")
    war_dem = np.where(parts[0] == "D", 1.0, -1.0) * parts[1].astype(float)
    if not np.allclose(war_dem, -raw["Sortable"]):
        raise ValueError("WAR label and Sortable disagree")
    if raw.duplicated(["Year", "Chamber", "Geography"]).any():
        raise ValueError("Duplicate (Year, Chamber, Geography) rows")
    return raw.assign(war_dem=war_dem)


def race_keys(raw: pd.DataFrame, senate_fec: pd.DataFrame) -> pd.DataFrame:
    """raw plus state, district (House), special and class (Senate), and off_cycle."""
    out = raw.copy()
    house = out["Chamber"] == "House"
    geo = out.loc[house, "Geography"].str.extract(r"^([A-Z]{2})-(AL|\d{2})$")
    if geo.isna().any().any():
        raise ValueError(f"Unparsed House geographies: {out.loc[house & geo[0].isna(), 'Geography'].tolist()}")
    out.loc[house, "state"] = geo[0]
    out.loc[house, "district"] = geo[1].replace("AL", "0").astype(int)

    senate = out["Chamber"] == "Senate"
    out.loc[senate, "special"] = out.loc[senate, "Geography"].str.contains(r"\s(?:Special|Spec)$").astype(int)
    out.loc[senate, "state"] = out.loc[senate, "Geography"].str.replace(r"\s+(Special|Spec)$", "", regex=True) \
        .map(STATE_CODES)
    if out.loc[senate, "state"].isna().any():
        raise ValueError(f"Unknown Senate states: {out.loc[senate & out['state'].isna(), 'Geography'].tolist()}")
    classes = senate_fec[["year", "state", "special", "class"]].rename(columns={"year": "Year"})
    if classes.duplicated(["Year", "state", "special"]).any():
        raise ValueError("More than one Senate race per (year, state, special)")
    out = out.merge(classes, on=["Year", "state", "special"], how="left", validate="many_to_one")
    only = classes.groupby(["Year", "state"]).filter(lambda g: len(g) == 1).set_index(["Year", "state"])
    for i in out.index[(out["Chamber"] == "Senate") & out["class"].isna()]:
        key = (out.at[i, "Year"], out.at[i, "state"])
        if key in only.index:  # an unlabeled special, the state's only race that year
            out.loc[i, ["class", "special"]] = only.loc[key, ["class", "special"]].to_numpy()
    out["off_cycle"] = (out["Chamber"] == "Senate") & out["class"].isna()
    found = set(zip(out.loc[out["off_cycle"], "Year"], out.loc[out["off_cycle"], "Geography"]))
    if found != set(OFF_CYCLE):
        raise ValueError(f"Senate WAR rows with no race: {sorted(found - set(OFF_CYCLE))}")
    return out


def strip_label(name: str) -> tuple[str, str | None]:
    """("Angus King", "Ind") for "Angus King (Ind)": the source's label for a candidate outside the two parties."""
    m = re.fullmatch(r"(.*?)\s*\(([^)]*)\)\s*", name)
    return (m.group(1), m.group(2)) if m else (name, None)


def match_nominees(war: pd.DataFrame, house_fec: pd.DataFrame, senate_fec: pd.DataFrame) -> pd.DataFrame:
    """One row per WAR race, with our D and R nominees (names, FEC IDs), each name's match quality, dem_slot, and the
    per-side usability of the module docstring."""
    house = war[war["Chamber"] == "House"].merge(
        house_fec[["year", "state", "district", *SLOT_COLUMNS]].rename(columns={"year": "Year"}),
        on=["Year", "state", "district"], how="left", validate="one_to_one", indicator=True)
    senate = war[war["Chamber"] == "Senate"].merge(
        senate_fec[["year", "state", "class", *SLOT_COLUMNS]].rename(columns={"year": "Year"}),
        on=["Year", "state", "class"], how="left", validate="one_to_one", indicator=True)
    out = pd.concat([house, senate], ignore_index=True)
    no_race = (out["_merge"] != "both") & ~out["off_cycle"]
    if no_race.any():
        raise ValueError(f"WAR races with no nominee row: {out.loc[no_race, ['Year', 'Geography']].values.tolist()}")

    rows = []
    for r in out.itertuples(index=False):
        dem_war, dem_label = strip_label(r.Democrat)
        gop_war, _ = strip_label(r.Republican)
        match = {}
        for side, war_name, ours in [("dem", dem_war, r.dem_name), ("gop", gop_war, r.gop_name)]:
            if (r.Year, r.Chamber, r.Geography, side) in NAME_OVERRIDES:
                match[side] = "override"
            else:
                match[side] = "none" if pd.isna(ours) else (match_name(war_name, ours) or "none")
        if r.off_cycle:
            dem_slot = "off_cycle"
        elif dem_label and pd.isna(r.dem_name):
            dem_slot = "not_dem_nominee"  # an independent or third-party candidate, and we have no Democrat
        elif match["dem"] in GOOD_MATCHES:
            dem_slot = "dem_nominee"
        else:
            dem_slot = "unmatched"  # a different Democrat than ours, or WAR's combined Democrats (LA jungles)
        rows.append({"dem_war_name": dem_war, "dem_war_label": dem_label, "gop_war_name": gop_war,
                     "dem_match": match["dem"], "gop_match": match["gop"], "dem_slot": dem_slot})
    out = pd.concat([out.drop(columns="_merge"), pd.DataFrame(rows)], axis=1)
    d_vs_r = ~out["off_cycle"] & out["dem_name"].notna() & (out["dem_slot"] != "not_dem_nominee")
    out["dem_usable"] = d_vs_r & (out["dem_slot"] == "dem_nominee") & out["dem_candidate_id"].notna()
    out["gop_usable"] = d_vs_r & out["gop_match"].isin(GOOD_MATCHES) & out["gop_candidate_id"].notna()
    return out


def nominee_slots(house_fec: pd.DataFrame, senate_fec: pd.DataFrame) -> pd.DataFrame:
    """Every D and R nominee slot with an FEC ID, 2006-2026: chamber, year, state, side, candidate_id, name, surname
    and given (name_parts' surname words joined, and first given word)."""
    frames = []
    for chamber, fec in [("House", house_fec), ("Senate", senate_fec)]:
        for side in ["dem", "gop"]:
            slots = fec[["year", "state", f"{side}_name", f"{side}_candidate_id"]].dropna()
            frames.append(pd.DataFrame({"chamber": chamber, "year": slots["year"], "state": slots["state"],
                                        "side": side, "candidate_id": slots[f"{side}_candidate_id"],
                                        "name": slots[f"{side}_name"]}))
    slots = pd.concat(frames, ignore_index=True)
    parts = slots["name"].map(name_parts)
    slots["surname"] = parts.map(lambda p: "".join(p[0]))
    slots["given"] = parts.map(lambda p: p[1][0] if p[1] else "")
    return slots


def link_ids(slots: pd.DataFrame, war: pd.DataFrame) -> pd.DataFrame:
    """split_ticket_war_people.csv: one row per FEC ID in a linked group (module docstring): chamber, state, surname,
    given, candidate_id, person_id (the group's earliest ID), first_year, last_year, names."""
    with_war = set(war.loc[war["dem_usable"], "dem_candidate_id"]) | set(war.loc[war["gop_usable"],
                                                                                 "gop_candidate_id"])
    slots = slots[~slots["candidate_id"].isin(DISTINCT_PEOPLE)]
    per_id = slots.groupby(["chamber", "state", "surname", "given", "candidate_id"]).agg(
        first_year=("year", "min"), last_year=("year", "max"),
        names=("name", lambda s: " | ".join(sorted(set(s))))).reset_index()
    groups = []
    for _, g in per_id.groupby(["chamber", "state", "surname", "given"]):
        if g["candidate_id"].nunique() > 1 and g["candidate_id"].isin(with_war).any():
            g = g.sort_values(["first_year", "candidate_id"])
            groups.append(g.assign(person_id=g["candidate_id"].iloc[0]))
    columns = ["chamber", "state", "surname", "given", "candidate_id", "person_id", "first_year", "last_year", "names"]
    return pd.concat(groups, ignore_index=True)[columns] if groups else pd.DataFrame(columns=columns)


def review_table(war: pd.DataFrame, people: pd.DataFrame) -> pd.DataFrame:
    """Every check that needed a decision: unusable sides, accepted weak name matches, linked IDs, and the identity
    checks of the module docstring (name changes under one ID, party changes, House district changes)."""
    rows = []
    race = lambda r: f"{r.Year} {r.Chamber} {r.Geography}"
    for r in war.itertuples(index=False):
        for side, war_name, ours in [("dem", r.Democrat, r.dem_name), ("gop", r.Republican, r.gop_name)]:
            usable, quality, cid = getattr(r, f"{side}_usable"), getattr(r, f"{side}_match"), \
                getattr(r, f"{side}_candidate_id")
            if not usable:
                why = r.dem_slot if r.dem_slot in ("off_cycle", "not_dem_nominee") else \
                    ("no_fec_id" if quality in GOOD_MATCHES else f"name_{quality}")
                rows.append({"check": f"{side}_not_usable:{why}", "candidate_id": cid,
                             "detail": f"{race(r)}: WAR {war_name}; ours {ours}"})
            elif quality in ("surname", "fuzzy", "override"):
                note = NAME_OVERRIDES.get((r.Year, r.Chamber, r.Geography, side), "")
                rows.append({"check": f"name_{quality}", "candidate_id": cid,
                             "detail": f"{race(r)}: WAR {war_name}; ours {ours} {note}".strip()})
    for person, g in people.groupby("person_id"):
        rows.append({"check": "ids_linked", "candidate_id": ",".join(g["candidate_id"]),
                     "detail": f"{g['chamber'].iloc[0]} {g['state'].iloc[0]}: " + "; ".join(
                         f"{c} {a}-{b} {n}" for c, a, b, n in
                         zip(g["candidate_id"], g["first_year"], g["last_year"], g["names"]))})

    apps = appearances(war, people)
    apps["surname"] = apps["name"].map(lambda n: "".join(name_parts(n)[0]))
    for pid, g in apps.groupby("person_id"):
        history = "; ".join(f"{y} {geo} {s} {n}" for y, geo, s, n in zip(g["year"], g["geography"], g["side"],
                                                                          g["name"]))
        if g["surname"].nunique() > 1:
            rows.append({"check": "surname_differs", "candidate_id": pid, "detail": history})
        if g["side"].nunique() > 1:
            rows.append({"check": "party_change", "candidate_id": pid, "detail": history})
        if (g["chamber"] == "House").all() and g["geography"].nunique() > 1:
            rows.append({"check": "district_change", "candidate_id": pid, "detail": history})
        if g.duplicated(["year", "chamber"]).any():
            raise ValueError(f"{pid} has two usable WAR races in one year: {history}")
    return pd.DataFrame(rows, columns=["check", "candidate_id", "detail"])


def appearances(war: pd.DataFrame, people: pd.DataFrame) -> pd.DataFrame:
    """One row per usable (race, side): year, chamber, geography, side, candidate_id, person_id, name, war_dem."""
    links = dict(zip(people["candidate_id"], people["person_id"]))
    frames = []
    for side in ["dem", "gop"]:
        rows = war[war[f"{side}_usable"]]
        frames.append(pd.DataFrame({"year": rows["Year"], "chamber": rows["Chamber"], "geography": rows["Geography"],
                                    "side": side, "candidate_id": rows[f"{side}_candidate_id"],
                                    "name": rows[f"{side}_name"], "war_dem": rows["war_dem"]}))
    apps = pd.concat(frames, ignore_index=True)
    apps["person_id"] = apps["candidate_id"].map(lambda c: links.get(c, c))
    return apps


def main() -> None:
    house_fec = pd.read_csv(HOUSE_FEC, low_memory=False)
    senate_fec = pd.read_csv(SENATE_FEC, low_memory=False)
    war = match_nominees(race_keys(read_raw(), senate_fec), house_fec, senate_fec)
    people = link_ids(nominee_slots(house_fec, senate_fec), war)
    review = review_table(war, people)

    links = dict(zip(people["candidate_id"], people["person_id"]))
    out = war.rename(columns={"Year": "year", "Chamber": "chamber", "Geography": "geography", "WAR": "war_label",
                              "Democrat": "war_democrat", "Republican": "war_republican"})
    for side in ["dem", "gop"]:
        out[f"{side}_person_id"] = out[f"{side}_candidate_id"].map(lambda c: links.get(c, c))
    out["district"] = out["district"].astype("Int64")
    out[["special", "class"]] = out[["special", "class"]].astype("Int64")
    out["source"] = SOURCE
    out = out[["year", "chamber", "state", "district", "class", "special", "geography", "war_label", "war_dem",
               "war_democrat", "war_republican", "dem_name", "dem_candidate_id", "dem_person_id", "gop_name",
               "gop_candidate_id", "gop_person_id", "dem_match", "gop_match", "dem_slot", "off_cycle", "dem_usable",
               "gop_usable", "source"]].sort_values(["chamber", "year", "state", "district", "class"],
                                                    ignore_index=True)
    REVIEW.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    people.to_csv(PEOPLE, index=False)
    review.to_csv(REVIEW, index=False)

    print(f"{OUT.relative_to(ROOT)}: {len(out)} WAR races")
    print(out.groupby(["chamber", "year"]).agg(races=("war_dem", "size"), dem_usable=("dem_usable", "sum"),
                                               gop_usable=("gop_usable", "sum")).to_string())
    print(f"{PEOPLE.relative_to(ROOT)}: {people['person_id'].nunique()} people with linked IDs ({len(people)} IDs)")
    print(f"{REVIEW.relative_to(ROOT)}: {len(review)} rows")

    from congress import war  # reads the files just written
    war_f = war.house_war_features(war.chamber_lags("House"))
    war_f.to_csv(WAR_F, index=False)
    print(f"{WAR_F.relative_to(ROOT)}: {len(war_f)} district-years, {int(war_f[war.AVAILABLE].sum())} with a lag")
    print(review["check"].value_counts().to_string())


if __name__ == "__main__":
    main()
