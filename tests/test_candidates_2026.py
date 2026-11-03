import candidates_2026
import fec_finance
import pandas as pd


def test_registry_classifies_lineups_and_keeps_ballot_identity(monkeypatch):
    # SYNTHETIC nominees: an independent GOP proxy, a top-two D vs D, an RCV race, and a D vs third party
    nominees = pd.DataFrame([
        {"year": 2026, "state": "CA", "district": 6, "party": "other", "party_label": "Independent", "name": "Kevin Kiley"},
        {"year": 2026, "state": "CA", "district": 6, "party": "D", "party_label": "Democratic", "name": "Richard Pan"},
        {"year": 2026, "state": "CA", "district": 11, "party": "D", "party_label": "Democratic", "name": "Scott Wiener"},
        {"year": 2026, "state": "CA", "district": 11, "party": "D", "party_label": "Democratic", "name": "Connie Chan"},
        {"year": 2026, "state": "ME", "district": 2, "party": "D", "party_label": "Democratic", "name": "Jordan Wood"},
        {"year": 2026, "state": "ME", "district": 2, "party": "R", "party_label": "Republican", "name": "Paul LePage"},
        {"year": 2026, "state": "MA", "district": 1, "party": "D", "party_label": "Democratic", "name": "Richard Neal"},
        {"year": 2026, "state": "MA", "district": 1, "party": "other", "party_label": "Independent", "name": "Nadia Milleron"},
    ]).assign(source="synthetic")
    fec = pd.DataFrame({"year": 2026, "office": "H", "state": ["CA", "CA"], "district": [11, 11],
                        "candidate_id": ["H1", "H2"], "fec_name": ["WIENER, SCOTT", "CHAN, CONNIE"],
                        "receipts": [1.0, 1.0], "coverage_end_date": pd.to_datetime(["2026-10-14"] * 2)})
    for field in fec_finance.FINANCE:
        fec[field] = fec.get(field, 0.0)
    slots = pd.DataFrame({"year": [2026], "state": ["ME"], "district": [2], "dem_name": ["Jordan Wood"],
                          "dem_candidate_id": ["H9"], "gop_name": ["Paul LePage"], "gop_candidate_id": [None]})
    monkeypatch.setattr(fec_finance, "INDEPENDENTS_AS_GOP", {("house", 2026, "CA", 6): "Kevin Kiley"})
    registry = candidates_2026.build_registry(nominees, fec, slots, "2026-10-30", at_large=set())
    by = registry.set_index("candidate_name")
    assert by.loc["Kevin Kiley", ["party", "party_label", "model_party"]].tolist() == ["other", "Independent", "R"]
    assert by.loc["Kevin Kiley", "data_flags"] == "no_fec_record;independent_as_gop"
    assert by.loc["Kevin Kiley", "race_type"] == "dr" and by.loc["Kevin Kiley", "election_method"] == "top_two"
    assert by.loc["Scott Wiener", ["race_type", "candidate_id"]].tolist() == ["dem_vs_dem", "H1"]
    assert by.loc["Paul LePage", "race_type"] == "dr_rcv" and by.loc["Jordan Wood", "candidate_id"] == "H9"
    assert by.loc["Nadia Milleron", ["race_type", "model_party"]].tolist() == ["dem_vs_3p", "other"]
    assert (registry["ballot_status"] == "listed_general_election").all() and (registry["as_of_date"] == "2026-10-30").all()
