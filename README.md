# Election Prediction Project, 2026
2026 Midterm Election Forecasting Project

Data Sources:

Presidential Results on Districts:
* 2026 Dist on 2024 Pres: https://www.the-downballot.com/p/the-downballots-calculations-of-presidential
* https://www.the-downballot.com/p/data

U.S. House Results:
* on districts (1976-2024) https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IG0UN2
MIT Election Data and Science Lab, 2017, "U.S. House 1976–2024", https://doi.org/10.7910/DVN/IG0UN2, Harvard Dataverse, V15, UNF:6:cIJWG41N8qe3Kyf7oKuPdw== [fileUNF]

Generic Ballot:
https://github.com/fivethirtyeight/data/tree/master/congress-generic-ballot generic_topline_historical.csv contains a trendline for the generic ballot for previous election cycles going back to 1996.

https://web.archive.org/web/20250306062821/https://projects.fivethirtyeight.com/polls/data/generic_ballot_averages.csv

Find the missin gap, and the present


UTILITIES
dict: Cycle to Election Date, and Poll Cut-off Date (T-1)

generic_dem_pct
generic_rep_pct
generic_margin
generic_two_party_margin
generic_undecided_other_pct

Then compare two simple models:

Baseline: Raw D–R generic-ballot margin.

Alternative: Two-party-normalized generic-ballot margin.

4. An alternative: normalize to the two-party vote
You could also calculate:

$$
G_t^{2P}
=
100\frac{D_t-R_t}{D_t+R_t}
$$

This estimates the Democratic–Republican margin among respondents expressing a major-party preference.