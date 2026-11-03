"""Download the raw sources of the Senate pipeline into data/raw/, once, and check every file's sha256.

  heideman/v1.0.0/        Heideman, U.S. county-level election results (v1.0.0 GitHub release and tag files)
  medsl_pres_state/       MEDSL U.S. President 1976-2024 (Harvard Dataverse doi:10.7910/DVN/42MVDX)
  fec_federal_elections/  FEC "Federal Elections" 2006-2022 (official results; 2024 isn't published)
  la_sos_parish/          Louisiana SOS official parish-level U.S. Senate results, 2014, 2016 and 2020
  kansas_city_mo/         Kansas City (MO) Election Board official results, 2006-2022, and OpenElections' 2024 MO
                          precinct results: the Kansas City votes Heideman leaves out of Missouri's counties
  openelections_senate/   OpenElections county, town and precinct results replacing Heideman's ME 2012, NH 2016,
                          NY 2016, SC 2016, DE 2018, NJ 2018 and SC 2022 Senate rows, and the RI and ME ballots that
                          belong to no county
  idaho_sos_2022/         Idaho SOS 2022 general election canvass (Power County, missing from Heideman)
  arkansas_sos_2022/      Arkansas SOS 2022 general election results (Phillips County, missing from Heideman)
  All-US-Senate-polls-2006-2024/  Jack Whitcomb's RealClearPolling Senate polls 2006-2024 (GitHub, pinned commit)

These collections were saved by hand (Harvard Dataverse asks for a guestbook entry; official sites refuse scripts),
so their files are only checked:

  medsl_senate/           MEDSL U.S. Senate statewide 1976-2024 (doi:10.7910/DVN/PEJ5QU)
  medsl_pres_county/      MEDSL County Presidential Returns 2000-2024 (doi:10.7910/DVN/VOQCHQ)
  senate_manual/          hand-entered official county results for rounds no downloadable source has
  mn-2018-senate-special/ Minnesota SOS 2018 general election county results files
  ms-2018-senate-special-r1/  Mississippi 2018 general election precinct results
  ga_2020_senate_special_r1/  Georgia 2020 general election precinct results
  maine_sos_2024/         Maine SOS 2024 U.S. Senate results by municipality (UOCAVA ballots)
  senate_ratings_historical/  JHK Forecasts' forecast archive (supplied as output.csv; download URL not recorded)

A file already present is never downloaded again or changed. Each collection's SOURCE.md (written by hand) gives the
citation, license, coverage and use.
"""

import hashlib
import os
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw"

HEIDEMAN_RELEASE = "https://github.com/pmheideman/us-county-election-results/releases/download/v1.0.0"
HEIDEMAN_TAG = "https://raw.githubusercontent.com/pmheideman/us-county-election-results/v1.0.0"
DATAVERSE = "https://dataverse.harvard.edu/api/access/datafile"
LA_PORTAL = "https://voterportal.sos.la.gov/ElectionResults/ElectionResults/Data?blob="
KCEB = "https://www.kceb.org/useruploads"
# Louisiana's U.S. Senate race IDs, from data/raw/la_sos/<date>_RacesCandidates_Multiparish.json
LA_SENATE_RACES = {"20141104": "47638", "20141206": "50494", "20161108": "53905", "20161210": "54490",
                   "20201103": "59569"}

# collection -> {file name: (url or None for a hand-saved file, sha256)}
SOURCES = {
    "heideman/v1.0.0": {
        **{name: (f"{HEIDEMAN_RELEASE}/{name}", sha) for name, sha in [
            ("us_county_results_long.parquet", "83b6da0d93958720f5d5822fc440f477791ef13757e66f3179a3653022fbca8a"),
            ("us_county_results_summary.csv", "6b57de978c6cdda6446adce3812557b8a63d1577693c535ebb10d85ebbde8bf6"),
            ("us_county_results_gaps.csv", "fbe37f4705ce7fcdee8a6519ae821302f54b61d31ec0ccf00258229af14a037e"),
            ("us_county_results_no_ballot.csv", "ad8633be68d10ed7b7d57415c06f2eba97393848327dd337a348533fcd8583c3"),
            ("SOURCES.csv", "0870e3ac621c205d20246a2873943aa49478b62be590c4b5ee34563c313c4c45"),
            ("data_corrections_log.csv", "7a955a65faff010b51d28077dd9d566158b2edc753c51ca4bdc7af37dca11283"),
            ("DATA_DICTIONARY.md", "b2f3da8c961d2f095145e490633538758b048c264629bd1ef0bd2e39d2e10d33"),
            ("CHANGELOG.md", "621ce4446759be519c470f850ddae87ac5f3b5bf2fcd7c1c4b6f5393590a8099"),
            ("SHA256SUMS.txt", None),
        ]},
        **{name: (f"{HEIDEMAN_TAG}/{name}", None) for name in ["LICENSE", "LICENSE-DATA.md", "CITATION.cff",
                                                                 "README.md"]},
    },
    "medsl_pres_state": {
        "1976-2024-president.csv": (f"{DATAVERSE}/13887042?format=original", None),
        "codebook-us-president-1976-2020.md": (f"{DATAVERSE}/7241480", None),
        "sources-president.tab": (f"{DATAVERSE}/4304791", None),
    },
    "fec_federal_elections": {
        name: (f"https://www.fec.gov/documents/{doc}/{name}", None) for doc, name in [
            (1642, "federalelections2006.xls"), (1666, "federalelections2008.xls"), (1675, "federalelections2010.xls"),
            (1691, "federalelections2012.xls"), (1700, "federalelections2014.xls"), (1890, "federalelections2016.xlsx"),
            (2706, "federalelections2018.xlsx"), (4228, "federalelections2020.xlsx"),
            (5676, "federalelections2022.xlsx"),
        ]
    },
    "la_sos_parish": {
        f"{date}_VotesRaceByParish_{race}.json": (f"{LA_PORTAL}{date}/VotesRaceByParish/Votes_{race}.htm", None)
        for date, race in LA_SENATE_RACES.items()
    },
    "medsl_senate": {name: (None, None) for name in
                     ["1976-2024-senate-state.csv", "codebook-us-senate-1976–2024.md", "sources-senate.csv"]},
    "medsl_pres_county": {name: (None, None) for name in
                          ["countypres_2000-2024.csv", "County Presidential Returns 2000-2024.md",
                           "sources-president.csv"]},
    "senate_manual": {
        "ga2008-senate-r1.csv": (None, "937f5342946fa969f33e2f8c53d0675302eff173f10b28879f2ed9e6a1902e7d"),
        "ms2008-senate.csv": (None, "8a862679224492d39a737e25b1f2f016435eddfe5e967719cc7a09e3d89f27ee"),
        "de2010-senate.csv": (None, "92de37a818218280be1cf6c9bd0f1a35e50815ba12765186a32e51e7d000715a"),
        "ny2010-senate-special.csv": (None, "3518c301d49e13dbd24a480fd4e94501973eb74793388b1fdf7eb30068e8ffc1"),
        "wv-2010-senate-dem.csv": (None, "40752d9aa8780ecf2afe38547fae988d2b4c7949ad2ebd589ac7e07c8c6b7f44"),
        "wv-2010-senate-gop.csv": (None, "0c57df4f5830ce400beb89ca1cc2340d7feaa178cd04d97ce9154bfd3b4a01f6"),
        "wv-2010-senate-3p-mountain.csv": (None, "4b003f80bfbe4d9b32c5698ad1e7820659e4e7df01fdd510c0954f95c7d9e8c0"),
        "wv-2010-senate-3p-constitution.csv": (None,
                                               "a0c6677db1a2acc95a10459f70c6830b8cf6b13382bd6c219441972d79ebdffc"),
        "sc-2014-senate-special-revised.csv": (None, "9e399e4425d13b063b4841c01271b6bb79f208ef350c837256b3c14ac1485600"),
        "ok-2014-senate-special.csv": (None, "cdd3c66d225682e7f12c4bd6f773e3d2e35746c7b103a12eda5fc144cfb65344"),
        "az-2020-senate-special.csv": (None, "6568d1ee4bd5745842419ff5e53111042119e84f3b87d3d4313058cdd18e565e"),
        "ga-2022-senate-r1.csv": (None, "06985dcc15633cf16ce48e82b6e28d07830b39da5135ab26b3057c4e9f90bec6"),
        "ok-2022-senate-special.csv": (None, "6369ef406bbbe109aef089b45257f2a2d4fbbd2e7ef697f442fa08bb5a986623"),
        "ne-2024-senate-special.csv": (None, "3c90dd15d4e9dfe61aa0a1aeb017abe33dbcddc2839b8164bc16d16dc5f26d99"),
    },
    "mn-2018-senate-special": {
        "mn_2018_county_results_all.txt": (None, "7bd5cc5b27a84444ceaa06e43816552e54c9856c2025de4f594634f42b89bf1e"),
        "county_index.txt": (None, "4b266c5b438c63fc567f2ad8909a21f503c2ea9dd9aaa59edab2fe211d7de337"),
        "overall_Index.txt": (None, "a5f81d69643424e8fff028ee0603b20746cf6b784b7af4df780b82dcd8670acb"),
    },
    "ga_2020_senate_special_r1": {
        "20201103__ga__general.txt": (None, "c432c02a5a4f088efe8e34141447c861c1bf163b2c50007b70e6a9789cedd937"),
    },
    "ms-2018-senate-special-r1": {
        "20181106__ms__general__precinct.txt": (None,
                                                "56ba2a64b34ddeef569b2c3d0efc665c050c744c998fa744b17c6af2e0f4bdcb"),
    },
    "maine_sos_2024": {
        "US Senator FINAL-Corrected 20241205.csv": (None,
                                                    "4bfa1e31af799af291d63d7b936ac56606a3bda2e48d97483fefd0bc39df112e"),
    },
    "kansas_city_mo": {
        **{Path(path).name: (f"{KCEB}/{path}", None) for path in [
            "Canvass/official_results_11_06.pdf",
            "Past_Election_Results/11_02_10_SUMMARY_REPORT.pdf",
            "Current_Results/ELECTION_SUMMARY_OFFICIAL_AMENDED_11-12.pdf",
            "11816Pres/results/Election_Summary_Official_11-16.pdf",
            "11_6_18/11_6_18_Results/Election_Summary_Official_11-18.pdf",
            "Election_Summary_Official_11-8-22.pdf",
        ]},
        "20241105__mo__general__precinct.csv": (
            "https://raw.githubusercontent.com/openelections/openelections-data-mo/master/2024/"
            "20241105__mo__general__precinct.csv", None),
    },
    "openelections_senate": {
        Path(path).name: (f"https://raw.githubusercontent.com/openelections/openelections-data-{path}", None)
        for path in [
            "nh/master/2016/20161108__nh__general__town.csv",
            "ny/master/2016/20161108__ny__general__county.csv",
            "sc/master/2016/20161108__sc__general__precinct.csv",
            "de/master/2018/20181106__de__general__precinct.csv",
            "nj/master/2018/20181106__nj__general__county.csv",
            "sc/master/2022/20221108__sc__general__precinct.csv",
            "ri/master/2020/20201103__ri__general__precinct.csv",
            "ri/master/2024/20241105__ri__general__precinct.csv",
            "me/master/2012/20121106__me__general__town.csv",
            "me/master/2018/20181106__me__general__town.csv",
            "me/master/2020/20201103__me__general__precinct.csv",
        ]
    },
    "idaho_sos_2022": {
        "2022_General_Canvass.zip": ("https://sos.idaho.gov/elections/data/results/2022/2022_General_Canvass.zip", None),
    },
    "arkansas_sos_2022": {
        "FullDataFile.json": ("https://results.tally-enr.com/results/arkansas/1844/FullDataFile.json", None),
    },
    "senate_ratings_historical": {
        "output.csv": (None, "0834c4fc53e9ffc2e1d7f651a6c87fa606bb02db325c41ce42ea8db25a15a67e"),
    },
    "All-US-Senate-polls-2006-2024": {
        "2006_to_2024_senate_polls.csv": (
            "https://raw.githubusercontent.com/Jack-Whitcomb/All-US-Senate-polls-2006-2024/"
            "5aea8c5c4572ddc71f4491fa84478e70d88a5898/2006_to_2024_senate_polls.csv",
            "34232aa4c3aba30023a9d8f08445ea0abca0d7eb9e396c40d22df5b86afa6403"),
    },
}
# sha256 of the files whose pin isn't given above, recorded on the first download (2026-10-30;
# kansas_city_mo and openelections_senate 2026-10-31)
PINS = {
    "heideman/v1.0.0/SHA256SUMS.txt": "b9e3645fa8b3a60909cdcb195aff772073f4daabc564d02c7aa394e7193aa66a",
    "heideman/v1.0.0/LICENSE": "7f65352273ba461d3798f616fea68c2173f377c86bd74e9421d1bc13603d3f11",
    "heideman/v1.0.0/LICENSE-DATA.md": "01b81ebd58f0a3b745198a5c28bffe89556b51383eaade3ff3ee13e0e547c3cc",
    "heideman/v1.0.0/CITATION.cff": "c8ca6fc31e645f046346291935212734780f53dd3de2994c3841c7d0d7ad21af",
    "heideman/v1.0.0/README.md": "a63b5a7aa4daca3d0a6a2025cccf972de5ce4195bfed1d5da66335e2d5fcecc1",
    "medsl_pres_state/1976-2024-president.csv": "9936410a6dd4ecc8be7fe202c6dff5b5f8fee59d88f35ab542ed12879941dcb7",
    "medsl_pres_state/codebook-us-president-1976-2020.md": "efed2007526e11ee38d21269841ef30712f7683772877f0d36b556449a6daad4",
    "medsl_pres_state/sources-president.tab": "e16360eac1947c94f135a9cf554652b310716b2fededc7002a7ccc985a8d996c",
    "fec_federal_elections/federalelections2006.xls": "53c758cd83fa2a603d811ea1e2a3974baf703f1099d64b471cc8b6457b8d89f5",
    "fec_federal_elections/federalelections2008.xls": "7ac35ceeaa27acca5e91f4048c025663bb4c8c770d3956e5093ecd54de524db5",
    "fec_federal_elections/federalelections2010.xls": "0ce95daa33e5754221420d463eef7871f742135b5798e1194e1755315f9a9942",
    "fec_federal_elections/federalelections2012.xls": "b92a2bd70e8b44d6c5eadc9cb8dc48c3a8d45bdbb029667b3bd874a66530b9e2",
    "fec_federal_elections/federalelections2014.xls": "2a3105404ce922024583f4b3c2e844bab0d7d54054227b9b11797d4e7433e349",
    "fec_federal_elections/federalelections2016.xlsx": "b4a1d1383602bc388cfbdf1fbea2476476d32e0b44b44236d3a3910fa9782eb6",
    "fec_federal_elections/federalelections2018.xlsx": "f2c42311862b4927df2ca5908c5f9c8f8da6221a8db88fd00434a22ec1e0e303",
    "fec_federal_elections/federalelections2020.xlsx": "5073b6d2c76c86c941508dfb1a11cc497e8529b0068c5132aceb0f385c19352e",
    "fec_federal_elections/federalelections2022.xlsx": "cdb258ea23803e50d752bcea19faa39b3c201eb960094bcb556c3a38c2350897",
    "la_sos_parish/20141104_VotesRaceByParish_47638.json": "1fb80a2a904802440fbfd4642597ff04c7a63adbb946e44716b2a960d271dc55",
    "la_sos_parish/20141206_VotesRaceByParish_50494.json": "e8d6e1a530d33579daec3ef49ceae6a3137bb46abb200afcff370746e0cfad6c",
    "la_sos_parish/20161108_VotesRaceByParish_53905.json": "546c913aecd7ea82cee80a5e1ab2feceac34afe4874889b7fdaf57ee0f32e54e",
    "la_sos_parish/20161210_VotesRaceByParish_54490.json": "3b353620b2e45b114003d59f9041b24b4fa1ba22eca3692389c5c1e71f5c0af8",
    "la_sos_parish/20201103_VotesRaceByParish_59569.json": "64b277af15b1b73664d1f2f6763bf67cb4dd9257385eabdf3efc76eb7d0ff6a9",
    "medsl_senate/1976-2024-senate-state.csv": "6f745db1b4a0026ad837e74428f9ed6f3f77fa51eb6f858b58ebdbc186fdb3bd",
    "medsl_senate/codebook-us-senate-1976–2024.md": "221ca02977af4f2d3803340eab798cd9d09830e3b2c5254e700e0b2b45637333",
    "medsl_senate/sources-senate.csv": "c1de9cd25f1da28eb14289485a0b338a1b0b46128b2584911895e023747203d5",
    "medsl_pres_county/countypres_2000-2024.csv": "9299583148f5f8264baf50b5d929c260699c46d915909b13ea3c3850918946f8",
    "medsl_pres_county/County Presidential Returns 2000-2024.md": "12447fa5e7a7a823bc4416f34a6c624313cc703b745fcbfb2fe64902d2b37c70",
    "medsl_pres_county/sources-president.csv": "427a44a3c35eb63536780ff31a9abfaef87cd11b9c8d0adff444c34adaed8e94",
    "kansas_city_mo/official_results_11_06.pdf": "df8be656c7729ca269ace19d3bea03098904a3c555fa111341581e853f1aecc9",
    "kansas_city_mo/11_02_10_SUMMARY_REPORT.pdf": "b6c46a7e601abd1908f3c5b660ff8421b7c08956485f32ed4b62e1d3dab29791",
    "kansas_city_mo/ELECTION_SUMMARY_OFFICIAL_AMENDED_11-12.pdf": "5331251006243247d5527ece450a3568e796b9c38bcfc1b1f586d9131b6b813e",
    "kansas_city_mo/Election_Summary_Official_11-16.pdf": "2f193687a3da25e3514c3a7d60c9ef6885a413eec3a312cfdd167b36134cb28d",
    "kansas_city_mo/Election_Summary_Official_11-18.pdf": "88c6f988826c89ce67b05df6a17bcefcd42d065aea36cb7ffa75e238da5c0cce",
    "kansas_city_mo/Election_Summary_Official_11-8-22.pdf": "8d096ace1157ed2bc81b648930996612b1a479ae3a395fa5bc6c7e78d6e2567a",
    "kansas_city_mo/20241105__mo__general__precinct.csv": "2f692c4f87a00026b083e94c44c899a8a4861c52b4bcc8764a7cb53a880325eb",
    "openelections_senate/20161108__nh__general__town.csv": "49b651ae20023f041ba13b7425211cbfae3a389bac879c71e99d0f5af2e4ad40",
    "openelections_senate/20161108__ny__general__county.csv": "636b8c2ea5751c6c5489bc590dd16ec0374eb94dcab6c5566a6415cf46a47bc4",
    "openelections_senate/20161108__sc__general__precinct.csv": "556b7cb8e30f0ea2be235e0c23e39953a9a9ee2d455e015995357db4384af705",
    "openelections_senate/20181106__de__general__precinct.csv": "f7b85f1b4c7847bd67920eb9d28cb1fe7804bc0d75ede681283327aa95524978",
    "openelections_senate/20181106__nj__general__county.csv": "780e63cdb80f4ce393b35fe6e8ca58c6b194c667d81a0b77d4fa626e7aecfc23",
    "openelections_senate/20221108__sc__general__precinct.csv": "0f30262df10ceb9cf72d933d52ce6e4c8e4f704c199fad743ac4a6518b4f446f",
    "openelections_senate/20201103__ri__general__precinct.csv": "7c2994b718f93a693eb406bd3ff1572b910edfcd8ba42492e387eb5d22672052",
    "openelections_senate/20241105__ri__general__precinct.csv": "4cdd2b0579c46997f6b9276d8da8f9b5edf37cf85ac80c13dfe419cc87a1d16d",
    "openelections_senate/20121106__me__general__town.csv": "42c404f00d845f6b49f5ae05cbc06012531da562b6de7101e2b80489d1ad5ce7",
    "openelections_senate/20181106__me__general__town.csv": "e50cadf5e97609b42f8b16c048f70ea5200dcfb509a34d2338598dc48a0d747a",
    "openelections_senate/20201103__me__general__precinct.csv": "95785d13cce3371da8abe614622de0bd7e8c566f846f12901580e0d9b25a2550",
    "idaho_sos_2022/2022_General_Canvass.zip": "27aac1e2c391b2d50541bc231550c48c101385714eeda53b92bd4fd6e83d5a69",
    "arkansas_sos_2022/FullDataFile.json": "01b0a4f97b3eed9f7305d46825550b559b12fadad8daf9f998e1a876bf8fe61f",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research; Election26)"})
    with urllib.request.urlopen(request, timeout=300) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f)
    os.replace(tmp, path)


def main() -> None:
    problems, unpinned = [], {}
    for collection, files in SOURCES.items():
        for name, (url, pin) in files.items():
            path = RAW / collection / name
            if not path.exists():
                if url is None:
                    problems.append(f"{path} is missing: save it by hand (see the module docstring)")
                    continue
                print(f"downloading {url}")
                fetch(url, path)
            pin = pin or PINS.get(f"{collection}/{name}")
            digest = sha256(path)
            if pin is None:
                unpinned[f"{collection}/{name}"] = digest
            elif digest != pin:
                problems.append(f"{path}: sha256 {digest} isn't the pinned {pin}; delete it to re-download")
        if not (RAW / collection / "SOURCE.md").exists():
            problems.append(f"{RAW / collection / 'SOURCE.md'} is missing")
    # the release's own checksums cover every Heideman release file
    sums = RAW / "heideman/v1.0.0/SHA256SUMS.txt"
    if sums.exists():
        for line in sums.read_text().splitlines():
            digest, name = line.split()
            path = sums.parent / name
            if path.exists() and sha256(path) != digest:
                problems.append(f"{path} doesn't match the release's SHA256SUMS.txt")
    if unpinned:
        print("Unpinned files (add to PINS):\n" + "\n".join(f'    "{k}": "{v}",' for k, v in unpinned.items()))
    if problems:
        raise SystemExit("\n".join(problems))
    print("All raw sources present and verified.")


if __name__ == "__main__":
    main()
