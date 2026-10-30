# Emission Reduction Techniques in Australian Manufacturing

A practical guide to analysing reported pollution reduction techniques, changes in emissions around installations, and patterns among manufacturing facilities using Australia's National Pollutant Inventory (NPI).

**Project status:** initial dataset exploration is complete. This README describes the analysis and website to build. The notebooks, generated datasets, model, website, and final findings listed below have not yet been implemented.

## 1. Project objective and scope

Answer this question:

> Which reduction techniques are reported across manufacturing, what emissions changes follow clearly dated installations, and what broader emissions patterns exist among facilities?

### Agreed scope

- Include **all manufacturing, ANZSIC division C**, across Australia.
- Catalogue all reported pollution control devices and cleaner production activities.
- Compare emissions before and after eligible, clearly dated device installations.
- Use a **two-year window on each side** as the main comparison, with one- and three-year sensitivity checks.
- Include **K-means clustering** as the machine learning component.
- Present findings on a simple HTML website hosted with GitHub Pages.
- Leave comments analysis, NLP, causal modelling, and production-adjusted emissions intensity outside this release.

Keeping manufacturing as the study population does not mean every record can enter every calculation. Show how many facilities qualify for each analysis and why others do not.

### What the project can establish

Describe reported technique use, observed emissions changes, consistency across window sizes, and groups of facilities with similar emissions profiles.

Use wording such as **“associated with a decrease in reported emissions”**. Before/after comparisons cannot establish that a device caused a change. Production, fuel use, other equipment, reporting coverage, and estimation methods may also change.

NPI concerns pollutant emissions to air, water, and land. This project is not an estimate of greenhouse gas reductions or CO2-equivalent savings. Do not sum different pollutants into a single pollution or harm score. See the [official NPI data interpretation guidance](https://www.dcceew.gov.au/environment/protection/npi/data).

## 2. The five analysis questions

| Question | How to answer it | Main output | Methods and graphs |
| --- | --- | --- | --- |
| **Q1. Which techniques are most frequently reported?** | Count distinct reporting facilities for each technique in a selected year. Compare patterns across manufacturing industries. | Technique catalogue and adoption summary | Descriptive statistics; horizontal bar chart; searchable table |
| **Q2. Which techniques have enough usable evidence?** | Track eligibility from reported devices through valid timing, complete periods, and pollutant coverage. | Evidence availability and exclusion counts | Data validation; attrition table or simple flow |
| **Q3. What emissions changes follow installation?** | Calculate each eligible facility's average annual emissions before and after installation, then summarise changes by technique and pollutant. | Comparison table and facility examples | Paired descriptive before/after analysis; median-and-IQR dot plot; facility line chart |
| **Q4. Do findings persist across different window sizes?** | Repeat calculations for one, two, and three years per side using both fixed and available cohorts. | Interactive sensitivity findings | Sensitivity analysis; window selector updating the comparison plot, results, counts, and explanation |
| **Q5. Are there groups of facilities with similar emissions profiles?** | Cluster a common year's numerical emissions features and describe the resulting groups. | Cluster profiles, sizes, and industry/technique composition | K-means; silhouette and stability checks; cluster profile heatmap |

There are no final answers to Q3–Q5 yet. Write conclusions after calculating and validating results; do not turn expected patterns into findings.

## 3. What initial exploration established

The local snapshot contains reporting years **1998/1999–2024/2025**.

| Item | Entire snapshot | Manufacturing |
| --- | ---: | ---: |
| Facilities | 8,140 | 2,364 |
| Annual reports | 105,252 | 33,265 |
| Emissions rows | 1,008,823 | 335,844 |
| Technique-report rows | 228,701 | 126,541 |

Manufacturing includes **32 technique categories** and **2,024 facilities that reported a technique at least once**. These are historical coverage counts, not counts of facilities currently operating.

Initial checks found no duplicate facility IDs, report IDs, facility–reporting-year combinations, report–substance keys, or report–technique keys. All technique and emissions records linked to reports. Recheck these properties whenever the source snapshot changes.

Installation timing needs substantial filtering:

| Manufacturing facility–device combinations | Count |
| --- | ---: |
| No installation year in any record | 1,958 |
| One distinct recorded installation year | 2,507 |
| Multiple recorded installation years | 469 |
| Total | 4,934 |

The dictionary defines `year_installed` for pollution control devices. **All cleaner production activities lack this field.** A technique's first reported appearance is not a verified installation date. Repeated annual technique rows are not repeated installations.

These counts describe exploration, not the final eligible cohort. The stricter rules below may reduce samples further.

## 4. Set up the project

### 4.1 Keep source data separate

Current input location:

```text
/Users/kam/Documents/Github/npi-dataset/
```

Use a separate analysis repository containing this README. Read the source files without editing them. Keep any synced project reference files unchanged.

The proposed structure is:

```text
manufacturing-emissions/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 00_prepare_data.ipynb
│   ├── 01_technique_adoption.ipynb
│   ├── 02_evidence_availability.ipynb
│   ├── 03_before_after.ipynb
│   ├── 04_sensitivity.ipynb
│   ├── 05_clustering.ipynb
│   └── 06_export_website.ipynb
├── data/
│   ├── processed/                  # Local intermediate tables; ignored by Git
│   └── results/                    # Small analysis outputs
└── docs/                           # GitHub Pages publishing folder
    ├── .nojekyll
    ├── index.html
    ├── styles.css
    ├── app.js
    └── data/                       # Small JSON files used by the website
```

**Create these files as you work through the modules. They are planned files, not existing tools.** Start in notebooks; only extract shared logic into Python modules if repetition becomes difficult to maintain.

Unless a module specifies otherwise, save intermediate Parquet tables under `data/processed/`, small CSV results under `data/results/`, and browser-ready JSON under `docs/data/`.

### 4.2 Create the environment

From the analysis repository, run these commands in a terminal on macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install pandas numpy scikit-learn matplotlib pyarrow jupyterlab
python -m pip freeze > requirements.txt
export NPI_DATA_DIR="/Users/kam/Documents/Github/npi-dataset"
mkdir -p notebooks data/processed data/results docs/data
python -m jupyterlab
```

`requirements.txt` records the versions used for this project. When recreating the environment later, install with `python -m pip install -r requirements.txt` instead of installing unpinned packages again. Set `NPI_DATA_DIR` again in each new terminal session before launching Jupyter.

Use a setup cell like this in each notebook so paths work when Jupyter starts either in the repository root or its `notebooks/` folder:

```python
import os
from pathlib import Path

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
assert (PROJECT_ROOT / "README.md").exists(), "Start Jupyter in the analysis repository."

DATA_DIR = Path(os.environ["NPI_DATA_DIR"]).expanduser()
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
SITE_DATA_DIR = PROJECT_ROOT / "docs" / "data"
for folder in (PROCESSED_DIR, RESULTS_DIR, SITE_DATA_DIR):
    folder.mkdir(parents=True, exist_ok=True)
```

Add these entries to `.gitignore`:

```gitignore
.venv/
__pycache__/
.ipynb_checkpoints/
.DS_Store
data/processed/
```

Keep the large raw CSV files outside the repository and browser. The emissions file exceeds GitHub's regular 100 MiB file limit. Publish the code and compact derived results. See [GitHub's file-size guidance](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

### 4.3 Keep shared settings in one place

Initially, put the same clearly labelled settings at the top of each notebook. Move them into one configuration file when the workflow is working.

| Setting | Initial value |
| --- | --- |
| Manufacturing filter | ANZSIC division `C` |
| Default adoption reporting year | `2024/2025` |
| Installation windows | `1`, `2`, `3` complete periods per side |
| Default comparison window | `2` |
| Default website comparison cohort | Fixed across all three windows |
| Initial comparison pathway | Total air emissions, kilograms per reporting period |
| Clustering reporting year | `2024/2025` |
| Clustering features | PM10, oxides of nitrogen, total VOCs, each as a separate total-air feature |
| Candidate cluster counts | `2` through `6`, subject to sufficient observations |
| Small-sample display threshold | Fewer than `5` distinct facilities; a presentation rule, not a statistical guarantee |

## 5. Module 0 — Prepare and validate the data

**Notebook:** `00_prepare_data.ipynb`

### Inputs and relationships

| File | What to use it for | Key |
| --- | --- | --- |
| `anzsic-2006.csv` | Map classes to division, subdivision, and group | `class_code` |
| `reports.csv` | Reporting dates, facility membership, report metadata | `report_id` |
| `emission-reduction-techniques.csv` | Reported technique categories and installation years | `report_id` + `reduction_technique_id` |
| `emissions.csv` | Substance-specific emissions and estimation methods | `report_id` + `substance_id` |
| `facilities.csv` | Facility lookup and current information | `facility_id` |
| `substances.csv` | Substance names, definitions, first reporting years | `substance_id` |
| `data-dictionary.csv` | Field meanings and units | Resource and field |

Do not load comments or public statements for this scope. Transfers and secondary industry classifications are not needed for the first release.

### Tasks

1. Create the notebook and load the CSVs with `encoding="utf-8-sig"` to handle the byte-order mark in their headers.
2. Read identifiers and ANZSIC codes as strings, preserving leading zeros. Parse emission amounts and dates separately.
3. Use the ANZSIC table to identify division C classes. Join these to **each report's** primary class, then retain the corresponding techniques and emissions.
4. Use `facility_id` to follow facilities over time. Names can change, so never join on names.
5. Keep emission estimation technique fields, including `air_point_eet` and `air_fugitive_eet`, for later change flags.
6. Validate keys and joins before calculating totals.
7. Save the manufacturing reports, techniques, emissions, and facility lookup as local Parquet tables.
8. Record prepared-table row counts and validation results in `data_quality_summary.csv`.

**Avoid a many-to-many join:** one report can contain several techniques and several substances. Keep these tables separate until a specific event and pollutant have been selected; otherwise emissions can be multiplied accidentally.

### Checks

- [ ] No missing or duplicated required keys; unexpected failures are investigated rather than silently removed.
- [ ] ANZSIC joins are many-to-one; emissions and technique records link to exactly one report.
- [ ] Report dates are parsed, durations calculated, and missing/partial periods flagged.
- [ ] Negative or non-numeric emissions are flagged; missing values remain missing.
- [ ] Use `air_total_emission_kg` directly. Do not add total air to point and fugitive emissions.
- [ ] No sum across different substances is used as a pollution score.

**Outputs:** prepared Parquet tables and `data_quality_summary.csv`.

**Done when:** the manufacturing subset and its joins can be reproduced from the original CSVs without manual edits.

## 6. Module 1 — Q1: Which techniques are most frequently reported?

**Notebook:** `01_technique_adoption.ipynb`

### Tasks

1. Build a catalogue of distinct `reduction_type`, `reduction_technique_id`, and `reduction_technique_name` values across manufacturing history.
2. For each reporting year, count distinct manufacturing facilities submitting a report. This is the denominator.
3. Count distinct facilities reporting each technique in the same year.
4. Calculate:

```text
Reported adoption share (%) =
    100 × facilities reporting the technique / manufacturing facilities reporting that year
```

5. Repeat the counts by subdivision/group for context, using that industry's reporting facilities as its denominator.
6. Make a horizontal bar chart of the ten most frequently reported techniques for the default year. Keep all techniques in the searchable table.

Use **“reported adoption”** or **“reported use”** in labels. No technique record means not reported, not verified non-use. A facility may report multiple techniques, so technique shares do not need to sum to 100%.

### Answer to write

Identify the most frequently reported techniques, their facility counts and shares, and meaningful industry differences. Keep yearly results separate from “ever reported” historical counts.

**Outputs:** `technique_catalogue.csv`, `adoption_by_year.csv`, and `adoption_by_industry.csv`.

**Done when:** the 32-category catalogue is checked against the source snapshot and every displayed percentage has a clear year and denominator.

## 7. Module 2 — Q2: Which techniques have usable before/after evidence?

**Notebook:** `02_evidence_availability.ipynb`

This module creates candidate installation events and identifies which can be compared. It must not estimate missing installation years from first appearances.

### 7.1 Build one candidate per facility and device category

Group all manufacturing technique records by `facility_id` and `reduction_technique_id`.

1. Retain all categories in the catalogue, but only pollution control devices enter installation analysis.
2. Collect the distinct non-missing installation years for each facility–device combination.
3. If none exist, label the combination `installation_year_missing`.
4. If more than one exists, label it `installation_year_conflicting`; do not guess whether these represent replacements.
5. If exactly one exists, validate it. Flag implausible years, such as years before 1900, and years later than a report's actual end year when that report states the year.
6. Check all historical records for the same facility–device category. If it was reported in a period ending before the stated installation calendar year, label it `device_reported_before_installation` and exclude it from the main installation comparison.

A usable event key is `(facility_id, reduction_technique_id, installation_year)`. Annual repetitions of this key still describe one candidate event, not multiple installations. A consistent year is evidence of reported timing, not proof of an exact commissioning date.

Because comments are excluded, ambiguous records remain flagged rather than being manually reinterpreted through free text.

### 7.2 Match techniques to pollutants

Create a small `technique_outcome_mapping.csv` with these columns:

```text
reduction_technique_id, substance_id, emission_column, rationale, reference_url, include_in_release
```

Start by evaluating these candidate matches against the relevant [official NPI industry manuals](https://www.dcceew.gov.au/environment/protection/npi/reporting/industry-reporting-materials/emission-estimation-technique-manuals):

| Device | Candidate outcome | Substance ID |
| --- | --- | --- |
| Electrostatic precipitator `1` | PM10 total air emissions | `70` |
| Fabric filter/baghouse `2` | PM10 total air emissions | `70` |
| Cyclone/multiclone `6` | PM10 total air emissions | `70` |
| Low NOx burner `8` | Oxides of nitrogen total air emissions | `69` |
| Activated carbon filter `9`, incineration/afterburner `10`, thermal oxidiser `12` | Total VOC air emissions, where the industry manual supports the intended application | `85` |

These are starting hypotheses, not evidence that every installation targets that pollutant. Document the rationale and industry applicability before including a match. If a match needs an industry-specific restriction, add that restriction to the mapping rather than applying it to every facility.

Other devices remain visible in the catalogue. Do not force them into an unsuitable outcome. In particular, opacity/dust monitors are measurement devices, and generic “other equipment” does not identify a specific treatment. Wastewater treatment needs a separate water-pollutant design and can wait for a later release.

### 7.3 Define the reporting windows precisely

`year_installed` gives a year, not an exact date. Assume the installation could have occurred at any time from **1 January to 31 December of that year**. Record this as an analysis assumption.

- **Before:** reporting periods ending before 1 January of the installation year.
- **Transition:** all periods overlapping that calendar year; exclude them.
- **After:** reporting periods starting after 31 December of the installation year.

For a window of `w`, take the nearest `w` periods on each side. Require:

1. Each selected period is a full year, lasting 365 or 366 days.
2. Periods are consecutive within each side, with no missing intervening reporting year.
3. The facility uses a consistent annual reporting calendar across the selected periods.
4. The closest before period ends within 366 days of 1 January, and the closest after period starts within 366 days of 31 December. This prevents selecting distant observations to fill a gap.
5. A partial or missing period cannot be skipped in favour of a more distant complete period.
6. The selected pollutant/pathway value is observed for every period on both sides.
7. The device category is reported in each selected post-installation report. Missing listings mean operation is uncertain, not that the device was removed.

Use actual `data_start` and `data_end` dates. Nonstandard reporting periods occur in NPI; the year label alone is insufficient. See [NPI reporting-period guidance](https://www.dcceew.gov.au/environment/protection/npi/reporting/industry-reporting-materials/industry-reporting-frequently-asked-questions).

Example for installation in **2015**, using standard financial-year reports and `w = 2`:

| Before | Transition: excluded | After |
| --- | --- | --- |
| 2012/13 and 2013/14 | 2014/15 and 2015/16 | 2016/17 and 2017/18 |

The window selector changes the number of periods on each side. It does not change the transition-year rule.

### 7.4 Keep an eligibility ledger

For each candidate event, mapped outcome, and window, save:

- Eligibility status and all failure flags.
- One primary exclusion reason for a non-overlapping attrition summary.
- Actual before/after report IDs, dates, observed values, and reporting gaps.
- Whether other dated device installations occur anywhere in the selected before-to-after interval.
- Whether the selected pathway's emission estimation method fields change across the included periods.

For total air, examine point and fugitive EET fields together. These codes describe estimation methods, not pollutant-removal technologies. Do not interpret missing method information as proof that methods stayed the same.

Other installation flags only capture dated changes present in the records. Cleaner production changes and unrecorded interventions remain unobserved.

### Cases to retain without calculating a reduction

| Case | Treatment |
| --- | --- |
| After data only | Keep in the catalogue and evidence counts; reduction unavailable |
| Before data only | Insufficient follow-up |
| Installation predates available records | Existing equipment without an observed baseline |
| Missing/conflicting timing or prior device presence | Exclude from the main event comparison |
| Missing pollutant/pathway value | Missing, never automatically zero |
| Zero before average | Absolute change available; percentage change undefined |
| Facility stops reporting | Missing follow-up; do not assume emissions ceased |

### Answer to write

Explain how much evidence remains for each technique and which exclusions dominate. Keep **candidate events**, **eligible events**, and **distinct facilities** separate in counts. The same facility may contribute to more than one technique.

**Outputs:** `installation_candidates.parquet`, `eligibility_by_window.parquet`, `evidence_summary.csv`, and the validated outcome mapping.

**Done when:** every inclusion and exclusion can be traced to source records, and after-only observations cannot appear as measured reductions.

## 8. Module 3 — Q3: What emissions changes follow installation?

**Notebook:** `03_before_after.ipynb`

### Calculate paired changes

For each eligible event, pollutant, and window:

```text
Before mean = sum of the w before-period emissions / w
After mean  = sum of the w after-period emissions / w

Annual reduction (kg/year) = Before mean − After mean

Percentage reduction (%) =
    100 × (Before mean − After mean) / Before mean
```

Calculate percentage reduction only when the before mean is greater than zero. Positive values mean emissions decreased; negative values mean emissions increased.

These are differences between average annual emissions. Do not describe them as cumulative avoided emissions, measured device capture efficiency, or production-adjusted performance.

### Summarise the two-year results

Group by technique, pollutant, and pathway. Report:

- Distinct facility count and event count.
- Count with a defined percentage change.
- Median percentage reduction and its 25th/75th percentiles.
- Median annual reduction in kilograms.
- Share of eligible events with a positive annual reduction; label its denominator.
- Counts flagged for other installations or estimation-method changes/unknown methods.

Calculate event-level changes first. The median of facility percentage changes is not the percentage change between two sector totals. Keep those calculations distinct.

### Graphs

1. **Comparison dot plot:** one row per technique, median percentage reduction, interquartile range, zero reference line, and sample size. Positive means lower emissions after installation. The IQR describes variation among observations; it is not a confidence interval.
2. **Results table:** include annual kilogram changes and evidence flags beside percentage results. Do not combine outcomes for different pollutants in a single ranking.
3. **Facility timeline:** show annual emissions for a selected eligible facility, shade the installation calendar year, and mark the periods included in the chosen window. Do not invent an exact installation date.

Display small samples, but label results with fewer than five facilities **“small sample: descriptive only”** and do not promote them as headline winners. Tiny baselines can produce extreme percentage increases; show the baseline and absolute change rather than silently clipping results.

### Checks

- [ ] Independently recalculate several examples, including an increase, a decrease, and an unavailable case.
- [ ] The same event is not multiplied by annual technique repetitions or unrelated substances.
- [ ] All values being compared use the same substance and pathway.
- [ ] Counts and denominators match the eligibility ledger.
- [ ] Subdivision/group composition is examined before interpreting a manufacturing-wide difference.

### Answer to write

For each supported outcome, state which techniques had larger typical decreases, how variable the observations were, and how much evidence supports the comparison. Report increases and mixed findings too. If evidence is sparse or confounded, say so rather than naming a best technique.

**Outputs:** `event_changes.parquet`, `technique_summary.csv`, and selected timeline data.

**Done when:** every summary can be reproduced from its event rows and every event from its source reports.

## 9. Module 4 — Q4: How sensitive are findings to window size?

**Notebook:** `04_sensitivity.ipynb`

The website should let users click **1 year**, **2 years**, or **3 years**, updating both the numerical results and the accompanying explanation.

### 9.1 Create two clearly labelled cohort modes

| Mode | Who is included? | What it tells the reader |
| --- | --- | --- |
| **Fixed cohort — default** | For each technique–pollutant–pathway comparison, use the intersection of events eligible for all three windows. | How results change when the window changes while the facilities stay the same |
| **Available cohort** | Use every event eligible for the selected window. | What can be estimated using all available evidence at that window length |

Define the fixed cohort separately for each technique and outcome. It is fixed across windows, not necessarily shared across techniques. Counts should remain unchanged when switching windows within a fixed-cohort comparison.

For the website's fixed cohort, also require a positive before mean at **all three windows**, so percentage changes are defined for the same events in each view. Retain baseline-zero events in the evidence ledger and in available-cohort absolute-change summaries; report them separately from percentage-change samples.

The available-cohort two-year results are the main broad-coverage estimates from Module 3. Label the website's fixed-cohort two-year results separately; they may differ because the fixed cohort requires three-year eligibility too.

If the fixed cohort is empty, display “insufficient common coverage.” Do not silently switch to the available cohort. Both modes still represent observed associations.

### 9.2 Calculate the sensitivity findings

1. Recalculate each event's before/after averages for `w = 1, 2, 3`.
2. Summarise the same metrics as Module 3 for each cohort mode.
3. Track whether the median change stays positive, stays negative, is zero, or changes direction across windows.
4. Show the range of median reductions and the IQRs. Consistent signs alone do not mean the magnitudes are similar.
5. Where estimation-method or concurrent-installation flags are common, optionally repeat on a clearly labelled restricted subset. This is a secondary check; do not replace the main results silently.

### 9.3 Make the explanation update with the selector

Use the following wording as templates, populated from actual results:

| Selection | Method explanation |
| --- | --- |
| 1 year | “Compares the nearest complete annual period before installation with the nearest complete annual period after the transition. This view is more sensitive to individual-year fluctuations.” |
| 2 years | “Compares averages of two complete annual periods on each side. This is the project's main window and reduces dependence on one unusual year.” |
| 3 years | “Compares averages of three complete annual periods on each side. This provides a longer view but, in the available cohort, requires more complete history and follow-up. More unrelated changes may occur over the longer interval.” |

Add a result sentence built from the selected data:

> “For [technique], the [window]-year comparison includes [n] facilities. Median [pollutant] emissions [decreased/increased/did not change] by [absolute percentage]%, with a median annual change of [kg value and direction].”

Use the percentage-change cohort's count for percentage summaries, and use the actual kilogram summary for the kilogram sentence. Their directions can differ because medians are calculated separately; describe both accurately. If a metric is unavailable, say so instead of displaying zero.

Also show the cohort explanation, small-sample status, and a cross-window sentence describing whether direction and magnitude are consistent. Do not hard-code a technique as the strongest performer.

### Interaction acceptance checks

- [ ] Switching 1 → 2 → 3 years updates the plot, IQRs, table, sample sizes, period highlights, and analysis description together.
- [ ] Switching cohort modes updates the results and explains why the sample can change.
- [ ] Fixed-cohort facility/event membership is identical across windows for each comparison.
- [ ] A technique with no eligible data shows an explicit unavailable state.
- [ ] Keyboard users can operate the window buttons; the selected button is clear.
- [ ] Clustering and adoption results do not change when this installation-window control changes.

**Outputs:** `sensitivity_summary.csv` and membership tables identifying the events in each cohort.

**Done when:** a reader can distinguish the effect of a longer window from the effect of a changed sample.

## 10. Module 5 — Q5: Can ML reveal facility emissions patterns?

**Notebook:** `05_clustering.ipynb`

### Define a separate clustering cohort

Use **one row per manufacturing facility reporting in 2024/2025**, regardless of whether it has an eligible installation event. This keeps ML focused on facility profiles rather than installation effects.

Initial numerical features:

| Feature | Substance ID | Source field |
| --- | --- | --- |
| PM10 emissions | `70` | `air_total_emission_kg` |
| Oxides of nitrogen emissions | `69` | `air_total_emission_kg` |
| Total VOC emissions | `85` | `air_total_emission_kg` |

Restrict this numerical comparison to full annual reports using a common reporting calendar. Start with 1 July 2024–30 June 2025; document exclusions of nonstandard periods. Require all three observed features and show the included/excluded facility counts and industry composition. The cluster findings apply to this observed subset, not automatically to every manufacturing facility.

### Fit and evaluate K-means

1. Pivot the three emissions features into a facility-by-feature table. Do not sum them.
2. Inspect missing, negative, extreme, and duplicated feature profiles. Do not convert missing emissions to zero or remove valid outliers simply to improve clusters.
3. Apply `log1p` separately to each feature to reduce skew.
4. Apply `StandardScaler` to the transformed features so one feature's numerical scale does not dominate distance.
5. Fit K-means for candidate `k = 2, 3, 4, 5, 6`, only where sample size and distinct profiles permit. Use explicit settings, such as `init="k-means++"`, `n_init=20`, and `random_state=42`.
6. Record silhouette score and cluster sizes for each candidate. Reject runs that produce fewer distinct clusters than requested.
7. Repeat with several fixed random seeds. Compare partitions using adjusted Rand index; cluster label numbers themselves are arbitrary.
8. Choose a small, interpretable, reasonably stable solution supported by the diagnostics. Record why you chose it. If separation is weak, report that no strong grouping was established rather than forcing a compelling story.
9. Save the feature specification, transformation/scaler parameters, model settings, seeds, diagnostics, and facility labels so the analysis can be reproduced.

K-means finds groups based on numerical distance. It does not discover causes, identify device efficiency, or automatically distinguish clean from dirty facilities. Use the [KMeans documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), [silhouette documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html), and [adjusted Rand index documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html) when implementing these checks.

### Describe the clusters

- Show facility counts and shares within the clustering cohort.
- Create a heatmap of mean standardised log-emission features. State clearly that the colours are transformed relative values, not kilograms.
- Provide median raw kilograms for each pollutant in a small supporting table.
- Summarise industry composition and the shares reporting particular techniques in the same year. Use the cluster's facilities as the denominator.
- Use labels based on observed profiles, such as “relatively higher reported PM10,” only after examining the fitted results.

Do not use facility IDs, numeric ANZSIC codes, or technique IDs as continuous features. Keep industries and reported techniques for interpretation after fitting. A missing technique record means not reported, not verified non-use.

Facility size and industrial processes can explain cluster differences. Production quantities are unavailable, so these are **absolute emissions profiles**, not emissions-intensity or environmental-performance classes.

**Outputs:** `clustering_cohort_summary.csv`, `cluster_diagnostics.csv`, `facility_clusters.csv`, `cluster_profiles.csv`, and model metadata.

**Done when:** the ML result is reproducible, its excluded population is visible, and the interpretation does not imply causality or pollution severity.

## 11. Write the five answers before building the website

Create a short findings document after the calculations are complete. Use one paragraph per question, supported by its table or figure.

| Question | What the final answer must contain |
| --- | --- |
| Q1 | Leading reported techniques, year, facility counts, adoption shares, and industry context |
| Q2 | Eligible sample sizes by technique/outcome and the main reasons evidence was unavailable |
| Q3 | Direction and magnitude of typical changes, variation, sample sizes, and material evidence flags |
| Q4 | Whether direction/magnitude persist across windows in the fixed cohort, plus coverage changes in the available cohort |
| Q5 | Whether useful clusters emerged, their profiles and sizes, stability, and scope of applicability |

Every headline should identify its pollutant, period/window, cohort mode, and supporting sample where relevant. Include null, mixed, and increasing-emissions findings. Do not promise a ranking when the data does not support a useful comparison.

## 12. Module 6 — Export small, consistent website data

**Notebook:** `06_export_website.ipynb`

All statistical calculations and model fitting happen in Python. The website loads precomputed summaries and selects among them; it does not run K-means or recalculate the analysis in the browser.

Export these files under `docs/data/`:

| JSON file | Contents |
| --- | --- |
| `metadata.json` | Data provenance, reporting coverage, assumptions, definitions, settings, and generation date |
| `adoption.json` | Technique catalogue and yearly/industry adoption summaries |
| `evidence.json` | Eligibility counts and exclusion reasons |
| `comparisons.json` | All technique summaries for each outcome, window, and cohort mode |
| `timelines.json` | Only the facility series needed for drill-down, with period dates and inclusion flags |
| `clusters.json` | Cluster profiles, sizes, composition, and fit diagnostics |

Each comparison row should have an unambiguous key:

```text
technique_id + substance_id + emission_pathway + window_years + cohort_mode
```

Include counts, before/after summaries, median reductions, percentage-change IQR, baseline-zero counts, evidence flags, and availability status. If industry-filtered comparisons are later added, include the industry filter in the key and recompute the summaries; do not filter already-aggregated medians in JavaScript.

Use strict JSON: convert unavailable values to `null`, not `NaN` or infinity. Keep percentages in one documented convention, preferably percentage points such as `12.5` for 12.5%, throughout the export and UI.

**Done when:** every displayed value matches its Python output, every lookup key is unique, and an unavailable result remains unavailable after export.

## 13. Build the minimal website

Use **one HTML page with three navigation sections**. Plain HTML, CSS, JavaScript, and one chart library are enough; a backend or frontend framework is unnecessary for this release.

| Section | Minimum content |
| --- | --- |
| **Overview** | Research question, manufacturing scope, years covered, three or four supported findings, adoption bar chart, all-technique table, and evidence counts |
| **Technique comparisons** | Pollutant and technique controls, 1/2/3-year buttons, fixed/available cohort control, comparison plot, results table, changing explanation, and expandable facility timelines |
| **Facility patterns** | Brief K-means explanation, clustering cohort coverage, cluster heatmap, sizes, industry/technique composition, and interpretation |

Use an expandable **Data and methods** panel for provenance, calculations, exclusion rules, ML settings, and limitations. A separate page is not needed initially.

### Window-selector behaviour

Default to **two years** and **fixed cohort**. Load the precomputed comparison rows once, then use a single rendering function whenever the selected pollutant, technique, window, or cohort changes:

```text
User changes a control
    → update selection state
    → select matching precomputed records
    → update plot, results table, counts, explanation, and timeline shading
    → announce unavailable or small-sample states when applicable
```

For a small first release, use PM10 as the default outcome, with the validated device matches available. Add NOx/VOC outcomes once their mappings and samples are checked. Keep the full manufacturing catalogue available regardless of comparison eligibility.

Do not introduce an industry filter into the first website unless it adds a clear insight and the underlying summaries are generated for it. Industry composition tables can supply context without adding controls.

Use a dot plot rather than a bubble chart initially: it communicates the median, variation, zero reference, and sample size directly. A separate sensitivity heatmap is optional because the window buttons already expose those results.

## 14. Check and publish the website

### Preview locally

After the HTML and JSON files exist, run this from the repository root:

```bash
python -m http.server 8000 --directory docs
```

Open [the local preview](http://localhost:8000). Serving the files over HTTP avoids local-file restrictions when JavaScript loads JSON.

### Verification

- [ ] All five questions have an evidence-based answer, or an explicit statement that evidence was insufficient.
- [ ] The 1/2/3-year buttons and cohort selector update every dependent display.
- [ ] Fixed-cohort membership stays constant across windows; available-cohort counts match Python.
- [ ] Graph axes, units, signs, and IQR definitions are clear.
- [ ] Missing results, zero baselines, and small samples render correctly.
- [ ] Facility timelines use actual period dates and the same event windows as the summaries.
- [ ] The ML heatmap agrees with cluster profiles and clearly labels transformed values.
- [ ] Data files use relative URLs such as `./data/comparisons.json`, so project-site subpaths work.
- [ ] The page is usable on a narrow screen and with a keyboard.
- [ ] Source links and the dataset's recorded attribution/licence information are included.
- [ ] No raw CSVs, local intermediate tables, or unnecessary notebook output are bundled into the site.

### Publish with GitHub Pages

1. Commit the analysis code, README, environment requirements, small results, and `docs/` website folder to the intended GitHub repository.
2. Add an empty `docs/.nojekyll` file for the plain static site.
3. In the repository, open **Settings → Pages**.
4. Select **Deploy from a branch**, choose the branch containing the website, select **`/docs`**, and save.
5. Open the published URL shown by GitHub and repeat the important interaction checks there.

GitHub documents the branch and `/docs` publishing options in its [Pages publishing guide](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).

## 15. Completion checklist

- [ ] Module 0: validated, reproducible manufacturing data preparation.
- [ ] Q1: all-technique catalogue and yearly reported adoption analysis.
- [ ] Q2: event eligibility and evidence availability, with exclusions retained.
- [ ] Q3: checked before/after summaries for validated technique–pollutant matches.
- [ ] Q4: fixed and available cohorts for one-, two-, and three-year windows.
- [ ] Q5: evaluated K-means experiment with reproducible profiles and honest interpretation.
- [ ] Five written answers supported by actual outputs.
- [ ] Three-section website with working window and cohort controls.
- [ ] Sources, methods, limitations, and sample sizes visible.
- [ ] Published GitHub Pages site verified against the analysis outputs.

## Future extensions — outside this release

- Comment analysis/NLP to clarify installations, replacements, and operating changes.
- Production or activity data for emissions-intensity analysis.
- PM2.5, water outcomes, and other validated technique–pollutant matches.
- More formal comparison groups or causal designs, if suitable data becomes available.

