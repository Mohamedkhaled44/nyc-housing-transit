# Does New York build affordable housing on its most dangerous streets?

**Data Engineering and Visualization — Winter Semester 2026 — Project Milestone 1**
German International University, Faculty of Informatics and Computer Science
Dr. Nada Sharaf · TA. Mariam Ali · TA. May Magdy · TA. Mohamed El dagla

| | |
|---|---|
| **Live website** | **<https://nyc-housing-transit-rho.vercel.app>** |
| **Notebook** | [`Milestone1_Housing_Transit.ipynb`](Milestone1_Housing_Transit.ipynb) |
| **Prompt log** | [`PROMPT_LOG.md`](PROMPT_LOG.md) |

---

## The project in one paragraph

New York City publishes, building by building, every affordable home it helps
finance, and crash by crash, every collision its police attend. The two are
almost never read together. They should be: a household that moves into a
subsidised unit does not get to choose the traffic outside its front door. We
took **1.04 million police-reported collisions** from 2018–2025, kept the
**699,962** that carry a ZIP code, and joined them to the **4,392 HPD-financed
buildings** completed in the same years, at the level of **ZIP code × year** —
then asked what kind of street environment the city's below-market housing
actually opens onto.

---

## Datasets

| Domain | Dataset | ID | Rows | Columns |
|---|---|---|---|---|
| Housing & Development | [Affordable Housing Production by Building](https://data.cityofnewyork.us/Housing-Development/Affordable-Housing-Production-by-Building/hg8x-zxpr) | `hg8x-zxpr` | 9,551 | 41 |
| Transportation | [Motor Vehicle Collisions – Crashes](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95) | `h9gi-nx95` | 1,043,903 (2018–2025 slice) | 29 |

Both exceed the nine-column minimum and both carry ZIP code, borough and a date,
which is what makes the integration possible.

**Why a slice of the crash data.** The full table holds ~2.2M rows and the
Socrata API caps a single response at 1,000,000. We pull calendar years
2018–2025 (1,043,903 rows), paged in blocks of 250,000 with a stable
`$order=:id`. 2026 is excluded from **both** datasets because it is partial in
each and would manufacture a downward trend in every time series.

---

## Integration

The two datasets share no entity — a building is not a collision — so a
row-level join would be meaningless. Both sides are **aggregated to ZIP code ×
calendar year** and joined there.

A ZIP code in NYC holds roughly 20,000–100,000 people, which is close to a
neighbourhood: the scale at which housing siting and street design are actually
decided. A year smooths crash seasonality while still leaving eight observations
per ZIP. In the event this gives **228 ZIP codes and 1,677 ZIP-year cells**.

The **crash table is the spine** of an outer join. A ZIP-year that completed no
affordable housing is a real observation of "built nothing" — it becomes a zero,
not a null — and it is precisely the comparison group the analysis needs. An
inner join would have deleted it. Full reasoning is in notebook section 4.3.

---

## What we found

**The premise turned out to be wrong in the direction we expected it to be
right.** Queens carries **23.1%** of the city's pedestrian and cyclist injuries
while receiving **13.3%** of its new affordable units; the Bronx is the mirror
image, taking **31.7%** of the units while accounting for **15.4%** of the
injuries. The borough building the most has comparatively safe streets.

| Finding | Evidence |
|---|---|
| Building *at all* separates the groups; building *more* does not | no-completion ZIP-years sit at 9.7 pedestrian/cyclist injuries per 100 crashes, the four building quartiles at 13.4, 13.6, 14.9, 14.6 — flat against each other (RQ1) |
| No local effect around a completion wave | injuries track the borough's own trend in the three years either side of each ZIP's peak year (RQ3) |
| Depth of affordability predicts street risk not at all | r = −0.03 across 628 ZIP-years (RQ5) |
| The biggest projects sit on the safest streets | Co-op City: 15,531 units at an injury rate of 9.3; East Harlem: 6,629 units at 17.1 (RQ7) |
| Injury and death point opposite ways | high-building ZIPs injure more pedestrians per crash but kill slightly fewer (r = −0.10, RQ10) |

The cross-sectional gap in RQ1 is best read as **composition, not effect**: 531
of the 956 no-completion ZIP-years belong to 84 ZIP codes that never built at
all over eight years — lower-density, more car-dependent places whose crashes
injure motorists rather than pedestrians. The defensible claim is geographic
and narrow: **housing goes where land is available, street danger is where the
traffic is, and in New York those are substantially different places.**

---

## Repository layout

```
.
├── Milestone1_Housing_Transit.ipynb   # EDA → cleaning → integration → 10 RQs
├── build_notebook.py                  # regenerates the notebook from source cells
├── download_data.py                   # fetches both datasets from the NYC API
├── app.py                             # the Dash website
├── assets/style.css                   # site styling (Dash auto-serves /assets)
├── data/
│   ├── raw/                           # git-ignored; recreate with download_data.py
│   └── processed/                     # committed — what the website reads
│       ├── zip_year.csv               # the merged ZIP × year analysis table
│       ├── housing_points.csv         # building-level points for the map
│       └── crash_factors.csv          # contributing factors per ZIP-year
├── requirements.txt
├── render.yaml                        # one-click Render blueprint
├── Procfile / .python-version         # deployment configuration
└── PROMPT_LOG.md                      # required log of every LLM prompt used
```

---

## Setup

```bash
git clone <this repository>
cd nyc-housing-transit

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run the website

```bash
python app.py                      # http://127.0.0.1:8050
```

The app reads only the small committed CSVs in `data/processed/`, so it starts
immediately and never calls the NYC Open Data API at runtime.

### Reproduce the analysis from scratch

```bash
python download_data.py            # ~280 MB into data/raw/, a few minutes
python build_notebook.py           # regenerate the .ipynb from build_notebook.py
jupyter lab Milestone1_Housing_Transit.ipynb
```

Running the notebook top to bottom rewrites the three files in
`data/processed/`, which is what the website serves.

---

## Deployment

The app is a normal WSGI application, so it runs on anything that can host
Flask. Configuration for the two easiest options is committed.

### Option A — Vercel (`vercel.json`, `api/index.py`)

```bash
npm i -g vercel     # if the CLI is not already installed
vercel login
vercel --prod
```

`api/index.py` re-exports the Dash app's underlying Flask server as the WSGI
callable Vercel looks for, and `vercel.json` routes every path to it and
bundles `data/processed/` and `assets/` with the function.

### Option B — Render (`render.yaml`)

1. Push this repository to GitHub.
2. On [render.com](https://render.com), **New → Blueprint** (picks up
   `render.yaml` automatically), or **New → Web Service** and set it by hand:
3. Settings:
   - **Environment**: Python 3
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `gunicorn app:server`
   - **Instance type**: Free
4. Deploy. The first build takes 2–4 minutes.

Alternatively, choose **New → Blueprint** and point Render at this repository:
`render.yaml` already carries all three settings.

**Note on the free tier:** a free Render instance sleeps after about 15 minutes
of inactivity and takes roughly 50 seconds to wake. The first page load after a
quiet period is therefore slow — this is Render, not the app. Reload once and
it is immediate.

`.python-version` pins Python 3.12, which both Render and Vercel support. `gunicorn` does not run on
Windows — use `python app.py` for local testing.

---

## The website

| Requirement | Where it is |
|---|---|
| Multiple dropdown filters | Borough, Year, affordability band (housing), road user injured (transport) |
| Search mode | free-text box; `Brooklyn 2022 pedestrians` or `Bronx 2019-2023 extremely low income` sets the dropdowns |
| **Generate Report button** | the only `Input` on the report callback; all four dropdowns are `State`, so nothing redraws until it is pressed |
| Chart variety | map, scatter, bar, dual-axis bar+line, diverging bar, heatmap |
| Cross-domain visualisation | the map (ZIP risk + individual buildings) **and** the scatter (units vs. injury rate) |
| Story section | question, data, limits, why it matters |
| Interactivity | hover, zoom, pan, legend toggles on every chart |

Every chart degrades to a "no data matches these filters" panel rather than
erroring when a filter combination returns nothing.

---

## Research questions

Ten questions, two per team member, each with one chart.

| # | Member | Question | Chart |
|---|---|---|---|
| 1 | Member 1 | Do ZIPs that build the most affordable housing have more dangerous streets for pedestrians and cyclists? | grouped bar by development quartile |
| 2 | Member 1 | Is that gap widening or closing over 2018–2025? | two-series line |
| 3 | Member 2 | After a ZIP's biggest completion wave, how do injuries move relative to its borough? | indexed event-study line with control |
| 4 | Member 2 | Do big completion years differ from small ones in injury rate? | box plot by tier |
| 5 | Member 3 | Does the *depth* of affordability track street risk? | bubble scatter with trend |
| 6 | Member 3 | Which affordability band lands in the riskiest ZIPs? | stacked bar by risk quintile |
| 7 | Member 4 | Where do new units and dangerous streets overlap geographically? | map, two layers |
| 8 | Member 4 | Which borough absorbs more street harm than its share of new housing? | diverging bar |
| 9 | Member 5 | Do crashes near new affordable housing happen for different reasons? | heatmap, factor × tier |
| 10 | Member 5 | Is unit density associated with how *deadly* a ZIP's crashes are? | log-scale scatter with trend |

---

## Team contributions

> Student IDs are deliberately **not** published here — they go in the submission
> email only. Adjust the task lists below so they match what each person
> actually did.

| Member | Name | Contribution |
|---|---|---|
| 1 | Mohamed Tarek Borham | RQ1, RQ2 and their charts; housing-side pre-integration cleaning (ZIP standardisation, completion-date fallback rule); KPI row on the website |
| 2 | Mohamed Khaled Mohamed Mohamed Ismail | RQ3, RQ4 and their charts; crash-side pre-integration cleaning (ZIP/borough recovery, coordinate validation); data download and paging; repository and deployment |
| 3 | Norhan Mohamed Ahmed Nabeih Abdelghany Hameed | RQ5, RQ6 and their charts; dataset integration (aggregation to ZIP × year, join-type selection); scatter and trend charts on the website |
| 4 | Mina Ayman Andrawes Bekhet Andrawes | RQ7, RQ8 and their charts; post-integration cleaning (join nulls, redundant columns, dtypes); map and borough charts on the website |
| 5 | Noureldin Ashraf Mohamed Hassan Abdo | RQ9, RQ10 and their charts; exploratory data analysis and validation plots; heatmap, search mode and Generate Report wiring |

---

## Limitations

1. **Nothing here is causal.** Every result is an association across ZIP-years.
   Affordable housing is sited deliberately — on cheaper land, in denser places,
   along commercial corridors — and each of those independently predicts street
   danger. We cannot separate them and do not try to.
2. **A third of crash records have no ZIP code** and were dropped, because ZIP
   is the join key. They are disproportionately highway, bridge and parkway
   collisions, so this is a picture of the *surface street grid*.
3. **ZIP codes are postal routes, not neighbourhoods** — the modifiable areal
   unit problem applies throughout.
4. **No exposure denominator.** NYC Open Data publishes no population or
   vehicle-miles figure at ZIP-year resolution, which is why every rate here is
   per 100 crashes rather than per resident.
5. **Only subsidised housing is visible.** Market-rate construction, the large
   majority of new units, is absent from the housing dataset entirely.
6. **Completion is not occupancy** — move-in can trail completion by months.

---

## Submission

One email to **ICS503Data@gmail.com** containing: all team member IDs, this
repository's link, the deployment link, and the two dataset links above.
