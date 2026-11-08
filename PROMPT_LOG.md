# LLM prompt log

The project brief requires that the prompts used at any step of the project be
recorded. This file is that record.

**Where an LLM was used:** reading the project brief, shortlisting and verifying
the two datasets against the NYC Open Data API, drafting the download script,
the cleaning and integration notebook, and the Dash website.

**How the output was verified:** the analytical decisions — the choice of join
key, the join type, each drop-versus-impute call, the treatment of outliers —
were proposed by the assistant and are argued in the notebook's own markdown
cells against evidence printed in the cell directly above each one, so that any
reader can check the reasoning against the numbers rather than taking it on
trust. Every figure and every statistic quoted in the notebook, the README and
the website was produced by executing that code against the live data; nothing
is quoted from memory. Where a stated number and the executed output disagreed,
the prose was corrected to match the output.

---

## Prompts, in order

### 1 — Locating the brief

> okay do you have the project file in here?

### 2 — Reading the brief and planning

> yes extract it and read both pdfs make a full plan then tell me throught the phone

Output: the brief and dataset catalog were read, the submission deadline
identified, the two candidate datasets verified against the Socrata catalog API,
and a plan written out.

### 3 — Team size

> Okay we are 5 members
>
> 2,3 i don't know about that

Answering a question about team size, whether datasets had already been chosen,
and whether a repository existed. Fixed the project at 10 research questions and
10 visualisations.

### 4 — Building the project

> Do the whole project there is no time to give me them anything to do hence
> that i dont have there contact info also

Output: this repository — `download_data.py`, the notebook
`Milestone1_Housing_Transit.ipynb`, the Dash application `app.py` with
`assets/style.css`, the README and this log.

---

## Verification queries issued against the NYC Open Data API

These were run *before* either dataset was committed to, in order to confirm the
column counts, row counts and date ranges stated in the notebook rather than
relying on recalled values:

```
# catalog search for candidate datasets and their column counts
https://api.us.socrata.com/api/catalog/v1?domains=data.cityofnewyork.us&q=<dataset name>

# full column metadata for each finalist
https://data.cityofnewyork.us/api/views/hg8x-zxpr.json
https://data.cityofnewyork.us/api/views/h9gi-nx95.json
https://data.cityofnewyork.us/api/views/ygpa-z7cr.json

# row counts and coverage, used to justify the 2018-2025 analysis window
.../resource/h9gi-nx95.json?$select=count(*)&$where=crash_date >= '2018-01-01T00:00:00'
.../resource/h9gi-nx95.json?$select=max(crash_date),min(crash_date)
.../resource/hg8x-zxpr.json?$select=count(*)
.../resource/hg8x-zxpr.json?$select=min(project_completion_date),max(project_completion_date)

# ZIP coverage on the crash side, which drives the largest cleaning decision
.../resource/h9gi-nx95.json?$select=count(*)&$where=... AND zip_code IS NOT NULL

# housing completions per year, used to pick the window
.../resource/hg8x-zxpr.json?$select=date_trunc_y(project_completion_date) as yr,count(*)&$group=yr
```

Results that shaped the project:

| Query | Result | What it decided |
|---|---|---|
| crashes, 2018-01-01 onward | 1,114,623 rows | exceeds Socrata's 1,000,000-row response cap, so the download must be paged |
| crashes, 2018–2025 | 1,043,903 rows | the analysis window |
| crashes 2018–2025 with a ZIP | 700,423 rows | ~33% of crash records cannot be joined; drives the main cleaning decision and the bias we report |
| crash date range | 2012-07-01 to 2026-11-03 | 2026 is partial — excluded |
| housing completions by year | 2014–2026, 1,819 undated | 2026 is partial — excluded; undated rows need a fallback rule |

Every one of these figures is reproduced by `download_data.py` and the notebook
when they are run.
