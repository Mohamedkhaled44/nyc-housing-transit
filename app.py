"""
Interactive data-story website — Dash + Plotly.

"Does New York build affordable housing on its most dangerous streets?"

Data Engineering and Visualization, Winter Semester 2026, Project Milestone 1.

The app reads three small pre-aggregated CSVs produced by the notebook
(`data/processed/`). It never calls the NYC Open Data API at runtime, so the
deployed instance starts fast and stays inside a free tier's memory budget.

Interaction model required by the brief:
  * four dropdown filters          -> held as State
  * a free-text search box         -> parsed into those same dropdowns
  * a single "Generate Report"     -> the only Input that redraws the charts
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

# --------------------------------------------------------------------------
# Theme
# --------------------------------------------------------------------------
HOUSING = "#2a6f97"
CRASH = "#c1462e"
INK = "#1d2430"
MUTED = "#6b7280"
PAPER = "#ffffff"
CANVAS = "#f4f6f8"
SEQ = ["#fff3b0", "#ffd166", "#f4845f", "#d1495b", "#8d2135"]

BOROUGH_COLORS = {
    "BRONX": "#8c5383",
    "BROOKLYN": "#2a6f97",
    "MANHATTAN": "#c1462e",
    "QUEENS": "#3f7d20",
    "STATEN ISLAND": "#d98e04",
}

TIER_ORDER = ["No completions", "Q1 lowest", "Q2", "Q3", "Q4 highest"]

ROAD_USERS = {
    "Pedestrians & cyclists": ("vru_injured", "vru_killed"),
    "Pedestrians only": ("ped_injured", "ped_killed"),
    "Cyclists only": ("cyc_injured", "cyc_killed"),
    "Motorists only": ("mot_injured", "mot_killed"),
    "Everyone": ("persons_injured", "persons_killed"),
}

INCOME_BANDS = {
    "All affordable units": "units_total",
    "Extremely low income": "units_extremely_low",
    "Very low income": "units_very_low",
    "Low income": "units_low",
    "Moderate income": "units_moderate",
    "Middle income": "units_middle",
    "Deep affordability (ext. low + very low)": "units_deep_affordable",
}

# An explicit height is required: dcc.Graph is `responsive`, and inside a CSS
# grid cell with no intrinsic height the plot would otherwise collapse to a
# few pixels. The map and the heatmap override it below.
PLOTLY_LAYOUT = dict(
    height=430,
    paper_bgcolor=PAPER,
    plot_bgcolor=PAPER,
    font=dict(family="Inter, Segoe UI, system-ui, sans-serif", size=12, color=INK),
    margin=dict(l=62, r=24, t=66, b=92),
    title=dict(font=dict(size=15, color=INK), x=0.01, xanchor="left"),
    hoverlabel=dict(font_size=12),
    legend=dict(orientation="h", y=-0.26, x=0, font=dict(size=11)),
)

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PROCESSED = os.path.join(HERE, "data", "processed")


def load_data():
    """Read the three processed tables written by the notebook."""
    zip_year = pd.read_csv(
        os.path.join(PROCESSED, "zip_year.csv"), dtype={"zip_code": str}
    )
    points = pd.read_csv(
        os.path.join(PROCESSED, "housing_points.csv"), dtype={"zip_code": str}
    )
    factors = pd.read_csv(
        os.path.join(PROCESSED, "crash_factors.csv"), dtype={"zip_code": str}
    )
    for frame in (zip_year, points, factors):
        if "borough" in frame:
            frame["borough"] = frame["borough"].astype(str).str.upper().str.strip()
    return zip_year, points, factors


ZIP_YEAR, POINTS, FACTORS = load_data()

BOROUGHS = sorted(ZIP_YEAR["borough"].dropna().unique())
YEARS = sorted(int(y) for y in ZIP_YEAR["year"].unique())


# --------------------------------------------------------------------------
# Filtering + search parsing
# --------------------------------------------------------------------------
def apply_filters(df, boroughs, years):
    """Subset any of the three tables by borough and year."""
    out = df
    if boroughs:
        out = out[out["borough"].isin([b.upper() for b in boroughs])]
    if years:
        out = out[out["year"].isin([int(y) for y in years])]
    return out.copy()


def parse_search(query):
    """Turn free text such as 'Brooklyn 2022 pedestrians extremely low' into filters.

    Returns (boroughs, years, income_band, road_user); each element is None when
    the query said nothing about it, so the caller can leave that dropdown alone.
    """
    if not query or not query.strip():
        return None, None, None, None

    text = query.lower()

    found_boroughs = [b for b in BOROUGHS if b.lower() in text]
    # "staten" on its own is unambiguous and people rarely type "island"
    if "staten" in text and "STATEN ISLAND" not in found_boroughs:
        found_boroughs.append("STATEN ISLAND")
    if "nyc" in text or "all boroughs" in text or "city" in text:
        found_boroughs = []

    found_years = [int(y) for y in re.findall(r"\b(20[12]\d)\b", text) if int(y) in YEARS]

    # Year ranges: "2020-2023", "2020 to 2023"
    span = re.search(r"\b(20[12]\d)\s*(?:-|–|to|through)\s*(20[12]\d)\b", text)
    if span:
        lo, hi = sorted(int(g) for g in span.groups())
        found_years = [y for y in YEARS if lo <= y <= hi]

    band = None
    if "extremely low" in text:
        band = "Extremely low income"
    elif "very low" in text:
        band = "Very low income"
    elif "deep" in text:
        band = "Deep affordability (ext. low + very low)"
    elif "middle" in text:
        band = "Middle income"
    elif "moderate" in text:
        band = "Moderate income"
    elif "low income" in text:
        band = "Low income"

    user = None
    if "pedestrian" in text and "cyclist" in text:
        user = "Pedestrians & cyclists"
    elif "pedestrian" in text or "walking" in text or "walk" in text:
        user = "Pedestrians only"
    elif "cyclist" in text or "bike" in text or "cycling" in text:
        user = "Cyclists only"
    elif "motorist" in text or "driver" in text or "car" in text:
        user = "Motorists only"
    elif "everyone" in text or "all injuries" in text:
        user = "Everyone"

    return (
        found_boroughs if found_boroughs else None,
        found_years if found_years else None,
        band,
        user,
    )


def development_tier(df):
    """Quartile-label each ZIP-year by affordable units completed."""
    built = df["units_total"] > 0
    tier = pd.Series("No completions", index=df.index, dtype="object")
    if int(built.sum()) >= 8:
        tier.loc[built] = pd.qcut(
            df.loc[built, "units_total"].rank(method="first"),
            4,
            labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"],
        ).astype(str)
    elif built.any():
        tier.loc[built] = "Q4 highest"
    return pd.Categorical(tier, categories=TIER_ORDER, ordered=True)


def empty_figure(message="No data matches these filters."):
    """Every chart falls back to this instead of raising on an empty frame."""
    fig = go.Figure()
    fig.add_annotation(
        text=message, showarrow=False, font=dict(size=13, color=MUTED),
        xref="paper", yref="paper", x=0.5, y=0.5,
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def style(fig, title, xtitle=None, ytitle=None):
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(title_text=title)
    if xtitle:
        fig.update_xaxes(title_text=xtitle)
    if ytitle:
        fig.update_yaxes(title_text=ytitle)
    fig.update_xaxes(gridcolor="#eceff3", zerolinecolor="#dde2e8")
    fig.update_yaxes(gridcolor="#eceff3", zerolinecolor="#dde2e8")
    return fig


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_map(data, points, injured_col):
    """Cross-domain map: ZIP bubbles (street risk) + individual buildings."""
    if data.empty:
        return empty_figure()

    zips = (
        data.groupby(["zip_code", "borough"], as_index=False)
        .agg(units=("units_total", "sum"), crashes=("crashes", "sum"),
             injured=(injured_col, "sum"), lat=("map_lat", "mean"), lon=("map_lon", "mean"))
        .dropna(subset=["lat", "lon"])
    )
    if zips.empty:
        return empty_figure("No mappable ZIP codes in this selection.")
    zips["rate"] = 100 * zips["injured"] / zips["crashes"].replace(0, np.nan)
    zips = zips.dropna(subset=["rate"])
    if zips.empty:
        return empty_figure("No mappable ZIP codes in this selection.")

    fig = px.scatter_map(
        zips,
        lat="lat", lon="lon", size="crashes", color="rate",
        color_continuous_scale=SEQ, size_max=34, zoom=9.2,
        center=dict(lat=40.714, lon=-73.94),
        custom_data=["zip_code", "borough", "units", "crashes", "rate"],
        labels={"rate": "injuries / 100 crashes"},
    )
    fig.update_traces(
        hovertemplate=(
            "<b>ZIP %{customdata[0]}</b> · %{customdata[1]}<br>"
            "affordable units built: %{customdata[2]:,.0f}<br>"
            "crashes: %{customdata[3]:,.0f}<br>"
            "selected injuries per 100 crashes: %{customdata[4]:.1f}<extra></extra>"
        )
    )

    mappable = points.dropna(subset=["lat", "lon"])
    if not mappable.empty:
        sample = mappable.nlargest(min(900, len(mappable)), "total_units")
        fig.add_trace(
            go.Scattermap(
                lat=sample["lat"], lon=sample["lon"], mode="markers",
                marker=dict(size=6, color=HOUSING, opacity=0.85),
                name="affordable buildings",
                customdata=np.stack([sample["project_name"].astype(str),
                                     sample["total_units"], sample["year"]], axis=-1),
                hovertemplate=("<b>%{customdata[0]}</b><br>%{customdata[1]:,.0f} units"
                               " · completed %{customdata[2]}<extra></extra>"),
            )
        )

    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title_text="Where the housing goes and where the streets hurt people",
        map_style="carto-positron",
        margin=dict(l=0, r=0, t=54, b=0),
        height=520,
        coloraxis_colorbar=dict(title="injuries<br>per 100<br>crashes", thickness=12),
        legend=dict(orientation="h", y=0.01, x=0.01, bgcolor="rgba(255,255,255,0.75)"),
    )
    return fig


def fig_scatter(data, unit_col, injured_col, band_label, user_label):
    """Cross-domain scatter: units built vs. injury rate, one point per ZIP."""
    if data.empty:
        return empty_figure()

    zips = (
        data.groupby(["zip_code", "borough"], as_index=False)
        .agg(units=(unit_col, "sum"), crashes=("crashes", "sum"), injured=(injured_col, "sum"))
    )
    zips = zips[(zips["crashes"] >= 50) & (zips["units"] > 0)]
    if len(zips) < 3:
        return empty_figure("Too few ZIP codes with both housing and crash data here.")
    zips["rate"] = 100 * zips["injured"] / zips["crashes"]

    fig = px.scatter(
        zips, x="units", y="rate", color="borough", size="crashes",
        color_discrete_map=BOROUGH_COLORS, size_max=26, log_x=True,
        custom_data=["zip_code", "units", "crashes", "rate"],
    )
    fig.update_traces(
        marker=dict(line=dict(width=0.6, color="white")),
        hovertemplate=("<b>ZIP %{customdata[0]}</b><br>%{customdata[1]:,.0f} units<br>"
                       "%{customdata[2]:,.0f} crashes<br>"
                       "%{customdata[3]:.1f} injuries / 100 crashes<extra></extra>"),
    )

    # numpy trendline — avoids a statsmodels dependency on the deployment host
    log_units = np.log10(zips["units"])
    if log_units.nunique() > 2:
        slope, intercept = np.polyfit(log_units, zips["rate"], 1)
        xs = np.linspace(log_units.min(), log_units.max(), 40)
        r = float(np.corrcoef(log_units, zips["rate"])[0, 1])
        fig.add_trace(
            go.Scatter(
                x=10 ** xs, y=slope * xs + intercept, mode="lines",
                line=dict(color=INK, dash="dash", width=2),
                name=f"trend (r = {r:+.2f})", hoverinfo="skip",
            )
        )

    return style(
        fig,
        f"Units built vs. {user_label.lower()} injury rate, by ZIP",
        "affordable units completed in the selection (log scale)",
        f"{user_label} injured per 100 crashes",
    )


def fig_tiers(data, injured_col, user_label):
    """Bar: injury rate by how much the ZIP-year built."""
    if data.empty:
        return empty_figure()
    frame = data.copy()
    frame["tier"] = development_tier(frame)
    frame["rate"] = 100 * frame[injured_col] / frame["crashes"].replace(0, np.nan)

    grouped = (
        frame.groupby("tier", observed=True)
        .agg(rate=("rate", "median"), n=("zip_code", "size"))
        .reindex(TIER_ORDER).dropna(subset=["rate"]).reset_index()
    )
    if grouped.empty:
        return empty_figure()

    fig = go.Figure(
        go.Bar(
            x=grouped["tier"].astype(str), y=grouped["rate"],
            marker_color=[MUTED if t == "No completions" else HOUSING for t in grouped["tier"]],
            text=[f"n={int(n)}" for n in grouped["n"]],
            textposition="outside", textfont=dict(size=10, color=MUTED),
            hovertemplate="<b>%{x}</b><br>%{y:.1f} injuries per 100 crashes<extra></extra>",
        )
    )
    return style(
        fig,
        f"Does building more go with more harm? ({user_label.lower()})",
        "affordable units completed in the ZIP-year",
        "median injuries per 100 crashes",
    )


def fig_trend(data, unit_col, injured_col, band_label, user_label):
    """Dual-axis line: units completed and injuries, year by year."""
    if data.empty:
        return empty_figure()
    yearly = (
        data.groupby("year", as_index=False)
        .agg(units=(unit_col, "sum"), injured=(injured_col, "sum"), crashes=("crashes", "sum"))
        .sort_values("year")
    )
    if len(yearly) < 2:
        return empty_figure("Select at least two years to see a trend.")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=yearly["year"], y=yearly["units"], name=band_label,
        marker_color=HOUSING, opacity=0.85, yaxis="y",
        hovertemplate="<b>%{x}</b><br>%{y:,.0f} units<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=yearly["year"], y=yearly["injured"], name=f"{user_label} injured",
        mode="lines+markers", line=dict(color=CRASH, width=3), yaxis="y2",
        hovertemplate="<b>%{x}</b><br>%{y:,.0f} injured<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title_text="Housing completed and people hurt, year by year",
        # extra right margin: this chart carries a second y-axis on that side
        margin=dict(l=62, r=76, t=66, b=92),
        xaxis=dict(title="year", gridcolor="#eceff3", dtick=1),
        # plotly>=6 removed the `titlefont` shortcut: the font lives inside `title`.
        yaxis=dict(title=dict(text="affordable units", font=dict(color=HOUSING)),
                   gridcolor="#eceff3", tickfont=dict(color=HOUSING)),
        yaxis2=dict(title=dict(text=f"{user_label} injured", font=dict(color=CRASH)),
                    overlaying="y", side="right", showgrid=False,
                    tickfont=dict(color=CRASH)),
    )
    return fig


def fig_boroughs(data, unit_col, injured_col):
    """Diverging bar: share of injuries minus share of new units."""
    if data.empty:
        return empty_figure()
    grouped = data.groupby("borough", as_index=False).agg(
        units=(unit_col, "sum"), injured=(injured_col, "sum")
    )
    grouped = grouped[(grouped["units"] > 0) | (grouped["injured"] > 0)]
    if len(grouped) < 2 or grouped["units"].sum() == 0 or grouped["injured"].sum() == 0:
        return empty_figure("Select at least two boroughs to compare them.")

    grouped["unit_share"] = 100 * grouped["units"] / grouped["units"].sum()
    grouped["injury_share"] = 100 * grouped["injured"] / grouped["injured"].sum()
    grouped["gap"] = grouped["injury_share"] - grouped["unit_share"]
    grouped = grouped.sort_values("gap")

    fig = go.Figure(go.Bar(
        x=grouped["gap"], y=[b.title() for b in grouped["borough"]], orientation="h",
        marker_color=[CRASH if g > 0 else HOUSING for g in grouped["gap"]],
        customdata=np.stack([grouped["unit_share"], grouped["injury_share"]], axis=-1),
        hovertemplate=("<b>%{y}</b><br>%{customdata[0]:.1f}% of new units<br>"
                       "%{customdata[1]:.1f}% of injuries<extra></extra>"),
    ))
    fig.add_vline(x=0, line_color=INK, line_width=1)
    return style(
        fig,
        "Which borough absorbs more harm than housing?",
        "share of injuries minus share of new units (pp)",
        "",
    )


def fig_factors(data, factors):
    """Heatmap: contributing-factor mix by development tier."""
    if data.empty or factors.empty:
        return empty_figure()

    tiers = data.copy()
    tiers["tier"] = development_tier(tiers).astype(str)
    lookup = tiers.set_index(["zip_code", "year"])["tier"]

    joined = factors.join(lookup.rename("tier"), on=["zip_code", "year"]).dropna(subset=["tier"])
    joined = joined[joined["factor"].str.lower() != "unspecified"]
    if joined.empty:
        return empty_figure("No contributing-factor data for this selection.")

    pivot = joined.pivot_table(index="factor", columns="tier", values="crashes", aggfunc="sum")
    pivot = pivot.reindex(columns=[t for t in TIER_ORDER if t in pivot.columns]).fillna(0)
    if pivot.empty or pivot.to_numpy().sum() == 0:
        return empty_figure("No contributing-factor data for this selection.")

    totals = pivot.sum(axis=0).replace(0, np.nan)
    share = (100 * pivot.div(totals, axis=1)).fillna(0)
    share = share.loc[share.mean(axis=1).nlargest(10).index]

    fig = px.imshow(
        share, color_continuous_scale=SEQ, aspect="auto",
        labels=dict(color="% of crashes"), text_auto=".1f",
    )
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}% of that tier's crashes<extra></extra>"
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title_text="Why crashes happen, by how much the ZIP built (each column = 100%)",
        margin=dict(l=190, r=24, t=64, b=52), height=440,
        coloraxis_colorbar=dict(title="% of<br>crashes", thickness=12),
    )
    fig.update_xaxes(title_text="affordable units completed in the ZIP-year", side="bottom")
    fig.update_yaxes(title_text="")
    return fig


# --------------------------------------------------------------------------
# Layout helpers
# --------------------------------------------------------------------------
def kpi(value, label, color=INK):
    return html.Div(
        className="kpi",
        children=[
            html.Div(value, className="kpi-value", style={"color": color}),
            html.Div(label, className="kpi-label"),
        ],
    )


def card(graph_id, note=None, height=430):
    """A chart in a panel.

    The height has to be set in CSS, not only in the figure layout: with
    `responsive: True` Plotly sizes the plot to its container and ignores
    `layout.height`, and a CSS grid cell has no intrinsic height, so the chart
    would collapse to a few pixels.
    """
    children = [
        dcc.Graph(
            id=graph_id,
            config={"displayModeBar": False, "responsive": True},
            style={"height": f"{height}px", "width": "100%"},
        )
    ]
    if note:
        children.append(html.P(note, className="chart-note"))
    return html.Div(className="card", children=children)


def dropdown(component_id, label, options, value, multi=False):
    return html.Div(
        className="control",
        children=[
            html.Label(label, htmlFor=component_id),
            dcc.Dropdown(
                id=component_id, options=options, value=value, multi=multi,
                clearable=multi, placeholder="All",
            ),
        ],
    )


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
app = Dash(
    __name__,
    title="Affordable housing on dangerous streets — NYC",
    update_title=None,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server  # gunicorn entry point: `gunicorn app:server`

app.layout = html.Div(
    className="page",
    children=[
        html.Header(
            className="hero",
            children=[
                html.P("NYC Open Data · Housing & Development × Transportation",
                       className="eyebrow"),
                html.H1("Does New York build affordable housing on its most dangerous streets?"),
                html.P(
                    "An affordable apartment is not only a rent number — it is a front door "
                    "that opens onto a particular street. We took 1.04 million police-reported "
                    "collisions from 2018 to 2025, kept the 699,962 that carry a ZIP code, and "
                    "joined them to the 4,392 HPD-financed buildings completed in the same "
                    "years, at the level of ZIP code and year — to ask what kind of street the "
                    "city's below-market housing actually opens onto.",
                    className="standfirst",
                ),
            ],
        ),

        html.Section(
            className="controls",
            children=[
                html.Div(
                    className="search-row",
                    children=[
                        html.Div(
                            className="control grow",
                            children=[
                                html.Label("Search", htmlFor="search"),
                                dcc.Input(
                                    id="search", type="text", debounce=False,
                                    placeholder='Try "Brooklyn 2022 pedestrians" or '
                                                '"Bronx 2019-2023 extremely low income"',
                                    className="search-input",
                                ),
                            ],
                        ),
                        html.Button("Apply search", id="search-btn", n_clicks=0,
                                    className="btn btn-ghost"),
                    ],
                ),
                html.Div(
                    className="filter-row",
                    children=[
                        dropdown("borough", "Borough",
                                 [{"label": b.title(), "value": b} for b in BOROUGHS],
                                 [], multi=True),
                        dropdown("year", "Year",
                                 [{"label": str(y), "value": y} for y in YEARS],
                                 [], multi=True),
                        dropdown("band", "Housing — affordability band",
                                 [{"label": k, "value": k} for k in INCOME_BANDS],
                                 "All affordable units"),
                        dropdown("user", "Transport — road user injured",
                                 [{"label": k, "value": k} for k in ROAD_USERS],
                                 "Pedestrians & cyclists"),
                    ],
                ),
                html.Div(
                    className="generate-row",
                    children=[
                        html.Button("Generate Report", id="generate", n_clicks=0,
                                    className="btn btn-primary"),
                        html.Span(id="filter-summary", className="filter-summary"),
                    ],
                ),
            ],
        ),

        html.Section(id="kpis", className="kpi-row"),

        html.Section(
            className="grid",
            children=[
                html.Div(className="span-2", children=[
                    card("map", "Circles are ZIP codes, sized by crash volume and coloured by "
                                "injury rate; blue dots are individual affordable buildings. "
                                "Drag to pan, scroll to zoom.", height=520)]),
                card("scatter", "Each point is one ZIP code. The dashed line is a least-squares "
                                "fit — it describes the pattern, it does not explain it."),
                card("tiers", "ZIP-years split into quartiles by units completed, plus those "
                              "that completed nothing as a comparison group."),
                card("trend", "Bars read against the left axis, the line against the right."),
                card("boroughs", "Positive means the borough carries a larger share of injuries "
                                 "than of new affordable units."),
                html.Div(className="span-2", children=[card("factors", height=460)]),
            ],
        ),

        html.Section(
            className="story",
            children=[
                html.H2("The story"),
                html.Div(
                    className="story-grid",
                    children=[
                        html.Div([
                            html.H3("The question"),
                            html.P(
                                "New York publishes, building by building, every affordable home "
                                "it finances, and crash by crash, every collision its police "
                                "attend. The two are almost never read together. They should be: "
                                "a household that moves into a subsidised unit does not get to "
                                "choose the traffic outside it. If the city systematically places "
                                "below-market housing on its most hostile streets, then the people "
                                "with the least housing choice absorb the most street risk."
                            ),
                        ]),
                        html.Div([
                            html.H3("The data"),
                            html.P([
                                "Two NYC Open Data sets: ",
                                html.A("Affordable Housing Production by Building",
                                       href="https://data.cityofnewyork.us/Housing-Development/"
                                            "Affordable-Housing-Production-by-Building/hg8x-zxpr",
                                       target="_blank", rel="noopener"),
                                " (41 columns) and ",
                                html.A("Motor Vehicle Collisions – Crashes",
                                       href="https://data.cityofnewyork.us/Public-Safety/"
                                            "Motor-Vehicle-Collisions-Crashes/h9gi-nx95",
                                       target="_blank", rel="noopener"),
                                " (29 columns). They share no entity, so we aggregated both to "
                                "ZIP code × year — a neighbourhood-sized unit at which both "
                                "housing siting and street design are actually decided — and "
                                "joined there.",
                            ]),
                        ]),
                        html.Div([
                            html.H3("What we found"),
                            html.P([
                                "The premise turned out to be wrong in the direction we "
                                "expected it to be right. ",
                                html.B("Queens carries 23.1% of the city's pedestrian and "
                                       "cyclist injuries but receives 13.3% of its new "
                                       "affordable units"),
                                "; the Bronx is the mirror image, taking 31.7% of the units "
                                "while accounting for 15.4% of the injuries. The borough "
                                "building the most has comparatively safe streets.",
                            ]),
                            html.P(
                                "ZIP-years that completed any affordable housing do show a "
                                "higher pedestrian-and-cyclist injury rate than those that "
                                "completed none — 13.4 to 14.9 against 9.7 per 100 crashes — "
                                "but the four building quartiles are flat against each other, "
                                "so building more does not go with more harm. Tracked around "
                                "each ZIP's largest completion wave, injuries simply follow "
                                "the borough trend, and the depth of a project's affordability "
                                "predicts street risk not at all (r = −0.03).",
                                style={"marginTop": "10px"},
                            ),
                            html.P(
                                "The surprise is that the largest projects sit on the safest "
                                "streets: Co-op City absorbed 15,531 units at an injury rate "
                                "of 9.3, while East Harlem's 6,629 units sit at 17.1. Big "
                                "schemes need big parcels, and big parcels have few "
                                "intersections. What is left is a coordination problem, not a "
                                "causal harm — housing goes where land is, danger is where "
                                "the traffic is, and in New York those are different places.",
                                style={"marginTop": "10px"},
                            ),
                        ], className="span-2"),
                        html.Div([
                            html.H3("The limits"),
                            html.P(
                                "None of this is causal. Affordable housing is sited deliberately: "
                                "on cheaper land, in denser places, along commercial corridors. "
                                "Every one of those factors independently predicts street danger, "
                                "and we cannot separate them. A third of crash records carry no "
                                "ZIP code — disproportionately highway collisions — and had to be "
                                "dropped, so this is a picture of the surface street grid. ZIP "
                                "codes are postal routes, not neighbourhoods. And we have no "
                                "exposure denominator, which is why every rate here is per 100 "
                                "crashes rather than per resident."
                            ),
                        ]),
                        html.Div([
                            html.H3("Why it matters"),
                            html.P(
                                "HPD and City Planning already publish street-safety metrics per "
                                "ZIP; they could sit in the siting scorecard beside land cost and "
                                "zoning, so that a high-risk location is knowable at the moment of "
                                "decision rather than afterwards. DOT's Vision Zero map is built "
                                "from crash history — adding pipeline affordable units would make "
                                "it forward-looking, treating a street before a thousand new "
                                "households arrive instead of after."
                            ),
                        ]),
                    ],
                ),
            ],
        ),

        html.Footer(
            className="foot",
            children=[
                html.P("Data Engineering and Visualization · Winter Semester 2026 · "
                       "Project Milestone 1 · German International University"),
                html.P("Data: NYC Open Data (hg8x-zxpr, h9gi-nx95), 2018–2025. "
                       "Figures describe associations between ZIP-year aggregates and are not "
                       "evidence of causation."),
            ],
        ),
    ],
)


# --------------------------------------------------------------------------
# Callbacks
# --------------------------------------------------------------------------
@app.callback(
    Output("borough", "value"),
    Output("year", "value"),
    Output("band", "value"),
    Output("user", "value"),
    Input("search-btn", "n_clicks"),
    Input("search", "n_submit"),
    State("search", "value"),
    State("borough", "value"),
    State("year", "value"),
    State("band", "value"),
    State("user", "value"),
    prevent_initial_call=True,
)
def search_to_filters(_clicks, _submits, query, boroughs, years, band, user):
    """Search mode: parse the text box into the dropdowns above."""
    parsed_boroughs, parsed_years, parsed_band, parsed_user = parse_search(query)
    return (
        parsed_boroughs if parsed_boroughs is not None else boroughs,
        parsed_years if parsed_years is not None else years,
        parsed_band or band,
        parsed_user or user,
    )


@app.callback(
    Output("kpis", "children"),
    Output("filter-summary", "children"),
    Output("map", "figure"),
    Output("scatter", "figure"),
    Output("tiers", "figure"),
    Output("trend", "figure"),
    Output("boroughs", "figure"),
    Output("factors", "figure"),
    Input("generate", "n_clicks"),
    State("borough", "value"),
    State("year", "value"),
    State("band", "value"),
    State("user", "value"),
)
def generate_report(_clicks, boroughs, years, band, user):
    """The single report callback. Dropdowns are State; only the button redraws."""
    band = band or "All affordable units"
    user = user or "Pedestrians & cyclists"
    unit_col = INCOME_BANDS[band]
    injured_col, killed_col = ROAD_USERS[user]

    data = apply_filters(ZIP_YEAR, boroughs, years)
    points = apply_filters(POINTS, boroughs, years)
    factors = apply_filters(FACTORS, boroughs, years)

    if data.empty:
        blank = empty_figure()
        return (
            [kpi("—", "no data")], "No ZIP-years match these filters.",
            blank, blank, blank, blank, blank, blank,
        )

    units = float(data[unit_col].sum())
    crashes = float(data["crashes"].sum())
    injured = float(data[injured_col].sum())
    killed = float(data[killed_col].sum())
    rate = (100 * injured / crashes) if crashes else 0.0

    cards = [
        kpi(f"{units:,.0f}", f"{band.lower()} completed", HOUSING),
        kpi(f"{data['buildings_completed'].sum():,.0f}", "buildings completed", HOUSING),
        kpi(f"{crashes:,.0f}", "crashes recorded", CRASH),
        kpi(f"{injured:,.0f}", f"{user.lower()} injured", CRASH),
        kpi(f"{killed:,.0f}", f"{user.lower()} killed", CRASH),
        kpi(f"{rate:.1f}", "injured per 100 crashes", INK),
    ]

    boro_text = ", ".join(b.title() for b in boroughs) if boroughs else "all five boroughs"
    if years:
        sorted_years = sorted(int(y) for y in years)
        year_text = (f"{sorted_years[0]}–{sorted_years[-1]}"
                     if len(sorted_years) > 1 else str(sorted_years[0]))
    else:
        year_text = f"{min(YEARS)}–{max(YEARS)}"
    summary = (f"Showing {boro_text}, {year_text} · {band} · {user} · "
               f"{len(data):,} ZIP-year records")

    return (
        cards,
        summary,
        fig_map(data, points, injured_col),
        fig_scatter(data, unit_col, injured_col, band, user),
        fig_tiers(data, injured_col, user),
        fig_trend(data, unit_col, injured_col, band, user),
        fig_boroughs(data, unit_col, injured_col),
        fig_factors(data, factors),
    )


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))
