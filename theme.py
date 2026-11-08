"""Figure palettes.

The website offers a light/dark toggle, and the charts are rendered server-side
by Plotly rather than styled by CSS, so each theme needs a matching palette
here. `assets/style.css` defines the same two themes for the page chrome; the
values are kept deliberately in step with it.
"""

THEMES = {
    "dark": {
        "paper": "#121823",        # .card background
        "ink": "#eaf0fb",
        "muted": "#7c89a0",
        "grid": "#222c3f",
        "housing": "#4cc9f0",
        "crash": "#ff5c68",
        "trend": "#e2e8f5",        # trendline / reference rules
        "marker_line": "#0a0d14",  # ring around scatter markers
        "building": "#ffffff",     # building dots on the map
        "legend_bg": "rgba(10,13,20,0.72)",
        "map_style": "carto-darkmatter",
        # Cold -> hot. The pale low end of a classic YlOrRd disappears on a
        # dark panel, so the ramp starts at a saturated blue instead.
        "seq": ["#1f4e79", "#2e8bc0", "#6fd3c7", "#f7d774", "#f79256", "#ff4d6d"],
        "boroughs": {
            "BRONX": "#c77dff",
            "BROOKLYN": "#4cc9f0",
            "MANHATTAN": "#ff5c68",
            "QUEENS": "#80ed99",
            "STATEN ISLAND": "#ffd166",
        },
    },
    "light": {
        "paper": "#ffffff",
        "ink": "#1d2430",
        "muted": "#6b7280",
        "grid": "#e7ebf1",
        "housing": "#2a6f97",
        "crash": "#c1462e",
        "trend": "#1d2430",
        "marker_line": "#ffffff",
        "building": "#15314a",
        "legend_bg": "rgba(255,255,255,0.82)",
        "map_style": "carto-positron",
        # On white the ramp can start pale, which gives better low-end contrast.
        "seq": ["#eaf3f8", "#9fd0e3", "#6fd3c7", "#f7d774", "#f08a4b", "#c1292e"],
        "boroughs": {
            "BRONX": "#8c5383",
            "BROOKLYN": "#2a6f97",
            "MANHATTAN": "#c1462e",
            "QUEENS": "#3f7d20",
            "STATEN ISLAND": "#c17d04",
        },
    },
}


def get_theme(name):
    """Resolve a theme name coming from the browser, defaulting to dark."""
    return THEMES.get(name if name in THEMES else "dark")


def plotly_layout(T):
    """Base layout shared by every figure in the given theme.

    No explicit `height`: the charts are sized fluidly by CSS so they adapt to
    the viewport, and Plotly's `responsive` mode fills whatever box it is given.
    """
    return dict(
        paper_bgcolor=T["paper"],
        plot_bgcolor=T["paper"],
        font=dict(
            family="'Space Grotesk', Inter, 'Segoe UI', system-ui, sans-serif",
            size=12,
            color=T["ink"],
        ),
        margin=dict(l=62, r=24, t=66, b=92),
        title=dict(font=dict(size=15, color=T["ink"]), x=0.01, xanchor="left"),
        hoverlabel=dict(
            font_size=12,
            bgcolor=T["paper"],
            bordercolor=T["grid"],
            font_color=T["ink"],
        ),
        legend=dict(orientation="h", y=-0.26, x=0, font=dict(size=11, color=T["ink"])),
        transition=dict(duration=250, easing="cubic-in-out"),
    )
