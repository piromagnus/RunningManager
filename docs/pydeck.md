# Pydeck + Streamlit — Guide for RunningManager

This document explains how to use **pydeck** (deck.gl bindings for Python) to plot polylines (routes) in Streamlit using either Mapbox or OpenStreetMap (OSM) tiles. It includes installation steps, token setup, OSM fallback, code examples (Path/Line layers) and a short API reference for common usage patterns.

---

## 1. Why pydeck?

- Pydeck is a Python wrapper around deck.gl and renders maps using WebGL. It's fast and well-suited for rendering long polylines and many points.
- Native Streamlit support via `st.pydeck_chart` makes integration straightforward.
- Use cases: activity routes, heatmaps, large-scale tracks, interactive exploration.

## 2. Installation

Install pydeck and the Streamlit integration (Streamlit is already used in this project):

```bash
uv add pydeck
# or with pip inside your venv:
pip install pydeck
```

Optional helpers:

```bash
pip install polyline  # decode Strava/Google encoded polylines
```

Note: Running `uv add pydeck` will add the dependency to your environment managed by `uv` (recommended for this repo).

## 3. Mapbox token (recommended for map styles)

1. Create a Mapbox account and get an access token at https://www.mapbox.com.
2. Store the token in an environment variable. Example in your `.env` (use `.env.example` as a template):

```
MAPBOX_API_KEY=pk.your_mapbox_token_here
```

3. In your Python code set the token for pydeck (two options):

```python
import pydeck as pdk
import os

# Option A: take from environment
pdk.settings.mapbox_api_key = os.getenv("MAPBOX_API_KEY")

# Option B: set directly (avoid in production code)
pdk.settings.mapbox_api_key = "pk.your_mapbox_token_here"
```

4. If you use Mapbox styles that require a token, setting `pdk.settings.mapbox_api_key` is required. For `map_style="open-street-map"` Mapbox token is not required.

## 4. Using OpenStreetMap tiles (no token)

Pydeck's `Deck` supports a `TileLayer` where you can provide custom tile URLs. OSM example:

```python
import pydeck as pdk

tile_layer = pdk.Layer(
    "TileLayer",
    data=None,
    get_tile_data=None,
    tile_size=256,
    get_url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
)

deck = pdk.Deck(layers=[tile_layer], initial_view_state=pdk.ViewState(latitude=0, longitude=0, zoom=1))
html = deck.to_html(as_string=True)
```

Streamlit note: `st.pydeck_chart` does not always accept custom TileLayer map providers. If you need full control over tiles, generate the deck HTML (`Deck.to_html`) and embed it in Streamlit with `st.components.v1.html(html, height=...)`.

## 5. Decoding Strava polylines

Strava activities often include `map.polyline` or `map.summary_polyline` encoded with the Google polyline algorithm. Use the `polyline` package to decode:

```python
import polyline

encoded = "{your_encoded_polyline_here}"
coords = polyline.decode(encoded)  # returns [(lat, lon), ...]

# Convert to (lon, lat) tuples for pydeck
path = [(lon, lat) for lat, lon in coords]
```

## 6. Plotting a polyline (PathLayer / LineLayer)

Use `PathLayer` or `LineLayer` to show routes. `PathLayer` is optimized for continuous paths and supports variable width and color.

Example (PathLayer) — Streamlit + pydeck using Mapbox or OSM:

```python
import os
import streamlit as st
import pydeck as pdk
import pandas as pd
import polyline

# Set Mapbox token (optional)
pdk.settings.mapbox_api_key = os.getenv("MAPBOX_API_KEY")

# Example: decode a Strava encoded polyline
encoded = "..."  # from activity JSON map.summary_polyline
coords_latlon = polyline.decode(encoded)  # list of (lat, lon)

# Convert to (lon, lat) for pydeck
path = [(lon, lat) for lat, lon in coords_latlon]

# Create PathLayer data schema: each item should have a 'path' key
data = [{"path": path, "name": "Activity 1"}]

layer = pdk.Layer(
    "PathLayer",
    data,
    get_path="path",
    get_width=4,
    get_color=[0, 128, 255],
    width_min_pixels=2,
)

view_state = pdk.ViewState(
    latitude=coords_latlon[len(coords_latlon)//2][0],
    longitude=coords_latlon[len(coords_latlon)//2][1],
    zoom=13,
)

deck = pdk.Deck(layers=[layer], initial_view_state=view_state, map_style="mapbox://styles/mapbox/light-v10")

# In Streamlit
st.pydeck_chart(deck)
```

If you want to use OSM tiles and `st.pydeck_chart` does not render custom TileLayer properly, use `deck.to_html()` and embed as HTML:

```python
html = deck.to_html(as_string=True)
import streamlit.components.v1 as components
components.html(html, height=600)
```

## 7. Working with `activities.csv`

- If your `activities.csv` contains encoded polylines (column name like `summary_polyline` or `map.polyline`), decode per-row using `polyline.decode`.
- If it contains lists of lat/lon, ensure you convert to the `(lon, lat)` order for pydeck.

Small example reading a CSV with an `encoded_polyline` column and plotting first route:

```python
import pandas as pd
import polyline

df = pd.read_csv("data/activities.csv")
encoded = df.loc[0, "encoded_polyline"]
coords_latlon = polyline.decode(encoded)
path = [(lon, lat) for lat, lon in coords_latlon]
```

## 8. Pydeck + Streamlit API cheat sheet

- `pdk.Deck(layers=[...], initial_view_state=ViewState(...), map_style=...)` — create a deck visualization.
- `pdk.ViewState(latitude, longitude, zoom, pitch=0, bearing=0)` — camera/view.
- `pdk.Layer(type, data, **props)` — create a layer. Common types:
  - `PathLayer` — polylines/routes, `get_path` property
  - `LineLayer` — simple lines between coordinates
  - `ScatterplotLayer` — points
  - `GeoJsonLayer` — geojson features
  - `TileLayer` — custom tile providers
  - `HexagonLayer` / `GridLayer` — aggregations/density
- `pdk.settings.mapbox_api_key` — set Mapbox token for map styles
- `Deck.to_html(as_string=True)` — export an embeddable HTML page

Streamlit-specific:
- `st.pydeck_chart(deck)` — render a `pdk.Deck` directly in Streamlit (best when using Mapbox styles)
- `st.components.v1.html(html_string, height=...)` — embed deck HTML (useful for custom TileLayer / OSM)

## 9. Tips & gotchas

- pydeck expects coordinate order `(longitude, latitude)` in layer `get_position`/`path` arrays.
- For very long polylines consider simplifying downsampled paths for faster rendering.
- Use `TileLayer` with OSM tiles but prefer Mapbox when you need styled basemaps and high performance.
- Respect Mapbox usage limits and tokens; do not hardcode tokens in source control — put them in `.env` and add to `.env.example`.

## 10. References

- pydeck docs: https://pydeck.gl
- deck.gl (JS) docs: https://deck.gl
- Streamlit `st.pydeck_chart`: https://docs.streamlit.io
- polyline package: https://pypi.org/project/polyline/
- Mapbox tokens: https://account.mapbox.com

---

If you want, I can now:

- implement a reusable helper in `utils/` to decode polylines and produce pydeck-ready paths,
- add a map view to `pages/Activities.py` using Pydeck (preferred) with a Mapbox token fallback to OSM tiles embedded via HTML,
- or create a small demo Streamlit page showing how to plot several activities from `data/activities.csv`.

Tell me which action you prefer and I will implement it.


