# Weather and Technicality Enrichment for Trail Digital Twin Modelling

Pierre Marrec Running Manager repository, local audit dated 2026-06-15.

## Abstract

This note specifies how weather and path technicality are collected, represented, and
allowed to enter the trail-running digital twin notebook. The current repository does
not ingest the weather layer shown in the Strava consumer application. It stores the
Strava detailed activity JSON returned by the sync service and can read only weather
fields that are present in that cache. A local audit found 711 cached Strava detail
files, 40 non-null `average_temp` fields, and 0 such fields for `TRAIL_RUN`
activities. No OpenWeather cache, Open-Meteo cache, or external technicality cache is
present under `data/`. Technicality is therefore computed locally from GPS geometry as
a reproducible proxy, not as a true surface or path-width classification. Enduraw
information is explicitly prohibited and must not be parsed from Strava descriptions or
any other text field.

## Keywords

Trail running; digital twin; weather enrichment; Strava cache; OpenWeather; GPS
technicality; grade-adjusted pace; reproducibility.

## 1. Introduction

The trail digital twin notebook models observed race or race-like performance from
distance, elevation, grade-adjusted pace, altitude, fatigue, heart rate, training load,
and optional environmental variables. Weather and trail technicality are useful but
high-risk features: they are often sparse, provider-dependent, and easy to contaminate
with unreviewed text scraped from activity descriptions.

The objective of this document is to define a reproducible collection protocol for two
feature families:

- weather: ambient temperature, wind, humidity, precipitation, and heat-stress proxies;
- technicality: route difficulty not already explained by distance, elevation, or grade.

The protocol is deliberately conservative. Missing weather stays missing. External data
must be cached with provenance before it can affect model selection.

## 2. Materials

### 2.1 Repository Data

The local audit used these files and directories:

| Source | Role | Observed status |
| --- | --- | --- |
| `data/activities.csv` | Activity metadata and `rawJsonPath` links | Present |
| `data/activities_metrics.csv` | Activity category, TRIMP, average HR | Present |
| `data/raw/strava/*.json` | Cached Strava detailed activity JSON | 711 files |
| `data/timeseries/*.csv` | Stream-derived HR, speed, altitude, lat/lon | 709 files |
| `data/enrichment/` | Optional weather and external technicality caches | Absent |

### 2.2 Strava Collection Path

The app sync service does not read weather from the Strava website or mobile UI. The
collection path is API based:

1. `StravaService.sync_last_n_days()` lists recent activities.
2. For a cache miss, `_get_activity()` downloads the detailed activity payload.
3. `_save_raw_activity()` writes that payload to `data/raw/strava/{activityId}.json`.
4. `_get_streams()` downloads activity streams.
5. `_save_timeseries()` writes selected streams to `data/timeseries/{activityId}.csv`.

In the current implementation, `_save_timeseries()` stores `time`, `heartrate`,
`velocity_smooth`, `altitude`, `cadence`, and `latlng` derived columns. Although the
Strava public API model includes a `temp` stream in `StreamSet`, this repository does
not currently request or persist a temperature stream.

### 2.3 External Weather Sources

OpenWeather is an allowed external source only through a reviewed offline cache. The
OpenWeather One Call 3.0 timestamp endpoint is coordinate/time based:

```text
https://api.openweathermap.org/data/3.0/onecall/timemachine?lat={lat}&lon={lon}&dt={time}&appid={API key}
```

No OpenWeather configuration key, service module, or cache file was found in the source
tree or `.env.example` during this audit. Therefore, current notebook runs cannot use
OpenWeather unless a cache is added.

## 3. Weather Methods

### 3.1 Allowed Source Hierarchy

Weather features must be collected in this order:

1. Strava cache field `average_temp` from `data/raw/strava/{activityId}.json`.
2. Reviewed external provider cache, preferably
   `data/enrichment/openweather_weather.csv`.
3. Legacy or alternative reviewed weather cache
   `data/enrichment/open_meteo_weather.csv`.
4. Missing value.

The hierarchy is conservative: Strava first-party fields are kept when present, and
external caches fill missing or additional variables such as wind and humidity. External
rows must not be fetched silently inside the modelling notebook.

### 3.2 Prohibited Sources

The following sources are not allowed:

- Enduraw text or any Enduraw-derived value;
- Strava activity `description` parsing;
- screenshot, HTML, or mobile-app UI scraping;
- manually copied weather text without provider, timestamp, and location provenance.

This rule is hard for the repository. Weather values must be machine-readable,
provider-attributed, and reproducible.

### 3.3 Weather Cache Schema

The preferred OpenWeather cache schema is:

| Column | Type | Meaning |
| --- | --- | --- |
| `activityId` | string | Repository activity identifier |
| `provider` | string | Example: `openweather` |
| `weatherSource` | string | Provider endpoint or cache name |
| `observedAt` | ISO datetime | Weather timestamp used for the activity |
| `latitude` | float | Query latitude |
| `longitude` | float | Query longitude |
| `temperatureC` | float | Ambient temperature in Celsius |
| `feelsLikeC` | float | Provider apparent temperature, if available |
| `windKmh` | float | Wind speed converted to km/h |
| `windDeg` | float | Wind direction in degrees |
| `humidityPct` | float | Relative humidity |
| `precipMm` | float | Rain or snow equivalent over the provider window |
| `weatherCode` | string | Provider weather condition code |
| `sourceUrl` | string | Endpoint or documentation URL |
| `retrievedAt` | ISO datetime | Cache retrieval timestamp |
| `confidence` | float | 0 to 1 review confidence |

The notebook accepts a smaller subset today, but new rows should follow this schema so
that provider quality can be audited later.

### 3.4 Current Weather Audit

The local audit found:

| Measurement | Value |
| --- | ---: |
| Cached Strava detail JSON files | 711 |
| Detail files with weather-like keys other than `average_temp` | 0 |
| Detail files with non-null `average_temp` | 40 |
| Timeseries files scanned | 709 |
| Timeseries files with weather-like columns | 0 |
| `TRAIL_RUN` activities | 80 |
| `TRAIL_RUN` activities with non-null `average_temp` | 0 |
| Hard/race-like `TRAIL_RUN` activities in the notebook filter | 45 |
| Hard/race-like `TRAIL_RUN` activities with non-null `average_temp` | 0 |
| OpenWeather cache files | 0 |
| Open-Meteo cache files | 0 |

Non-null `average_temp` values are present only outside trail running in the current
cache: 32 `RUN`, 5 `BACKCOUNTRY_SKI`, and 3 `OTHER` activities.

## 4. Technicality Methods

### 4.1 Definition

Technicality is the portion of trail difficulty not captured by grade, elevation gain,
altitude, distance, or fatigue. In an ideal dataset it would include surface roughness,
rockiness, roots, mud, snow, path width, exposure, and obstacle frequency.

The current repository does not have a trusted external surface dataset. The notebook
therefore computes a local GPS technicality proxy for each segment.

### 4.2 GPS Technicality Proxy

The helper `gps_technicality_index(latitudes, longitudes)` computes technicality from
route geometry:

1. Remove samples with missing latitude or longitude.
2. Convert latitude and longitude to a local planar coordinate system using an
   equirectangular approximation around the segment mean latitude.
3. Draw the straight chord from the first point to the last point of the segment.
4. For every GPS point, compute the perpendicular distance to that chord.
5. Compute the standard deviation of those distances.
6. Scale by 50 m and clamp to the interval `[0, 1]`.

In formula form:

```text
technicalityGps = clamp(std(perpendicular_distance_to_segment_chord) / 50, 0, 1)
```

The value is computed at segment level in `segment_timeseries()` and then summarized at
activity level with distance or segment aggregation in the notebook.

### 4.3 Interpretation

The proxy should be interpreted as GPS meander and local path complexity, not as a
ground-truth trail-surface measure.

High values can indicate:

- switchbacks;
- sinuous singletrack;
- GPS noise in forested or mountainous terrain;
- route sections where straight-line distance poorly represents travelled path.

Low values can indicate:

- straight runnable trail;
- road or fire-road sections;
- straight but still rocky or technical trails that the proxy cannot detect.

### 4.4 Terrain Family Features

The notebook also models terrain families from segment grade. These are separate from
technicality:

- steep climb;
- low climb;
- flat;
- low descent;
- steep descent.

Grade family captures slope-dependent speed. `technicalityGps` captures route geometry
that remains after grade and elevation have already been measured.

### 4.5 External Technicality Cache

External technicality is allowed only through a reviewed cache:

```text
data/enrichment/technicality_external.csv
```

Recommended schema:

| Column | Type | Meaning |
| --- | --- | --- |
| `activityId` | string | Repository activity identifier |
| `technicalityExternalScore` | float | 0 to 1 external score |
| `method` | string | Source method, e.g. manual race report review |
| `sourceUrl` | string | URL or local source reference |
| `retrievedAt` | ISO datetime | Retrieval or review timestamp |
| `confidence` | float | 0 to 1 confidence |
| `notes` | string | Short review note |

No such cache is currently present.

## 5. Data Quality and Bias

Weather and technicality should be treated as missing-not-at-random. A missing
temperature is not equivalent to mild weather. It means the repository does not have an
allowed, auditable weather observation for that activity.

The Strava `average_temp` field is activity-level, not segment-level. It cannot explain
temperature changes over long mountain races unless a provider cache adds time-varying
observations. The current technicality proxy is segment-level but indirect. It can
overstate technicality on winding but smooth paths and understate technicality on
straight rocky trails.

These limitations imply that model comparisons should report weather and technicality
coverage before interpreting fitted coefficients.

## 6. Reproducible Repository Rules

The repository rules are:

1. Do not parse activity descriptions for weather.
2. Do not use Enduraw information.
3. Do not scrape Strava UI weather.
4. Use Strava `average_temp` only when it exists in the cached raw JSON.
5. Use OpenWeather only from an offline cache with provider, coordinates, timestamp,
   retrieval time, and confidence.
6. Preserve missing weather values instead of imputing neutral conditions by default.
7. Treat `technicalityGps` as a proxy and label it as such in outputs.
8. Require source URL and confidence before using external technicality in model
   selection.

## 7. Conclusion

For the current trail-running data, weather is not yet an effective model input because
no allowed weather observation exists for any TrailRun. The reproducible path forward is
to add an OpenWeather cache keyed by `activityId`, start coordinate, and start time, then
rerun the notebook and compare cross-validated error with and without the weather
features. Technicality is currently gathered from GPS geometry; it is useful as a
consistent local proxy but should not be presented as measured trail surface difficulty.

## References

1. Strava Developers. API reference, `getActivityById` detailed activity response:
   <https://developers.strava.com/docs/reference/#api-Activities-getActivityById>.
2. Strava Developers. API reference, `StreamSet` including `temp`:
   <https://developers.strava.com/docs/reference/#api-models-StreamSet>.
3. OpenWeather. One Call API 3.0, timestamp weather endpoint:
   <https://openweathermap.org/api/one-call-3>.
4. MDPI Sensors paper used by the notebook:
   `docs/science/sensors-26-03731.pdf`.
5. Repository implementation references:
   `services/strava_service.py`, `services/trail_performance_model.py`, and
   `notebooks/trail_digital_twin_reproduction.ipynb`.
