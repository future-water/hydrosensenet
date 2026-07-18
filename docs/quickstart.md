# Quickstart

This walkthrough designs a sensor network end-to-end on synthetic data,
so it runs anywhere without downloads. Swap the synthetic matrix for
your own streamflow data (CSV, Parquet, NetCDF, ...) to use it for real.

## 1. The data model

hydrosensenet works on two objects:

- a **streamflow matrix** of shape `(n_timesteps, n_locations)` — each
  column is a candidate sensor location's time series;
- a **locations GeoDataFrame** with one row per column of the matrix,
  in the same order.

```python
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

rng = np.random.default_rng(42)

# Synthetic basin: 60 candidate locations driven by 4 latent signals,
# i.e. a rank-4 spatio-temporal structure typical of streamflow fields.
n_timesteps, n_locations, rank = 730, 60, 4
U = rng.uniform(0.5, 1.5, (n_timesteps, rank))
V = rng.uniform(0.5, 2.0, (rank, n_locations))
streamflow = U @ V

locations = gpd.GeoDataFrame(
    {"gauge_id": [f"gauge_{i:03d}" for i in range(n_locations)]},
    geometry=[Point(-98 + 0.1 * (i % 10), 29 + 0.1 * (i // 10)) for i in range(n_locations)],
    crs="EPSG:4326",
)
```

## 2. Design a network

```python
from hydrosensenet import SensorNetworkDesigner

designer = SensorNetworkDesigner(
    streamflow_data=streamflow,
    locations=locations,
    location_labels=list(locations["gauge_id"]),
)

result = designer.design_network(
    n_sensors=8,      # how many gauges to place
    evaluate=True,    # hold out test data and score the reconstruction
    train_frac=0.7,   # 70% of timesteps for design, 30% for evaluation
    verbose=False,
)
```

`design_network` splits the record in time, runs QR column pivoting on
the training block, and (with `evaluate=True`) reconstructs the *full*
field on the held-out block from just the selected sensors.

## 3. Inspect the result

```python
result.print_summary()

df = result.get_dataframe()      # rank, label, lon/lat, per-sensor scores
print(df.head())

nse = result.performance_metrics["nse"]          # per-location NSE
print("median NSE:", np.nanmedian(nse))
```

Because the synthetic field has rank 4, as few as 4 well-placed sensors
reconstruct it almost perfectly — the median NSE will be ~1.0.

## 4. Export

```python
result.export("sensors.csv")       # CSV with coordinates + metrics
result.export("sensors.geojson")   # or any GeoDataFrame-supported format
```

## 5. Real data

The package bundles a real sample basin — two years of daily National
Water Model v3.0 streamflow for 368 reaches of the lower Colorado
River, Texas (HUC8 12090302, between Columbus and the Gulf coast). In
a repository checkout it loads directly; installed packages download
it once (~0.8 MB) and cache it:

```python
from hydrosensenet import load_example_basin

streamflow_df, locations_gdf = load_example_basin()
print(streamflow_df.shape)  # (731, 368): daily flows, one column per reach

designer = SensorNetworkDesigner(
    streamflow_data=streamflow_df,
    locations=locations_gdf,
    location_labels=list(streamflow_df.columns),
)
real = designer.design_network(n_sensors=20, evaluate=True, verbose=False)
real.print_summary()
```

```{figure} _static/sample_basin_design.png
:alt: River network of the lower Colorado sample basin with reaches colored by test-period NSE and line width proportional to mean flow; the thick mainstem is deep blue (near-perfect reconstruction), the 20 selected sensors sit along it as green dots, and a few thin headwater creeks are red.
:width: 85%

The 20-sensor design on the sample basin (line width follows mean
flow). Reaches are colored by how well their 2021 flows are
reconstructed from the 20 sensors chosen on 2020 data: the Colorado
River mainstem and major tributaries reconstruct near-perfectly
(median per-reach NSE 0.60, flow-weighted relative error ~1%), while
a few thin local creeks (red) respond to rainfall no distant gauge
can see.
```

For your own studies, load data the same way with
{func}`~hydrosensenet.load_streamflow` /
{func}`~hydrosensenet.load_locations` (CSV, Parquet, GeoJSON, ...), or
download NWM directly. From here, the [user guide](guide/index.md)
covers downloading NWM data, locking in existing USGS gauges, and
weighting the design by flood risk.
