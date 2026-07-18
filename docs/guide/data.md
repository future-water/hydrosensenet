# Loading and saving data

hydrosensenet needs two inputs: a streamflow time-series table (time ×
locations) and a table of candidate sensor locations. This page shows how
to read both from common formats, save them efficiently, and pull data
from the National Water Model (NWM) and the USGS gauge network. See
[installation](../installation.md) and the [quickstart](../quickstart.md).

## Streamflow time series

{func}`hydrosensenet.load_streamflow_data` reads CSV, Parquet, Excel,
NetCDF, GRIB, and HDF5 through one interface. With `format="auto"` (the
default) the format is inferred from the extension (`.csv`/`.txt`,
`.parquet`/`.pq`, `.xlsx`/`.xls`, `.nc`/`.nc4`, `.grib`/`.grib2`/`.grb`,
`.h5`/`.hdf5`; anything else falls back to CSV). Tabular formats return a
`pandas.DataFrame`; NetCDF and GRIB return an `xarray.Dataset` (GRIB needs
the `cfgrib` engine); an already-loaded `DataFrame` or `Dataset` is
returned unchanged. For CSV files the time index is detected automatically:
an unnamed first column becomes the index; otherwise a column named `time`,
`Time`, `datetime`, `date`, `Date`, or `timestamp` is parsed as dates. Use
`time_col=` for other names, `location_cols=` to keep a subset of stations,
and pass a list of paths to concatenate per-year files along the time axis.

```python
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import hydrosensenet as hsn

rng = np.random.default_rng(42)
workdir = Path(tempfile.mkdtemp())

# Synthetic daily flows: 120 days at 5 model reaches
flow = pd.DataFrame(
    rng.gamma(2.0, 15.0, size=(120, 5)),
    index=pd.date_range("2020-01-01", periods=120, freq="D"),
    columns=[f"reach_{i}" for i in range(5)],
)
flow.index.name = "time"
flow.to_csv(workdir / "flow.csv")
flow.to_parquet(workdir / "flow.parquet")
df_csv = hsn.load_streamflow_data(workdir / "flow.csv")      # index auto-detected
df_pq = hsn.load_streamflow_data(workdir / "flow.parquet")   # index preserved

# A list of files is concatenated along the time axis
flow.iloc[:60].to_csv(workdir / "flow_a.csv")
flow.iloc[60:].to_csv(workdir / "flow_b.csv")
df_multi = hsn.load_streamflow_data([workdir / "flow_a.csv", workdir / "flow_b.csv"])
assert df_multi.shape == flow.shape
```

## Gauge locations

{func}`hydrosensenet.prepare_gauge_locations` turns a CSV/Excel file,
`DataFrame`, or vector file (`.shp`, `.geojson`, `.gpkg`) into a
`GeoDataFrame` in `EPSG:4326` (override with `crs=`). Vector files are
read and reprojected as-is; for tabular inputs, point geometry is built
from coordinate columns found by alias — `lat`, `Latitude`, `LAT`,
`gauge_lat`, `y`, `Y` for latitude, analogous names for longitude — or via
`lat_col=`/`lon_col=`. Rows with missing coordinates are dropped, the
columns are renamed to `latitude`/`longitude`, and `id_col=` (if present)
is renamed to `gauge_id`; otherwise a sequential `gauge_id` is generated.

```python
gauge_df = pd.DataFrame({
    "site_no": ["08154700", "08155200", "08155300", "08156800", "08158000"],
    "lat": [30.30, 30.29, 30.27, 30.26, np.nan],   # last row is dropped
    "lon": [-97.81, -97.87, -97.80, -97.77, -97.69],
})
locations = hsn.prepare_gauge_locations(gauge_df, id_col="site_no")
print(locations[["gauge_id", "latitude", "longitude"]])  # 4 rows, EPSG:4326
```

## Fast saving and loading: the io module

For data you write yourself, prefer the `hydrosensenet.io` functions. They
default to Parquet (snappy-compressed, pyarrow engine), typically far
smaller and faster than CSV for wide streamflow tables.
{func}`hydrosensenet.io.save_streamflow` appends `.parquet` when the path
has no suffix; {func}`hydrosensenet.io.load_streamflow` accepts `columns=`
to read a station subset without loading the whole file.
`save_locations`/`load_locations` do the same for location tables,
round-tripping `GeoDataFrame`s through GeoParquet (GeoJSON, shapefile, and
CSV also work, by extension). `migrate_csv_to_parquet` converts a legacy
CSV; `migrate_directory` converts every CSV matching a glob pattern.

```python
from hydrosensenet.io import (
    load_locations, load_streamflow, migrate_csv_to_parquet,
    save_locations, save_streamflow,
)

flow_path = save_streamflow(flow, workdir / "flow_daily")  # -> flow_daily.parquet
subset = load_streamflow(flow_path, columns=["reach_0", "reach_3"])
assert list(subset.columns) == ["reach_0", "reach_3"]
loc_path = save_locations(locations, workdir / "gauges.parquet")
locations_back = load_locations(loc_path)                  # GeoDataFrame again
assert locations_back.crs == locations.crs
pq = migrate_csv_to_parquet(workdir / "flow_a.csv")        # -> flow_a.parquet
```

## Downloading NWM retrospective streamflow

The block below is display-only: it streams data from AWS and needs the
optional download stack (`pip install "hydrosensenet[nwm]"`, which also
provides `pynhd` for HUC filtering).
{class}`hydrosensenet.data.NWMDataLoader` opens the NWM retrospective Zarr
store on S3 (versions "2.0", "2.1", "3.0") and by default starts a local
Dask cluster (8 workers × 4 threads, 8 GB per worker — tune via
`n_workers=`, `threads_per_worker=`, `memory_limit=`, or `use_dask=False`).
The full hourly archive is multiple terabytes, so filter first:
`filter_by_huc` resolves a HUC code to NHDPlusV2 COMIDs (first use
downloads a ~245 MB attribute table, cached afterwards); `filter_by_comids`
takes an explicit reach list. Only the selected reaches and time slice are
transferred, but downloads can still take hours for large basins.

```{code-block} python
from hydrosensenet.data import NWMDataLoader

loader = NWMDataLoader(version="3.0")   # starts the Dask cluster
loader.filter_by_huc("1204")            # Texas-Gulf HUC4
# or: loader.filter_by_comids([5781369, 5781371, 5781401])

df = loader.download_streamflow(
    start_date="2020-01-01", end_date="2020-12-31",
    resample="D",                       # daily means of the hourly archive
    output_file="streamflow_2020.parquet",
)

# Multi-year: one streamflow_<year>.parquet per year, ready for load_streamflow_data
files = loader.download_multiple_years(
    years=[2018, 2019, 2020], output_dir="nwm_data", resample="D",
)
```

## USGS gauges as existing sensors

Also display-only (network download, core dependencies only).
`download_usgs_gauges` fetches the USGS Streamgage NHDPlus dataset (12,422
gauges with pre-computed COMID linkages) from ScienceBase, optionally
filtered to a HUC2 region, and returns paths to a locations GeoJSON and a
STAID–COMID linkage CSV; `load_usgs_gauges` reads them back (optional
`huc2=`/`comids=` filters). `match_usgs_to_nwm` returns the COMIDs of
gauges present in your NWM columns plus the matched gauge `GeoDataFrame`;
converted to column indices, they become the `existing_sensors` argument
of `design_network`, locking already instrumented reaches in place during
optimization (see [network design](design.md)).

```{code-block} python
from hydrosensenet.data import download_usgs_gauges, load_usgs_gauges, match_usgs_to_nwm

geojson_file, comid_file = download_usgs_gauges("usgs_data", huc2="12")
usgs_gdf, linkage = load_usgs_gauges(geojson_file, comid_file)
usgs_comids, matched_gdf = match_usgs_to_nwm(
    usgs_gdf, linkage, available_comids=df.columns.tolist()
)
existing = [df.columns.get_loc(c) for c in usgs_comids]
result = designer.design_network(n_sensors=50, existing_sensors=existing)
```
