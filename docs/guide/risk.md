# Risk-informed design

QR pivoting picks sensors purely by how much variance they explain, yet two
candidate reaches often carry nearly the same information — and when they do,
you would rather instrument the one next to a hospital than the one in an empty
canyon. Risk-informed design encodes that preference as a per-location weight
vector: {func}`hydrosensenet.sensor_placement_qr` multiplies each column of the
streamflow matrix by its weight *before* pivoting (`X_w = X * weights`), so
high-risk locations win ties and near-ties while the data's low-rank structure
still dominates the selection — moderate weights shift placements toward risk
without destroying reconstruction quality. This page assumes the
[quickstart](../quickstart.md); the `python` blocks below concatenate into one
runnable script on in-memory synthetic data, while the FEMA blocks are
display-only because they need a manually downloaded file.

## Building a weight vector

{func}`hydrosensenet.calculate_spatial_weights` turns several kinds of risk data
into a vector aligned with your locations. First, some synthetic gauges and
streamflow (a few shared modes plus noise, so the matrix is genuinely low-rank):

```python
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, box

rng = np.random.default_rng(5)
n_locations, n_timesteps = 12, 300

xy = rng.uniform(0, 4, size=(n_locations, 2))
gauges = gpd.GeoDataFrame(
    {"gauge_id": [f"G{i:02d}" for i in range(n_locations)]},
    geometry=[Point(x, y) for x, y in xy],
    crs="EPSG:4326",
)

modes = rng.standard_normal((n_timesteps, 3))
loadings = rng.standard_normal((3, n_locations))
streamflow = 10.0 + modes @ loadings + 0.05 * rng.standard_normal((n_timesteps, n_locations))
```

`weight_source` accepts four types. An **ndarray** already in row order passes
through unchanged (only its length is checked), and a **dict** is looked up by
location id — the `id_column` you name, else a `gauge_id` column, else the
GeoDataFrame index. Ids missing from the dict get `fill_value`:

```python
from hydrosensenet import calculate_spatial_weights

w_arr = calculate_spatial_weights(gauges, np.linspace(1.0, 2.0, n_locations))

w_dict = calculate_spatial_weights(
    gauges, {"G03": 5.0, "G07": 3.0}, fill_value=1e-10
)  # every other gauge gets fill_value
```

A **polygon GeoDataFrame** plus `weight_column` triggers a spatial join: the
polygons are reprojected to the gauges' CRS, each gauge collects the value of
every polygon it intersects, and `aggregation` (`"mean"`, `"max"`, `"sum"`, or
`"min"`) reduces multiple hits — `"max"` is a conservative choice for gauges on
zone boundaries. `align_to` reorders the result to match an external id
sequence, essential whenever the GeoDataFrame rows are not already in your
streamflow matrix's column order:

```python
risk_zones = gpd.GeoDataFrame(
    {"risk_score": [10.0, 40.0, 90.0]},
    geometry=[box(0, 0, 2, 2), box(2, 0, 4, 2), box(0, 2, 4, 4)],
    crs="EPSG:4326",
)

w_poly = calculate_spatial_weights(
    gauges, risk_zones, weight_column="risk_score",
    aggregation="max", fill_value=1e-10,
    align_to=gauges["gauge_id"],  # same order here; shown for the pattern
)
```

A **file path** (`str` or `Path`) is read with `geopandas.read_file` and then
treated exactly like the GeoDataFrame case:

```python
import tempfile
from pathlib import Path

tmpdir = Path(tempfile.mkdtemp())
risk_zones.to_file(tmpdir / "risk_zones.geojson", driver="GeoJSON")

w_file = calculate_spatial_weights(
    gauges, tmpdir / "risk_zones.geojson",
    weight_column="risk_score", fill_value=1e-10,
)
```

Two parameters deserve care. `fill_value` replaces NaNs at gauges no polygon
covers: the default `0.0` zeroes those columns and *removes them from
consideration entirely*, while a tiny positive value such as `1e-10` keeps
uncovered locations selectable as a last resort — use `1e-10` for the FEMA
workflow below. `normalize=True` min–max scales weights to `[0, 1]`; it is off
by default because mapping the minimum weight to `0` removes that location too.

## Weighting by FEMA flood risk

These blocks are display-only: they require the FEMA National Risk Index
GeoDatabase, downloaded manually from
<https://www.fema.gov/about/openfema/data-sets/national-risk-index-data> and
then loaded with {func}`hydrosensenet.load_nri`.

```{code-block} python
from hydrosensenet import load_nri

nri = load_nri(
    "NRI_GDB_CensusTracts.gdb",
    scale="tract",              # census tracts; "county" uses the county layer
    columns=("RFLD_RISKS",),    # riverine flood risk score
    mask=basin_boundary,        # optional GeoDataFrame; clips to your basin
)
```

`columns` selects the hazard scores to keep — `RFLD_RISKS` (riverine flooding),
`CFLD_RISKS` (coastal flooding), `HRCN_RISKS` (hurricane), among others; pass
`None` to keep every column. The result is a polygon GeoDataFrame, so it feeds
straight into the workflow above:

```{code-block} python
weights = calculate_spatial_weights(
    gauges, nri, weight_column="RFLD_RISKS", aggregation="mean",
    fill_value=1e-10,                # tracts rarely cover every reach
    align_to=streamflow_df.columns,  # match the matrix's column order
)

result = designer.design_network(n_sensors=50, weights=weights)
```

{meth}`hydrosensenet.SensorNetworkDesigner.design_network` also accepts a file
path plus `weight_column` directly (with `weight_fill_value` to control the
fill for uncovered locations), but it applies no `align_to` — so for the FEMA
workflow compute the array yourself as above and pass the ndarray.

## Checking the effect

Weights should change *which* locations are picked without materially hurting
reconstruction. Compare designs with and without `w_poly`: on this data the
weighted design moves every sensor into the 90-point northern zone while the
median NNSE is essentially unchanged.

```python
from hydrosensenet import SensorNetworkDesigner

designer = SensorNetworkDesigner(streamflow, gauges, list(gauges["gauge_id"]))

plain = designer.design_network(n_sensors=4, evaluate=True, verbose=False)
risky = designer.design_network(n_sensors=4, weights=w_poly, evaluate=True, verbose=False)

for name, res in [("unweighted", plain), ("weighted  ", risky)]:
    nnse = float(np.nanmedian(res.performance_metrics["nnse"]))
    print(name, sorted(res.location_labels), "median NNSE:", round(nnse, 3))
```

If the weighted NNSE drops sharply, the weights are too aggressive — risk is
overriding information content rather than breaking ties; shrink their dynamic
range (take a square root or log of the risk scores) and rerun. The
[evaluation guide](evaluation.md) covers the metrics in detail.

The same two checks on the real Texas-Gulf case study (from the
`design_texas_gulf_risk` example notebook) show the intended behavior at
basin scale:

```{figure} ../_static/risk_cdf.png
:alt: Cumulative distribution of normalized flood-risk index at selected sensor sites for USGS, QR, and risk-weighted QR networks; the risk-weighted curve sits to the right.
:width: 75%

Flood risk at the selected sites: risk-weighted QR (orange) shifts the
network toward high-risk reaches compared to plain QR (green) and the
existing USGS network (gray).
```

```{figure} ../_static/risk_dnnse_hist.png
:alt: Histogram of the per-reach NNSE difference between plain QR and risk-weighted QR, sharply peaked at zero.
:width: 75%

The price of that shift: the per-reach change in reconstruction skill
(ΔNNSE) between the plain and risk-weighted designs is concentrated at
zero — risk weighting redirects sensors without materially degrading
predictions.
```
