# hydrosensenet

**Optimal hydrological monitoring network design, at scale.**

hydrosensenet (*HydroSensorNet*) is an open-source Python package for
designing streamflow monitoring networks. It identifies the sensor
locations that best capture the spatio-temporal structure of a
hydrological system using rank-revealing QR decomposition, so that
streamflow at *ungauged* locations can be reconstructed from a small
number of well-placed gauges.

```{figure} _static/texas_gulf_dnnse.png
:alt: Map of the Texas-Gulf region comparing USGS gauges (black) with QR-optimized sensors (green); river reaches are shaded blue where reconstruction skill improves and red where it declines.
:width: 100%

An optimized network for the Texas-Gulf region (HUC2 12), designed from
National Water Model retrospective data. Blue reaches are ungauged
locations where the QR-based design (green) reconstructs streamflow
better than the existing USGS network (black); produced with the
`design_texas_gulf_baseline` example notebook.
```

## Why hydrosensenet?

Water agencies must monitor ever-larger river networks with limited
budgets. Where should the next gauge go — and which existing gauges
matter most? hydrosensenet answers these questions with a data-driven,
scalable approach:

- **Data-driven placement** — QR column pivoting on streamflow
  time-series matrices (e.g. National Water Model retrospective runs)
  selects the most informative locations, no hydrological model
  calibration required.
- **Scalable** — handles basins with tens of thousands of candidate
  locations on a laptop.
- **Risk-informed** — spatial weights (e.g. FEMA National Risk Index
  flood scores) steer the design toward high-risk communities without
  sacrificing reconstruction accuracy.
- **Adaptive** — existing gauges can be locked in place and the network
  expanded incrementally around them.
- **Batteries included** — loaders for NWM retrospective data, USGS
  gauge networks, and FEMA NRI; Parquet-optimized I/O; evaluation
  metrics (NSE, NNSE, reconstruction error) built in.

The methodology is described in
[Oh & Bartos (2025), *Scalable, Adaptive, and Risk-Informed Design of
Hydrological Sensor Networks*](https://doi.org/10.21203/rs.3.rs-6038740/v1).

## Quick example

```python
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from hydrosensenet import SensorNetworkDesigner

# Streamflow matrix: rows = timesteps, columns = candidate locations
rng = np.random.default_rng(0)
streamflow = rng.uniform(0.5, 1.5, (365, 3)) @ rng.uniform(0.5, 2.0, (3, 50))

locations = gpd.GeoDataFrame(
    {"gauge_id": [f"g{i}" for i in range(50)]},
    geometry=[Point(-97 + 0.05 * i, 30 + 0.02 * i) for i in range(50)],
    crs="EPSG:4326",
)

designer = SensorNetworkDesigner(streamflow, locations)
result = designer.design_network(n_sensors=10, evaluate=True, verbose=False)

result.print_summary()
result.export("sensor_locations.csv")
```

## Where next?

```{toctree}
:maxdepth: 2

installation
quickstart
guide/index
api
```
