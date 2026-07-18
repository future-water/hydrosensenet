# hydrosensenet

An Open-Source Python Package for Optimal Hydrological Monitoring Network Design

**⚠️ In active development**

## Installation

Requires Python >= 3.9.

```bash
pip install hydrosensenet

# With optional extras:
pip install "hydrosensenet[viz]"   # map plotting (matplotlib, cartopy)
pip install "hydrosensenet[nwm]"   # NWM streamflow download stack (fsspec, dask, pynhd, ...)
```

The core install covers network design, evaluation, and I/O.
`NetworkDesignResult.plot()` and the NWM download utilities tell you
which extra to install if it is missing.

### Development

```bash
git clone https://github.com/future-water/hydrosensenet.git
cd hydrosensenet
pip install -e ".[dev]"   # editable install with tests, linting, build tooling
pytest                    # run the test suite
```

## Quick Start

### Examples

- [design_texas_gulf_baseline.ipynb](https://github.com/future-water/hydrosensenet/blob/main/design_texas_gulf_baseline.ipynb) - Baseline HUC2-scale design with data download and USGS network comparison
- [design_texas_gulf_flexible.ipynb](https://github.com/future-water/hydrosensenet/blob/main/design_texas_gulf_flexible.ipynb) - Flexible planning: per-HUC6 batch design across sub-basins
- [design_texas_gulf_risk.ipynb](https://github.com/future-water/hydrosensenet/blob/main/design_texas_gulf_risk.ipynb) - Risk-aware design using FEMA flood-risk weights

### Basic Usage

Try it immediately on the bundled sample basin (368 reaches of the
lower Colorado River, Texas — NWM v3.0 daily streamflow):

```python
from hydrosensenet import SensorNetworkDesigner, load_example_basin

streamflow_df, locations_gdf = load_example_basin()
designer = SensorNetworkDesigner(streamflow_df, locations_gdf)
result = designer.design_network(n_sensors=20)
result.print_summary()
```

Or with your own data:

```python
from hydrosensenet import SensorNetworkDesigner
from hydrosensenet.io import load_streamflow
import geopandas as gpd

# Load data
streamflow_df = load_streamflow("streamflow_2000.parquet")
locations_gdf = gpd.read_file("gauges.geojson")

# Design network
designer = SensorNetworkDesigner(
    streamflow_data=streamflow_df.values,
    locations=locations_gdf,
    location_labels=list(streamflow_df.columns)
)
result = designer.design_network(n_sensors=50, evaluate=True)
result.print_summary()
```
