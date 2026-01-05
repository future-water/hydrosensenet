# hydrosensenet

An Open-Source Python Package for Optimal Hydrological Monitoring Network Design

**⚠️ In active development**

## Installation

```bash
pip install -e .  # development mode
```

## Quick Start

See [design_texas_gulf.ipynb](design_texas_gulf.ipynb) for a complete example with data download and network comparison.

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
