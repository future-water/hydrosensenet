# hydrosensenet

An Open-Source Python Package for Optimal Hydrological Monitoring Network Design

**⚠️ In active development**

## Installation

```bash
pip install -e .  # development mode
```

## Quick Start

### Examples

- [design_texas_gulf_baseline.ipynb](design_texas_gulf_baseline.ipynb) - Baseline HUC2-scale design with data download and USGS network comparison
- [design_texas_gulf_flexible.ipynb](design_texas_gulf_flexible.ipynb) - Flexible planning: per-HUC6 batch design across sub-basins
- [design_texas_gulf_risk.ipynb](design_texas_gulf_risk.ipynb) - Risk-aware design using FEMA flood-risk weights

### Basic Usage

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
