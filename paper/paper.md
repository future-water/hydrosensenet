---
title: 'hydrosensenet: A Python package for scalable, risk-informed design of hydrological sensor networks'
tags:
  - Python
  - hydrology
  - streamflow
  - sensor placement
  - monitoring network design
  - QR decomposition
authors:
  - name: Jeil Oh
    # TODO: add ORCID iDs for all authors
    corresponding: true
    affiliation: 1
  - name: John Lee
    # TODO: confirm John Lee's affiliation (assumed Center for Water and the Environment, UT Austin)
    affiliation: 1
  - name: Matthew Bartos
    # TODO: confirm Matthew Bartos's affiliation (assumed Center for Water and the Environment, UT Austin)
    affiliation: 1
affiliations:
  - name: Center for Water and the Environment, The University of Texas at Austin, United States
    index: 1
date: 30 July 2026
bibliography: paper.bib
---

# Summary

Water agencies must monitor ever-larger river networks with limited budgets:
where should the next stream gauge go, and which existing gauges matter most?
`hydrosensenet` answers these questions with a data-driven approach. Given a
matrix of streamflow time series at candidate locations — for example, model
output from the NOAA National Water Model (NWM) retrospective
[@nwm_retrospective] — it applies pivoted QR decomposition [@manohar2018] to
select the small set of locations that best captures the spatio-temporal
structure of the basin, so that flows at *ungauged* reaches can be
reconstructed from a few well-placed sensors. The methodology is described in
@oh2025.

The package wraps this algorithm in a workflow built for water-resources
practice. It ingests NWM v3.0 retrospective streamflow directly from cloud
Zarr stores, matches existing USGS gauges to model reaches so that a current
network can be locked in and expanded incrementally, and weights candidate
locations by flood risk (e.g., FEMA National Risk Index scores [@fema_nri]) so
that designs favor high-risk communities without sacrificing reconstruction
accuracy. Built-in evaluation reconstructs held-out flows at every reach and
reports skill metrics such as the Nash–Sutcliffe efficiency. A truncated
greedy factorization path makes HUC2-scale problems (tens of thousands of
reaches) tractable on a laptop, and a bundled example basin lets users run the
full workflow in a few lines of code.

# Statement of need

Long-standing methods for streamflow monitoring network design typically
require calibrated hydrological models or dense historical observations, and
rarely ship as reusable software. Generic QR/POD-based sparse sensor placement
tooling does exist — notably PySensors [@desilva2021], which implements the
algorithms of @manohar2018 for arbitrary data matrices — but it stops at the
linear-algebra layer. A hydrologist who wants to apply these methods to a real
basin must still assemble the forcing data, reconcile it with the existing
gauge network, and encode planning priorities by hand.

`hydrosensenet` closes that gap: to our knowledge it is the first open-source
package that integrates national-scale hydrological forcing (the NWM
retrospective), USGS gauge-network matching, and risk-informed weighting into
a single sensor-placement workflow. The intended users are water-resources
engineers, state and federal monitoring programs, and researchers studying
observation network design. Because placement operates on model-simulated
flows, networks can be designed for basins with few or no existing
observations, and evaluated against the reaches the design leaves ungauged.

# Functionality

- **Pivoted-QR sensor placement** (`sensor_placement_qr`,
  `SensorNetworkDesigner`): deterministic, greedy selection of the most
  informative locations from a `(timesteps, locations)` streamflow matrix,
  with support for locking in existing (e.g., USGS) gauges and for per-region
  sensor quotas.
- **Scalable truncated path**: an O(mnk) truncated greedy pivoted QR performs
  only the k elimination steps needed, selecting sensors identical to the full
  factorization. On a 3,000 × 64,954 (HUC2-scale) matrix with 50 sensors, the
  truncated path runs in ~14 s versus ~168 s for full LAPACK pivoted QR — a
  12x speedup — on an Apple M1 Pro (`benchmarks/bench_placement.py`);
  `float32` input halves memory.
- **NWM ingestion**: `NWMDataLoader` reads NWM v3.0 retrospective streamflow
  from cloud Zarr stores via Dask, subset by reach and time.
- **USGS gauge matching**: utilities download the USGS streamgage dataset with
  its NHDPlus COMID linkage and match gauges to NWM reaches, so existing
  networks can be evaluated or extended.
- **Risk-informed weighting**: `calculate_spatial_weights` builds per-location
  weight vectors from arrays, id-keyed dictionaries, or polygon risk layers,
  including FEMA National Risk Index flood scores, which bias placement
  toward high-risk areas.
- **Evaluation**: temporal train/test splitting and linear reconstruction of
  the full flow field from the selected sensors, scored with NSE/NNSE and
  error metrics at every held-out reach.
- **I/O and examples**: Parquet-optimized readers/writers, loaders for
  CSV/Excel/NetCDF/GRIB/HDF5, CSV/GeoJSON export of designs, and a bundled
  example basin (368 reaches of the lower Colorado River, Texas; two years of
  daily NWM v3.0 flows). The package is validated by a 155-test suite with
  continuous integration and documented at
  <https://future-water.org/hydrosensenet/>.

The package builds on NumPy [@harris2020], SciPy [@virtanen2020], and
GeoPandas [@jordahl2020].

# Example

```python
from hydrosensenet import SensorNetworkDesigner, load_example_basin

streamflow_df, locations_gdf = load_example_basin()
designer = SensorNetworkDesigner(streamflow_df, locations_gdf)
result = designer.design_network(n_sensors=20, evaluate=True)
result.print_summary()
result.export("sensor_locations.geojson")
```

This designs a 20-gauge network for the bundled lower Colorado River basin and
reports how well the held-out year of flows at all 368 reaches is
reconstructed from those 20 sensors.

# Acknowledgements

This work was supported by the CUAHSI Hydroinformatics Fellowship.
<!-- TODO: confirm exact funder wording, fellowship year, and any award number -->

# References
