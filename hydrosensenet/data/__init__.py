"""Data loading and preprocessing for any streamflow data source."""

from .fema import load_nri
from .loaders import load_streamflow_data, prepare_gauge_locations
from .nwm import NWMDataLoader, get_huc_info, list_available_hucs
from .usgs import download_usgs_gauges, load_usgs_gauges, match_usgs_to_nwm
from .glofas import GloFASDataLoader
from .brazil import download_brazil_gauges, download_brazil_watersheds, download_country_boundary
from .preprocessors import (
    filter_valid_data, 
    prepare_matrix, 
    split_timeseries,
    clip_to_region,
    split_and_filter_data,
    match_gauges_to_grid,
    grid_to_point_gdf,
    assign_basins,
    count_gauges_per_basin,
)

__all__ = [
    "load_streamflow_data",
    "prepare_gauge_locations",
    "split_timeseries",
    "prepare_matrix",
    "filter_valid_data",
    "NWMDataLoader",
    "get_huc_info",
    "list_available_hucs",
    "download_usgs_gauges",
    "load_usgs_gauges",
    "match_usgs_to_nwm",
    "load_nri",
    "GloFASDataLoader",
    "download_brazil_gauges",
    "download_brazil_watersheds",
    "download_country_boundary",
    "clip_to_region",
    "split_and_filter_data",
    "match_gauges_to_grid",
    "grid_to_point_gdf",
    "assign_basins",
    "count_gauges_per_basin",
]
