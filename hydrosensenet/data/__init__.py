"""Data loading and preprocessing for any streamflow data source."""

from .loaders import load_streamflow_data, prepare_gauge_locations
from .preprocessors import split_timeseries, prepare_matrix, filter_valid_data, clip_to_region, match_gauges_to_grid, count_gauges_per_basin, grid_to_point_gdf, assign_basins
from .nwm import NWMDataLoader, get_huc_info, list_available_hucs
from .usgs import download_usgs_gauges, load_usgs_gauges, match_usgs_to_nwm
from .glofas import GLOFASDataLoader, get_glofas_info
from .brazil import download_brazil_gauges, download_brazil_watersheds, download_country_boundary

__all__ = [
    "load_streamflow_data",
    "prepare_gauge_locations",
    "split_timeseries",
    "prepare_matrix",
    "filter_valid_data",
    "clip_to_region",
    "match_gauges_to_grid",
    "count_gauges_per_basin",
    "grid_to_point_gdf",
    "assign_basins",
    "NWMDataLoader",
    "get_huc_info",
    "list_available_hucs",
    "download_usgs_gauges",
    "load_usgs_gauges",
    "match_usgs_to_nwm",
    "GLOFASDataLoader",
    "get_glofas_info",
    "download_brazil_gauges",
    "download_brazil_watersheds",
    "download_country_boundary"
]
