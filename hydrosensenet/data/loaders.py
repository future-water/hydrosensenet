"""Universal data loaders for streamflow and gauge data."""

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path
from typing import Union, List, Optional, Tuple
from shapely.geometry import Point


def load_streamflow_data(
    data_source: Union[str, Path, List[str], pd.DataFrame, xr.Dataset],
    format: str = "auto",
    time_col: Optional[str] = None,
    location_cols: Optional[List[str]] = None,
    **kwargs
) -> Union[pd.DataFrame, xr.Dataset]:
    """Load streamflow data from any source (CSV, Parquet, Excel, NetCDF, etc.)."""
    # Handle already loaded data
    if isinstance(data_source, (pd.DataFrame, xr.Dataset)):
        return data_source

    # Convert to Path
    if isinstance(data_source, (str, Path)):
        data_source = [Path(data_source)]
    elif isinstance(data_source, list):
        data_source = [Path(f) for f in data_source]
    else:
        raise TypeError(
            f"data_source must be str, Path, list, DataFrame, or Dataset, "
            f"got {type(data_source)}"
        )

    # Detect format
    if format == "auto":
        suffix = data_source[0].suffix.lower()
        format_map = {
            '.csv': 'csv',
            '.txt': 'csv',
            '.xlsx': 'excel',
            '.xls': 'excel',
            '.nc': 'netcdf',
            '.nc4': 'netcdf',
            '.grib': 'grib',
            '.grib2': 'grib',
            '.grb': 'grib',
            '.parquet': 'parquet',
            '.pq': 'parquet',
            '.h5': 'hdf5',
            '.hdf5': 'hdf5',
        }
        format = format_map.get(suffix, 'csv')

    # Load based on format
    if format == "csv":
        return _load_csv(data_source, time_col, location_cols, **kwargs)
    elif format == "excel":
        return _load_excel(data_source, time_col, location_cols, **kwargs)
    elif format in ["netcdf", "nc"]:
        return _load_netcdf(data_source, **kwargs)
    elif format == "grib":
        return _load_grib(data_source, **kwargs)
    elif format == "parquet":
        return _load_parquet(data_source, time_col, location_cols, **kwargs)
    elif format in ["hdf5", "h5"]:
        return _load_hdf5(data_source, time_col, location_cols, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _load_csv(
    files: List[Path],
    time_col: Optional[str],
    location_cols: Optional[List[str]],
    **kwargs
) -> pd.DataFrame:
    """Load CSV file(s)."""
    dfs = []
    for file in files:
        # Try to auto-detect time column
        df = pd.read_csv(file, **kwargs)

        # Handle common unnamed index column
        if df.columns[0].startswith('Unnamed'):
            df = pd.read_csv(file, index_col=0, parse_dates=True, **kwargs)
        elif time_col:
            df = pd.read_csv(file, index_col=time_col, parse_dates=True, **kwargs)
        else:
            # Auto-detect common time column names
            time_aliases = ['time', 'Time', 'datetime', 'date', 'Date', 'timestamp']
            for alias in time_aliases:
                if alias in df.columns:
                    df = pd.read_csv(file, index_col=alias, parse_dates=True, **kwargs)
                    break

        dfs.append(df)

    # Concatenate if multiple files
    if len(dfs) > 1:
        result = pd.concat(dfs, axis=0)
    else:
        result = dfs[0]

    # Select location columns if specified
    if location_cols:
        result = result[location_cols]

    return result


def _load_excel(
    files: List[Path],
    time_col: Optional[str],
    location_cols: Optional[List[str]],
    **kwargs
) -> pd.DataFrame:
    """Load Excel file(s)."""
    dfs = []
    for file in files:
        if time_col:
            df = pd.read_excel(file, index_col=time_col, parse_dates=True, **kwargs)
        else:
            df = pd.read_excel(file, **kwargs)
        dfs.append(df)

    if len(dfs) > 1:
        result = pd.concat(dfs, axis=0)
    else:
        result = dfs[0]

    if location_cols:
        result = result[location_cols]

    return result


def _load_netcdf(files: List[Path], **kwargs) -> xr.Dataset:
    """Load NetCDF file(s)."""
    if len(files) == 1:
        return xr.open_dataset(files[0], **kwargs)
    else:
        # Concatenate multiple files
        datasets = [xr.open_dataset(f, **kwargs) for f in files]
        return xr.concat(datasets, dim="time").sortby("time")


def _load_grib(files: List[Path], **kwargs) -> xr.Dataset:
    """Load GRIB file(s)."""
    # Set default engine for GRIB
    if 'engine' not in kwargs:
        kwargs['engine'] = 'cfgrib'
    if 'backend_kwargs' not in kwargs:
        kwargs['backend_kwargs'] = {'indexpath': ''}

    if len(files) == 1:
        return xr.open_dataset(files[0], **kwargs)
    else:
        datasets = [xr.open_dataset(f, **kwargs) for f in files]
        return xr.concat(datasets, dim="time").sortby("time")


def _load_parquet(
    files: List[Path],
    time_col: Optional[str],
    location_cols: Optional[List[str]],
    **kwargs
) -> pd.DataFrame:
    """Load Parquet file(s)."""
    dfs = [pd.read_parquet(f, **kwargs) for f in files]
    result = pd.concat(dfs, axis=0) if len(dfs) > 1 else dfs[0]

    if time_col and time_col in result.columns:
        result = result.set_index(time_col)

    if location_cols:
        result = result[location_cols]

    return result


def _load_hdf5(
    files: List[Path],
    time_col: Optional[str],
    location_cols: Optional[List[str]],
    **kwargs
) -> pd.DataFrame:
    """Load HDF5 file(s)."""
    dfs = [pd.read_hdf(f, **kwargs) for f in files]
    result = pd.concat(dfs, axis=0) if len(dfs) > 1 else dfs[0]

    if time_col and time_col in result.columns:
        result = result.set_index(time_col)

    if location_cols:
        result = result[location_cols]

    return result


def prepare_gauge_locations(
    gauge_data: Union[str, Path, pd.DataFrame, gpd.GeoDataFrame],
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    id_col: Optional[str] = None,
    crs: str = "EPSG:4326"
) -> gpd.GeoDataFrame:
    """Prepare gauge location data as GeoDataFrame."""
    # If already GeoDataFrame, return as-is
    if isinstance(gauge_data, gpd.GeoDataFrame):
        return gauge_data

    # Load data
    if isinstance(gauge_data, (str, Path)):
        path = Path(gauge_data)
        if path.suffix.lower() in ['.shp', '.geojson', '.gpkg']:
            gdf = gpd.read_file(path)
            if gdf.crs != crs:
                gdf = gdf.to_crs(crs)
            return gdf
        elif path.suffix.lower() in ['.csv', '.txt']:
            df = pd.read_csv(path)
        elif path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    elif isinstance(gauge_data, pd.DataFrame):
        df = gauge_data.copy()
    else:
        raise TypeError(
            f"gauge_data must be str, Path, DataFrame, or GeoDataFrame, "
            f"got {type(gauge_data)}"
        )

    # Find lat/lon columns (support common variations)
    lat_aliases = [lat_col, 'lat', 'Latitude', 'LAT', 'gauge_lat', 'y', 'Y']
    lon_aliases = [lon_col, 'lon', 'Longitude', 'LON', 'gauge_lon', 'x', 'X']

    actual_lat = None
    actual_lon = None

    for alias in lat_aliases:
        if alias in df.columns:
            actual_lat = alias
            break

    for alias in lon_aliases:
        if alias in df.columns:
            actual_lon = alias
            break

    if actual_lat is None or actual_lon is None:
        raise ValueError(
            f"Could not find latitude/longitude columns. "
            f"Available columns: {list(df.columns)}"
        )

    # Drop rows with missing coordinates
    df = df.dropna(subset=[actual_lat, actual_lon])

    # Rename to standard names
    df = df.rename(columns={actual_lat: "latitude", actual_lon: "longitude"})

    # Create or rename ID column
    if id_col is None:
        df["gauge_id"] = np.arange(len(df))
    elif id_col in df.columns:
        df = df.rename(columns={id_col: "gauge_id"})
    else:
        df["gauge_id"] = np.arange(len(df))

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs=crs
    )

    return gdf
