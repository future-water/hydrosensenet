"""Data preprocessing utilities for sensor network optimization."""

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from typing import Union, Tuple, Dict, Optional, List
from shapely.geometry import Point


def split_timeseries(
    data: Union[np.ndarray, pd.DataFrame, xr.Dataset],
    train_frac: float = 0.7,
    filter_invalid: bool = True,
    return_mapping: bool = False
) -> Union[Tuple, Tuple[any, any, Dict]]:
    """Split time series data into training and testing sets."""
    if isinstance(data, np.ndarray):
        return _split_array(data, train_frac, filter_invalid, return_mapping)
    elif isinstance(data, pd.DataFrame):
        return _split_dataframe(data, train_frac, filter_invalid, return_mapping)
    elif isinstance(data, xr.Dataset):
        return _split_dataset(data, train_frac, filter_invalid, return_mapping)
    else:
        raise TypeError(
            f"data must be np.ndarray, pd.DataFrame, or xr.Dataset, "
            f"got {type(data)}"
        )


def _split_array(
    data: np.ndarray,
    train_frac: float,
    filter_invalid: bool,
    return_mapping: bool
):
    """Split numpy array."""
    n_train = int(train_frac * data.shape[0])
    train_data = data[:n_train, :]
    test_data = data[n_train:, :]

    if filter_invalid:
        # Filter columns with NaN/inf in training
        finite_mask = np.isfinite(train_data).all(axis=0)
        good_cols = np.where(finite_mask)[0]

        train_data = train_data[:, good_cols]
        test_data = test_data[:, good_cols]

        if return_mapping:
            mapping = {
                "old_to_new": {old: new for new, old in enumerate(good_cols)},
                "good_cols": good_cols,
                "n_removed": data.shape[1] - len(good_cols)
            }
            return train_data, test_data, mapping

    if return_mapping:
        mapping = {
            "old_to_new": {i: i for i in range(data.shape[1])},
            "good_cols": np.arange(data.shape[1]),
            "n_removed": 0
        }
        return train_data, test_data, mapping

    return train_data, test_data


def _split_dataframe(
    data: pd.DataFrame,
    train_frac: float,
    filter_invalid: bool,
    return_mapping: bool
):
    """Split pandas DataFrame."""
    n_train = int(train_frac * len(data))
    train_data = data.iloc[:n_train, :]
    test_data = data.iloc[n_train:, :]

    if filter_invalid:
        # Filter columns with NaN/inf in training
        finite_mask = np.isfinite(train_data.values).all(axis=0)
        good_cols = data.columns[finite_mask]

        train_data = train_data[good_cols]
        test_data = test_data[good_cols]

        if return_mapping:
            mapping = {
                "good_cols": good_cols,
                "removed_cols": data.columns[~finite_mask],
                "n_removed": (~finite_mask).sum()
            }
            return train_data, test_data, mapping

    if return_mapping:
        mapping = {"good_cols": data.columns, "removed_cols": [], "n_removed": 0}
        return train_data, test_data, mapping

    return train_data, test_data


def _split_dataset(
    data: xr.Dataset,
    train_frac: float,
    filter_invalid: bool,
    return_mapping: bool
):
    """Split xarray Dataset."""
    time_dim = "time"  # Assume time dimension is named "time"
    n_train = int(train_frac * len(data[time_dim]))

    train_data = data.isel({time_dim: slice(0, n_train)})
    test_data = data.isel({time_dim: slice(n_train, None)})

    # Filtering for xarray is more complex, skipping for now
    if return_mapping:
        mapping = {"n_removed": 0}
        return train_data, test_data, mapping

    return train_data, test_data

def clip_to_region(
    data: Union[pd.DataFrame, xr.Dataset],
    boundary_poly,
    extent: List[float],
    lon_col: Optional[str] = None,
    lat_col: Optional[str] = None,
    fix_longitude: bool = True,
    round_coords: Optional[int] = 3  # Add rounding parameter
) -> Union[pd.DataFrame, xr.Dataset]:
    """
    Clip data to a specific geographic region.
    
    Parameters
    ----------
    data : DataFrame or Dataset
        Input data with geographic coordinates.
        - DataFrame: must have longitude/latitude columns or MultiIndex
        - Dataset: uses 'longitude' and 'latitude' dimensions
    boundary_poly : shapely.geometry.Polygon
        Polygon for clipping.
    extent : list of float
        Bounding box as [west, east, south, north].
    lon_col : str, optional
        Column name for longitude (DataFrame only). 
        If None, looks for 'longitude', 'lon', or 'x' in columns/index.
    lat_col : str, optional
        Column name for latitude (DataFrame only).
        If None, looks for 'latitude', 'lat', or 'y' in columns/index.
    fix_longitude : bool, default=True
        Convert longitude from [0, 360] to [-180, 180] range.
    round_coords : int, optional, default=3
        Number of decimal places to round coordinates.
        Set to None to skip rounding.
    
    Returns
    -------
    DataFrame or Dataset
        Clipped data in the same format as input.
    
    Examples
    --------
    >>> # xarray Dataset
    >>> ds_clipped = clip_to_region(ds, poly, [-10, 10, 30, 50])
    
    >>> # DataFrame with columns
    >>> df = pd.DataFrame({'lon': [...], 'lat': [...], 'value': [...]})
    >>> df_clipped = clip_to_region(df, poly, [-10, 10, 30, 50])
    
    >>> # No rounding
    >>> df_clipped = clip_to_region(df, poly, extent, round_coords=None)
    """
    if isinstance(data, xr.Dataset):
        return _clip_xarray_to_region(data, boundary_poly, extent, fix_longitude, round_coords)
    
    elif isinstance(data, pd.DataFrame):
        return _clip_dataframe_to_region(
            data, boundary_poly, extent, lon_col, lat_col, fix_longitude, round_coords
        )
    
    else:
        raise TypeError(f"data must be DataFrame or Dataset, got {type(data)}")


def _clip_xarray_to_region(
    ds: xr.Dataset,
    boundary_poly,
    extent: List[float],
    fix_longitude: bool,
    round_coords: Optional[int]
) -> xr.Dataset:
    """Helper function to clip xarray Dataset."""
    # Fix longitude coordinates if needed
    if fix_longitude:
        lon_values = ((ds.longitude.values + 180) % 360) - 180
        if round_coords is not None:
            lon_values = np.round(lon_values, round_coords)
        ds = ds.assign_coords(longitude=lon_values)
    
    # Round latitude if needed
    if round_coords is not None:
        ds = ds.assign_coords(
            latitude=np.round(ds.latitude.values, round_coords)
        )
    
    # Set CRS
    ds = ds.rio.write_crs("EPSG:4326", inplace=True)
    
    # Apply bounding box
    ds = ds.sel(
        longitude=slice(extent[0], extent[1]),
        latitude=slice(extent[3], extent[2])
    )
    
    # Clip to polygon
    ds = ds.rio.clip([boundary_poly], "EPSG:4326", drop=True)
    
    return ds


def _clip_dataframe_to_region(
    df: pd.DataFrame,
    boundary_poly,
    extent: List[float],
    lon_col: Optional[str],
    lat_col: Optional[str],
    fix_longitude: bool,
    round_coords: Optional[int]
) -> pd.DataFrame:
    # Identify longitude and latitude columns
    lon_col, lat_col = _identify_coordinate_columns(df, lon_col, lat_col)
    
    # Get coordinate values
    if lon_col in df.columns:
        lons = df[lon_col].values.copy()
        lats = df[lat_col].values.copy()
        using_index = False
    else:
        # Coordinates are in the index
        if isinstance(df.index, pd.MultiIndex):
            lons = df.index.get_level_values(lon_col).values.copy()
            lats = df.index.get_level_values(lat_col).values.copy()
        else:
            raise ValueError(f"Could not find '{lon_col}' in columns or index")
        using_index = True
    
    # Fix longitude if needed
    if fix_longitude:
        lons = ((lons + 180) % 360) - 180
    
    # Round coordinates
    if round_coords is not None:
        lons = np.round(lons, round_coords)
        lats = np.round(lats, round_coords)
    
    # Apply bounding box filter
    bbox_mask = (
        (lons >= extent[0]) & (lons <= extent[1]) &
        (lats >= extent[2]) & (lats <= extent[3])
    )
    
    df_clipped = df[bbox_mask].copy()
    
    # Update coordinates in the clipped dataframe
    if using_index:
        # Reconstruct index with updated coordinates
        if isinstance(df_clipped.index, pd.MultiIndex):
            index_data = {
                name: df_clipped.index.get_level_values(name).values
                for name in df_clipped.index.names
            }
            index_data[lon_col] = lons[bbox_mask]
            index_data[lat_col] = lats[bbox_mask]
            df_clipped.index = pd.MultiIndex.from_arrays(
                [index_data[name] for name in df_clipped.index.names],
                names=df_clipped.index.names
            )
    else:
        df_clipped[lon_col] = lons[bbox_mask]
        df_clipped[lat_col] = lats[bbox_mask]
    
    # Get updated coordinates after bbox clipping
    if using_index:
        lons_clipped = df_clipped.index.get_level_values(lon_col).values
        lats_clipped = df_clipped.index.get_level_values(lat_col).values
    else:
        lons_clipped = df_clipped[lon_col].values
        lats_clipped = df_clipped[lat_col].values
    
    # Apply polygon clipping
    points = [Point(lon, lat) for lon, lat in zip(lons_clipped, lats_clipped)]
    poly_mask = [boundary_poly.contains(pt) for pt in points]
    
    df_clipped = df_clipped[poly_mask]
    
    return df_clipped

def _identify_coordinate_columns(
    df: pd.DataFrame,
    lon_col: Optional[str],
    lat_col: Optional[str]
) -> Tuple[str, str]:
    """Identify longitude and latitude column names."""
    # Check both columns and index
    available = list(df.columns)
    if isinstance(df.index, pd.MultiIndex):
        available.extend(df.index.names)
    elif df.index.name:
        available.append(df.index.name)
    
    # Find longitude column
    if lon_col is None:
        lon_candidates = ['longitude', 'lon', 'x', 'Longitude', 'LON']
        lon_col = next((c for c in lon_candidates if c in available), None)
        if lon_col is None:
            raise ValueError(
                f"Could not find longitude column. Available: {available}. "
                "Specify using lon_col parameter."
            )
    
    # Find latitude column
    if lat_col is None:
        lat_candidates = ['latitude', 'lat', 'y', 'Latitude', 'LAT']
        lat_col = next((c for c in lat_candidates if c in available), None)
        if lat_col is None:
            raise ValueError(
                f"Could not find latitude column. Available: {available}. "
                "Specify using lat_col parameter."
            )
    
    return lon_col, lat_col

def prepare_matrix(
    data: Union[pd.DataFrame, xr.Dataset],
    variable: Optional[str] = None,
    drop_na: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    Convert DataFrame or xarray Dataset to 2D matrix for optimization.

    Parameters
    ----------
    data : DataFrame or Dataset
        Input data.
        - DataFrame: columns are locations, rows are time
        - Dataset: will stack spatial dimensions
    variable : str, optional
        Variable name to extract from Dataset (required for Dataset input).
    drop_na : bool, default=True
        Drop locations with any NaN values.

    Returns
    -------
    matrix : np.ndarray
        2D array of shape (n_timesteps, n_locations).
    location_labels : list of str
        Labels for each location (column).

    Examples
    --------
    >>> # DataFrame input
    >>> df = pd.DataFrame(np.random.randn(365, 100))
    >>> matrix, labels = prepare_matrix(df)

    >>> # xarray Dataset input
    >>> ds = xr.Dataset(...)
    >>> matrix, labels = prepare_matrix(ds, variable="discharge")
    """
    if isinstance(data, pd.DataFrame):
        if drop_na:
            data = data.dropna(axis=1)
        matrix = data.values
        location_labels = list(data.columns)

    elif isinstance(data, xr.Dataset):
        if variable is None:
            # Try to find the first data variable
            variable = list(data.data_vars)[0]

        data_array = data[variable]

        # Stack spatial dimensions
        spatial_dims = [d for d in data_array.dims if d != 'time']

        if len(spatial_dims) == 0:
            raise ValueError("No spatial dimensions found")

        stacked = data_array.stack(location=spatial_dims)

        if drop_na:
            stacked = stacked.dropna(dim='location', how='any')

        matrix = stacked.values
        location_labels = [str(loc) for loc in stacked.location.values]

    else:
        raise TypeError(f"data must be DataFrame or Dataset, got {type(data)}")

    return matrix, location_labels

def filter_valid_data(
    data: np.ndarray,
    location_labels: Optional[List[str]] = None,
    min_valid_fraction: float = 0.8,
    remove_zeros: bool = False,
    return_mapping: bool = True
) -> Union[Tuple[np.ndarray, np.ndarray], 
           Tuple[np.ndarray, np.ndarray, List[str]],
           Tuple[np.ndarray, np.ndarray, Dict],
           Tuple[np.ndarray, np.ndarray, List[str], Dict]]:
    """
    Filter locations based on data quality.

    Parameters
    ----------
    data : np.ndarray
        Data matrix of shape (n_timesteps, n_locations).
    location_labels : list of str, optional
        Labels for each location. If provided, will be filtered to match.
    min_valid_fraction : float, default=0.8
        Minimum fraction of valid (non-NaN) values required.
    remove_zeros : bool, default=False
        Remove locations with all zeros (dry streams).
    return_mapping : bool, default=True
        If True, return mapping dictionary for use with grid_to_point_gdf.

    Returns
    -------
    filtered_data : np.ndarray
        Filtered data matrix.
    valid_indices : np.ndarray
        Indices of kept locations.
    filtered_labels : list of str (only if location_labels provided)
        Labels corresponding to filtered locations.
    mapping : dict (only if return_mapping=True)
        Dictionary with filtering metadata including:
        - 'good_cols': indices of kept locations
        - 'old_to_new': mapping from original to new indices
        - 'n_removed': number of removed locations
        - 'valid_lat_lon': filtered location labels (if location_labels provided)
    """
    n_timesteps = data.shape[0]

    # Check valid fraction
    valid_fraction = np.isfinite(data).sum(axis=0) / n_timesteps
    valid_mask = valid_fraction >= min_valid_fraction

    # Optionally remove all-zero locations
    if remove_zeros:
        nonzero_mask = (data != 0).any(axis=0)
        valid_mask = valid_mask & nonzero_mask

    valid_indices = np.where(valid_mask)[0]
    filtered_data = data[:, valid_indices]

    print(
        f"Filtered data: kept {len(valid_indices)} of {data.shape[1]} locations "
        f"({100*len(valid_indices)/data.shape[1]:.1f}%)"
    )

    # Create mapping dictionary
    if return_mapping:
        mapping = {
            "good_cols": valid_indices,
            "old_to_new": {old: new for new, old in enumerate(valid_indices)},
            "n_removed": data.shape[1] - len(valid_indices)
        }
        
        # Add filtered labels to mapping if available
        if location_labels is not None:
            filtered_labels = [location_labels[i] for i in valid_indices]
            mapping["valid_lat_lon"] = filtered_labels
            return filtered_data, valid_indices, filtered_labels, mapping
        else:
            return filtered_data, valid_indices, mapping
    
    # No mapping
    if location_labels is not None:
        filtered_labels = [location_labels[i] for i in valid_indices]
        return filtered_data, valid_indices, filtered_labels
            
    return filtered_data, valid_indices

def match_gauges_to_grid(gauge_gdf: gpd.GeoDataFrame, 
                        lat_vals: np.ndarray, 
                        lon_vals: np.ndarray,
                        valid_lat_lon: List[str]) -> Tuple[np.ndarray, gpd.GeoDataFrame]:
    """
    Match gauge locations to the nearest grid cells.
    
    Parameters:
    - gauge_gdf: GeoDataFrame with gauge locations
    - lat_vals: Array of latitude values from the grid
    - lon_vals: Array of longitude values from the grid
    - valid_lat_lon: List of valid lat/lon labels
    
    Returns:
    - Tuple of (original_indices, matched gauge point gdf)
    """
    def nearest_idx(axis_vals, pts):
        return np.abs(axis_vals[:, None] - pts).argmin(axis=0)
    
    # Create lookup table
    lookup = pd.DataFrame({
        "lat_c": [float(s.split(',')[0][1:]) for s in valid_lat_lon],
        "lon_c": [float(s.split(',')[1][:-1]) for s in valid_lat_lon],
        "matrix_col": np.arange(len(valid_lat_lon)),
    }).round(4)
    
    # Find nearest grid cells for gauges
    lat_idx = nearest_idx(lat_vals, gauge_gdf.gauge_lat.values)
    lon_idx = nearest_idx(lon_vals, gauge_gdf.gauge_lon.values)
    
    gauges = pd.DataFrame({
        "gauge_id": gauge_gdf.gauge_id.values,
        "lat_c": lat_vals[lat_idx],
        "lon_c": lon_vals[lon_idx],
    }).round(4)
    
    sensor_cols = gauges.merge(lookup, on=["lat_c", "lon_c"], how="inner")
    sensor_column_indices_orig = sensor_cols.matrix_col.to_numpy()

    matched_gauge_ids = sensor_cols['gauge_id'].values
    gauge_gdf_matched = gauge_gdf[gauge_gdf['gauge_id'].isin(matched_gauge_ids)].copy()
    
    print(f"{sensor_column_indices_orig.size} gauges matched to grid cells")
    return sensor_column_indices_orig, gauge_gdf_matched

def count_gauges_per_basin(gauge_gdf: gpd.GeoDataFrame,
                          basin_gdf: Optional[gpd.GeoDataFrame] = None,
                          basin_id_col: str = None,
                          basin_name_col: str = None,
                          basin_crs: str = None,
                          country_name: str = "Country",
                          total_gauges: int = None) -> Dict:
    """
    Count the number of gauges in each basin.
    
    Parameters:
    - gauge_gdf: GeoDataFrame with gauge locations
    - basin_gdf: GeoDataFrame with basin polygons (optional)
    - country_name: Name to use if treating whole area as one basin
    - total_gauges: Total number of gauges (used when basin_gdf is None)
    
    Returns:
    - Dictionary mapping basin names to gauge counts
    """
    if basin_gdf is not None:
        # Build a case-insensitive column map
        if basin_id_col is None or basin_name_col is None:
            raise KeyError("Required columns basin id and/or basin name not found (case-insensitive).")
    
        gauges_with_basin = gpd.sjoin(
            gauge_gdf[["gauge_id", "geometry"]],
            basin_gdf[[basin_id_col, basin_name_col, "geometry"]],
            how="left",
            predicate="within",
        )
        
        if basin_id_col not in gauges_with_basin.columns:
            gauges_with_basin[basin_id_col] = np.nan
        if basin_name_col not in gauges_with_basin.columns:
            gauges_with_basin[basin_name_col] = None
            
        unassigned_mask = gauges_with_basin[basin_id_col].isna()
        n_unassigned = unassigned_mask.sum()
        
        if n_unassigned > 0:
            print(f"Found {n_unassigned} gauges not within any basin. Assigning to nearest basin...")
        
            # Get unassigned indices
            unassigned_indices = gauges_with_basin[unassigned_mask].index
            
            crs_projected = basin_crs
            gauges_proj = gauges_with_basin.loc[unassigned_mask].to_crs(crs_projected)
            basins_proj = basin_gdf.to_crs(crs_projected)
            
            # Find nearest basin for each unassigned gauge
            for idx in unassigned_indices:
                gauge_geom = gauges_proj.loc[idx, "geometry"]
                distances = basins_proj.geometry.distance(gauge_geom)
                nearest_idx = distances.idxmin()
                
                gauges_with_basin.at[idx, basin_id_col] = basin_gdf.at[nearest_idx, basin_id_col]
                gauges_with_basin.at[idx, basin_name_col] = basin_gdf.at[nearest_idx, basin_name_col]
        
        if "index_right" in gauges_with_basin.columns:
            gauges_with_basin = gauges_with_basin.drop(columns=["index_right"])
            
        gauge_counts = gauges_with_basin.groupby(basin_name_col).size().to_dict()
        print(f"Total gauges assigned: {sum(gauge_counts.values())}")
    else:
        gauge_counts = {country_name: total_gauges or len(gauge_gdf)}
        
    return gauge_counts

def grid_to_point_gdf(mapping_dict: Dict) -> gpd.GeoDataFrame:
    """
    Create GeoDataFrame of grid cell centroids that survived filtering.
    
    Parameters:
    - mapping_dict: Dictionary from filter_valid_data with keys:
        - 'valid_lat_lon': List of valid lat/lon labels
        - 'good_cols': Array of indices of kept locations
        - 'old_to_new': Mapping from original to new indices
    
    Returns:
    - GeoDataFrame with point geometries
    """
    valid_lat_lon = mapping_dict["valid_lat_lon"]

    coords = np.column_stack((
        [float(s.split(',')[0][1:]) for s in valid_lat_lon],
        [float(s.split(',')[1][:-1]) for s in valid_lat_lon],
    ))
    
    points_gdf = gpd.GeoDataFrame(
        {
            "matrix_col": mapping_dict["good_cols"],
            "col_pos": np.arange(len(valid_lat_lon)),
            "lat": coords[:, 0],
            "lon": coords[:, 1],
        },
        geometry=[Point(lon, lat) for lat, lon in coords],
        crs="EPSG:4326",
    )
    
    return points_gdf

def assign_basins(points_gdf: gpd.GeoDataFrame, 
                 basin_gdf: Optional[gpd.GeoDataFrame] = None,
                 basin_id_col: str = None,
                 basin_name_col: str = None,
                 basin_crs: str = None,
                 country_name: str = "Country") -> gpd.GeoDataFrame:
    """
    Assign basin information to grid points.
    
    Parameters:
    - points_gdf: GeoDataFrame of grid points
    - basin_gdf: GeoDataFrame with basin polygons (optional)
    - country_name: Name to use if treating whole area as one basin
    
    Returns:
    - GeoDataFrame with basin assignments
    """
    if basin_gdf is not None:
        # First, ensure basin_gdf has the required columns
        if basin_id_col not in basin_gdf.columns or basin_name_col not in basin_gdf.columns:
            raise ValueError("basin_gdf must have basin id and basin name columns")
        
        # Use left join to keep all points
        points_with_basin = gpd.sjoin(
            points_gdf,
            basin_gdf[[basin_id_col, basin_name_col, "geometry"]],
            how="left",
            predicate="within",
        )
        
        # Initialize columns if they don't exist
        if basin_id_col not in points_with_basin.columns:
            points_with_basin[basin_id_col] = np.nan
        if basin_name_col not in points_with_basin.columns:
            points_with_basin[basin_name_col] = None
            
        # For points that didn't match any basin, assign them to the nearest basin
        unassigned_mask = points_with_basin[basin_id_col].isna()
        n_unassigned = unassigned_mask.sum()
        
        if n_unassigned > 0:
            print(f"Found {n_unassigned} points not within any basin. Assigning to nearest basin...")
    
            crs_projected = basin_crs
            
            unassigned_indices = points_with_basin[unassigned_mask].index
            
            # Project geometries for distance calculation
            points_proj = points_with_basin.loc[unassigned_mask].to_crs(crs_projected)
            basins_proj = basin_gdf.to_crs(crs_projected)
            
            # For each unassigned point, find nearest basin
            for idx in unassigned_indices:
                point_geom = points_proj.loc[idx, "geometry"]
                distances = basins_proj.geometry.distance(point_geom)
                nearest_idx = distances.idxmin()
                
                points_with_basin.at[idx, basin_id_col] = basin_gdf.at[nearest_idx, basin_id_col]
                points_with_basin.at[idx, basin_name_col] = basin_gdf.at[nearest_idx, basin_name_col]
        
        if "index_right" in points_with_basin.columns:
            points_with_basin = points_with_basin.drop(columns=["index_right"])
            
    else:
        points_with_basin = points_gdf.copy()
        points_with_basin[basin_id_col] = 0
        points_with_basin[basin_name_col] = country_name
        
    return points_with_basin