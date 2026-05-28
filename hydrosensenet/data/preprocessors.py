"""Data preprocessing utilities for sensor network optimization.

This is the GitHub-original preprocessors.py with GloFAS support added:
  - split_timeseries / _split_* / filter_valid_data : ORIGINAL, unchanged
    (the three Texas-Gulf notebooks and data/__init__.py rely on these).
  - prepare_matrix : ENHANCED so gridded (latitude, longitude) data gets
    "(lat, lon)" string labels rounded to `round_coords` decimals. The GloFAS
    notebook's gauge-to-grid matching depends on this label format. For
    DataFrame input the behaviour is identical to the original.
  - clip_to_region ... count_gauges_per_basin : NEW, used only by the GloFAS
    notebook. None of these are imported by the Texas-Gulf notebooks.
"""

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
from typing import Union, Tuple, Dict, Optional, List


# =========================================================================
# Train/test split  (ORIGINAL — unchanged)
# =========================================================================

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


# =========================================================================
# Matrix preparation  (ENHANCED for gridded "(lat, lon)" labels)
# =========================================================================

def prepare_matrix(
    data: Union[pd.DataFrame, xr.Dataset],
    variable: Optional[str] = None,
    drop_na: bool = True,
    round_coords: Optional[int] = 4,
) -> Tuple[np.ndarray, List[str]]:
    """
    Convert tabular or gridded data to a (time x locations) matrix.

    DataFrame : columns are locations, rows are time.
    Dataset   : non-time dims are stacked. For (latitude, longitude) data,
                labels are formatted "(lat, lon)" rounded to `round_coords`
                decimals — this matters when matching gauge coords later.

    Returns (matrix, location_labels).
    """
    if isinstance(data, pd.DataFrame):
        if drop_na:
            data = data.dropna(axis=1)
        return data.values, list(data.columns)

    if isinstance(data, xr.Dataset):
        if variable is None:
            variable = list(data.data_vars)[0]
        da = data[variable]
        spatial_dims = [d for d in da.dims if d != "time"]
        if not spatial_dims:
            raise ValueError("no spatial dimensions found")

        stacked = da.stack(location=spatial_dims)
        if drop_na:
            stacked = stacked.dropna(dim="location", how="all")

        # Lat/lon get readable rounded labels; other coord pairs fall back to str()
        if set(spatial_dims) == {"latitude", "longitude"} and round_coords is not None:
            lats = np.round(stacked.latitude.values, round_coords)
            lons = np.round(stacked.longitude.values, round_coords)
            labels = [f"({lat}, {lon})" for lat, lon in zip(lats, lons)]
        else:
            labels = [str(loc) for loc in stacked.location.values]

        return stacked.values.astype(np.float32), labels

    raise TypeError(f"data must be DataFrame or Dataset, got {type(data)}")


# =========================================================================
# Data-quality filter  (ORIGINAL — unchanged; used by data/__init__.py)
# =========================================================================

def filter_valid_data(
    data: np.ndarray,
    min_valid_fraction: float = 0.8,
    remove_zeros: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter locations based on data quality.

    Parameters
    ----------
    data : np.ndarray
        Data matrix of shape (n_timesteps, n_locations).
    min_valid_fraction : float, default=0.8
        Minimum fraction of valid (non-NaN) values required.
    remove_zeros : bool, default=False
        Remove locations with all zeros (dry streams).

    Returns
    -------
    filtered_data : np.ndarray
        Filtered data matrix.
    valid_indices : np.ndarray
        Indices of kept locations.
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

    return filtered_data, valid_indices


# =========================================================================
# GloFAS: region clipping  (NEW)
# =========================================================================

def clip_to_region(
    data: Union[pd.DataFrame, xr.Dataset],
    boundary_poly,
    extent: List[float],
    lon_col: Optional[str] = None,
    lat_col: Optional[str] = None,
    fix_longitude: bool = True,
) -> Union[pd.DataFrame, xr.Dataset]:
    """
    Clip data to a region: bounding box first, then polygon clip.

    Parameters
    ----------
    data : DataFrame (rows are points) or xr.Dataset (lat/lon coords)
    boundary_poly : shapely polygon
    extent : [west, east, south, north]
    lon_col, lat_col : DataFrame column or MultiIndex level names
        (auto-detected if omitted)
    fix_longitude : convert [0, 360] longitudes to [-180, 180]
    """
    if isinstance(data, xr.Dataset):
        return _clip_xarray(data, boundary_poly, extent, fix_longitude)
    if isinstance(data, pd.DataFrame):
        return _clip_dataframe(data, boundary_poly, extent, lon_col, lat_col, fix_longitude)
    raise TypeError(f"data must be DataFrame or Dataset, got {type(data)}")


def _clip_xarray(ds, boundary_poly, extent, fix_longitude):
    import rioxarray  # noqa: F401  (registers the .rio accessor used below)

    if fix_longitude:
        ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360) - 180)
        ds = ds.sortby("longitude")  # assign_coords leaves coords non-monotonic
    ds = ds.rio.write_crs("EPSG:4326", inplace=True)
    ds = ds.sel(
        longitude=slice(extent[0], extent[1]),
        latitude=slice(extent[3], extent[2]),   # GloFAS lat descends
    )
    return ds.rio.clip([boundary_poly], "EPSG:4326", drop=True)


def _clip_dataframe(df, boundary_poly, extent, lon_col, lat_col, fix_longitude):
    lon_col, lat_col = _resolve_coord_cols(df, lon_col, lat_col)
    in_index = lon_col not in df.columns

    if in_index:
        lons = df.index.get_level_values(lon_col).to_numpy()
        lats = df.index.get_level_values(lat_col).to_numpy()
    else:
        lons = df[lon_col].to_numpy()
        lats = df[lat_col].to_numpy()

    if fix_longitude:
        lons = ((lons + 180) % 360) - 180

    # Bounding box, then polygon
    bbox = (
        (lons >= extent[0]) & (lons <= extent[1]) &
        (lats >= extent[2]) & (lats <= extent[3])
    )
    poly = np.fromiter(
        (boundary_poly.contains(Point(x, y)) for x, y in zip(lons, lats)),
        dtype=bool, count=len(lons),
    )
    keep = bbox & poly
    out = df[keep].copy()

    # Write fixed longitudes back so the output is self-consistent
    if fix_longitude:
        new_lons = lons[keep]
        if in_index:
            idx_arrays = [out.index.get_level_values(name).to_numpy()
                          for name in out.index.names]
            lon_pos = out.index.names.index(lon_col)
            idx_arrays[lon_pos] = new_lons
            out.index = pd.MultiIndex.from_arrays(idx_arrays, names=out.index.names)
        else:
            out[lon_col] = new_lons

    return out


def _resolve_coord_cols(df, lon_col, lat_col):
    """Find longitude and latitude names in columns or MultiIndex levels."""
    available = list(df.columns)
    if isinstance(df.index, pd.MultiIndex):
        available.extend(df.index.names)
    elif df.index.name:
        available.append(df.index.name)

    lon_candidates = [lon_col] if lon_col else ["longitude", "lon", "x", "Longitude", "LON"]
    lat_candidates = [lat_col] if lat_col else ["latitude", "lat", "y", "Latitude", "LAT"]
    lon_col = next((c for c in lon_candidates if c and c in available), None)
    lat_col = next((c for c in lat_candidates if c and c in available), None)
    if lon_col is None or lat_col is None:
        raise ValueError(f"could not find lon/lat in columns or index. Available: {available}")
    return lon_col, lat_col


# =========================================================================
# GloFAS: split + filter (returns mapping for gridded workflows)  (NEW)
# =========================================================================

def split_and_filter_data(
    matrix: np.ndarray,
    location_labels: List[str],
    train_frac: float = 0.7,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Split by time, then drop locations that have NaN/inf in the training set.

    Returns (X_train, X_test, mapping_df), where mapping_df has columns:
      old_idx  : original column position (0 .. n_loc-1)
      new_idx  : position in the filtered matrix (-1 if dropped)
      lat_lon  : original label
      is_valid : True if kept
    """
    n_time, n_loc = matrix.shape
    if len(location_labels) != n_loc:
        raise ValueError(
            f"len(location_labels)={len(location_labels)} != n_loc={n_loc}"
        )

    # Split, then validity check on training set
    n_train = int(train_frac * n_time)
    train_full, test_full = matrix[:n_train], matrix[n_train:]
    is_valid = np.isfinite(train_full).all(axis=0)
    X_train = train_full[:, is_valid]
    X_test  = test_full[:,  is_valid]

    # Vectorized mapping (the original loop did the same thing in O(n))
    new_idx = np.full(n_loc, -1, dtype=int)
    new_idx[is_valid] = np.arange(is_valid.sum())

    mapping_df = pd.DataFrame({
        "old_idx": np.arange(n_loc),
        "new_idx": new_idx,
        "lat_lon": location_labels,
        "is_valid": is_valid,
    })

    if verbose:
        n_kept = int(is_valid.sum())
        print(f"Split: train {X_train.shape}, test {X_test.shape}")
        print(f"Filter: kept {n_kept:,} of {n_loc:,} locations "
              f"({100 * n_kept / n_loc:.1f}%)")

    return X_train, X_test, mapping_df


# =========================================================================
# GloFAS: gauge-to-grid matching  (NEW)
# =========================================================================

def match_gauges_to_grid(
    gauge_gdf: gpd.GeoDataFrame,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    valid_lat_lon: List[str],
    verbose: bool = True,
) -> Tuple[np.ndarray, gpd.GeoDataFrame]:
    """
    Snap each gauge to its nearest grid cell, then keep only the gauges whose
    nearest cell survived NaN filtering (i.e. is in `valid_lat_lon`).

    Returns
    -------
    matrix_cols : column indices into the filtered matrix, one per matched gauge
    matched_gdf : subset of `gauge_gdf` that was successfully matched
    """
    # Coordinates from geometry — independent of column naming
    gauge_lons = gauge_gdf.geometry.x.to_numpy()
    gauge_lats = gauge_gdf.geometry.y.to_numpy()

    # Nearest grid coord for each gauge
    lat_idx = np.abs(grid_lats[:, None] - gauge_lats).argmin(axis=0)
    lon_idx = np.abs(grid_lons[:, None] - gauge_lons).argmin(axis=0)

    # Lookup: (snapped lat, snapped lon) -> matrix column
    lats, lons = zip(*(_parse_latlon_label(s) for s in valid_lat_lon))
    lookup = pd.DataFrame({
        "lat_c": np.round(lats, 4),
        "lon_c": np.round(lons, 4),
        "matrix_col": np.arange(len(valid_lat_lon)),
    })
    gauges = pd.DataFrame({
        "gauge_id": gauge_gdf["gauge_id"].to_numpy(),
        "lat_c": np.round(grid_lats[lat_idx], 4),
        "lon_c": np.round(grid_lons[lon_idx], 4),
    })
    matched = gauges.merge(lookup, on=["lat_c", "lon_c"], how="inner")

    matrix_cols = matched["matrix_col"].to_numpy()
    matched_gdf = gauge_gdf[gauge_gdf["gauge_id"].isin(matched["gauge_id"])].copy()

    if verbose:
        print(f"Matched {len(matrix_cols)} of {len(gauge_gdf)} gauges to valid grid cells")

    return matrix_cols, matched_gdf


def _parse_latlon_label(label: str) -> Tuple[float, float]:
    """Parse '(lat, lon)' into (lat, lon) floats. Tolerates whitespace."""
    lat_str, lon_str = label.strip().strip("()").split(",")
    return float(lat_str.strip()), float(lon_str.strip())


# =========================================================================
# GloFAS: grid -> point GeoDataFrame  (NEW)
# =========================================================================

def grid_to_point_gdf(
    mapping_df: pd.DataFrame,
    crs: str = "EPSG:4326",
    verify: bool = True,
) -> gpd.GeoDataFrame:
    """
    Convert the mapping_df from `split_and_filter_data` to a GeoDataFrame of
    Point geometries — one row per surviving grid cell.

    Columns: col_pos (0..n-1), old_idx, lat, lon, lat_lon, geometry.

    verify=True checks col_pos is strictly 0..n-1 (required for QR indexing).
    """
    valid = mapping_df[mapping_df["is_valid"]].copy()
    if len(valid) == 0:
        raise ValueError("no valid locations in mapping_df")

    coords = np.array([_parse_latlon_label(s) for s in valid["lat_lon"]])
    lats, lons = coords[:, 0], coords[:, 1]

    gdf = gpd.GeoDataFrame(
        {
            "col_pos": valid["new_idx"].to_numpy(),
            "old_idx": valid["old_idx"].to_numpy(),
            "lat": lats,
            "lon": lons,
            "lat_lon": valid["lat_lon"].to_numpy(),
        },
        geometry=gpd.points_from_xy(lons, lats),
        crs=crs,
    )

    if verify:
        expected = np.arange(len(gdf))
        if not np.array_equal(gdf["col_pos"].to_numpy(), expected):
            raise ValueError("col_pos is not sequential 0..n-1; this breaks QR indexing")

    return gdf


# =========================================================================
# GloFAS: basin assignment  (NEW)
# =========================================================================

def assign_basins(
    points_gdf: gpd.GeoDataFrame,
    basin_gdf: Optional[gpd.GeoDataFrame] = None,
    basin_id_col: Optional[str] = None,
    basin_name_col: Optional[str] = None,
    basin_crs: Optional[str] = None,
    country_name: str = "Country",
) -> gpd.GeoDataFrame:
    """
    Spatially join points to basin polygons in two stages:
      1. 'within' join (fast: point-in-polygon, indexed).
      2. 'nearest' fallback for points outside every polygon (typically
         a handful, near coastlines or basin gaps).

    Doing 'nearest' for the full point set is much slower because it
    needs point-to-polygon-edge distance against complex polygons.

    If basin_gdf is None, every point gets basin_id=0, basin_name=country_name.
    """
    if basin_gdf is None:
        out = points_gdf.copy()
        out["basin_id"] = 0
        out["basin_name"] = country_name
        return out

    if basin_id_col is None or basin_name_col is None:
        raise ValueError("basin_id_col and basin_name_col required when basin_gdf is given")
    for col in (basin_id_col, basin_name_col):
        if col not in basin_gdf.columns:
            raise ValueError(f"'{col}' not found in basin_gdf")

    basin_subset = basin_gdf[[basin_id_col, basin_name_col, "geometry"]]

    # Stage 1: fast 'within' join
    joined = gpd.sjoin(points_gdf, basin_subset, how="left", predicate="within")
    joined = joined.drop(columns=[c for c in ("index_right",) if c in joined.columns])

    # Stage 2: nearest fallback for points outside every polygon
    unassigned = joined[basin_id_col].isna()
    n_un = int(unassigned.sum())
    if n_un > 0:
        target_crs = basin_crs or basin_gdf.crs
        pts_un = points_gdf.loc[unassigned.values].to_crs(target_crs)
        basins_proj = basin_subset.to_crs(target_crs)
        nearest = gpd.sjoin_nearest(pts_un, basins_proj, how="left")
        joined.loc[unassigned.values, basin_id_col]   = nearest[basin_id_col].values
        joined.loc[unassigned.values, basin_name_col] = nearest[basin_name_col].values

    return joined.rename(columns={basin_id_col: "basin_id", basin_name_col: "basin_name"})


def count_gauges_per_basin(
    gauge_gdf: gpd.GeoDataFrame,
    basin_gdf: Optional[gpd.GeoDataFrame] = None,
    basin_id_col: Optional[str] = None,
    basin_name_col: Optional[str] = None,
    basin_crs: Optional[str] = None,
    country_name: str = "Country",
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Count gauges per basin. Same matching logic as `assign_basins`.
    """
    assigned = assign_basins(
        gauge_gdf, basin_gdf,
        basin_id_col=basin_id_col,
        basin_name_col=basin_name_col,
        basin_crs=basin_crs,
        country_name=country_name,
    )
    counts = assigned.groupby("basin_name").size().to_dict()
    if verbose:
        print(f"Assigned {sum(counts.values())} gauges to {len(counts)} basins")
    return counts