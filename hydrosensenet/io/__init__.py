"""Efficient I/O utilities for streamflow and sensor data.

This module provides optimized save/load functions with automatic format detection
and intelligent defaults. Parquet format is recommended for better performance and
smaller file sizes.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Union, Optional, Literal

__all__ = [
    'save_streamflow',
    'load_streamflow',
    'save_locations',
    'load_locations',
    'migrate_csv_to_parquet',
    'migrate_directory',
]


def save_streamflow(
    df: pd.DataFrame,
    path: Union[str, Path],
    format: Literal["auto", "parquet", "csv"] = "auto",
    compression: Optional[str] = "snappy",
    **kwargs
) -> Path:
    """Save streamflow data with optimal format and compression."""
    path = Path(path)

    # Auto-detect format
    if format == "auto":
        suffix = path.suffix.lower()
        if suffix in ['.parquet', '.pq']:
            format = "parquet"
        elif suffix in ['.csv', '.txt']:
            format = "csv"
        else:
            # Default to parquet if no clear extension
            format = "parquet"
            if not suffix:
                path = path.with_suffix('.parquet')

    # Create parent directory if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save with appropriate format
    if format == "parquet":
        # Set optimal defaults for timeseries data
        parquet_kwargs = {
            'engine': 'pyarrow',
            'compression': compression,
            'index': True,  # Preserve time index
        }
        parquet_kwargs.update(kwargs)
        df.to_parquet(path, **parquet_kwargs)

    elif format == "csv":
        csv_kwargs = {'index': True}
        csv_kwargs.update(kwargs)
        df.to_csv(path, **csv_kwargs)

    return path


def load_streamflow(
    path: Union[str, Path],
    format: Literal["auto", "parquet", "csv"] = "auto",
    columns: Optional[list] = None,
    **kwargs
) -> pd.DataFrame:
    """Load streamflow data with automatic format detection."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Auto-detect format
    if format == "auto":
        suffix = path.suffix.lower()
        if suffix in ['.parquet', '.pq']:
            format = "parquet"
        elif suffix in ['.csv', '.txt']:
            format = "csv"
        else:
            raise ValueError(f"Cannot auto-detect format from extension: {suffix}")

    # Load with appropriate format
    if format == "parquet":
        parquet_kwargs = {'engine': 'pyarrow'}
        if columns:
            parquet_kwargs['columns'] = columns
        parquet_kwargs.update(kwargs)
        df = pd.read_parquet(path, **parquet_kwargs)

    elif format == "csv":
        csv_kwargs = {'index_col': 0, 'parse_dates': True}
        # Handle column selection for CSV (less efficient)
        if columns:
            csv_kwargs['usecols'] = [csv_kwargs['index_col']] + columns
        csv_kwargs.update(kwargs)
        df = pd.read_csv(path, **csv_kwargs)

    return df


def save_locations(
    gdf: Union[pd.DataFrame, gpd.GeoDataFrame],
    path: Union[str, Path],
    format: Literal["auto", "parquet", "csv", "geojson", "shapefile"] = "auto",
    **kwargs
) -> Path:
    """Save gauge/sensor location data."""
    path = Path(path)

    # Auto-detect format
    if format == "auto":
        suffix = path.suffix.lower()
        format_map = {
            '.parquet': 'parquet',
            '.pq': 'parquet',
            '.csv': 'csv',
            '.geojson': 'geojson',
            '.json': 'geojson',
            '.shp': 'shapefile',
        }
        format = format_map.get(suffix, 'parquet')
        if suffix not in format_map:
            path = path.with_suffix('.parquet')

    path.parent.mkdir(parents=True, exist_ok=True)

    # Save with appropriate format
    if isinstance(gdf, gpd.GeoDataFrame):
        if format == "parquet":
            gdf.to_parquet(path, **kwargs)
        elif format == "geojson":
            gdf.to_file(path, driver='GeoJSON', **kwargs)
        elif format == "shapefile":
            gdf.to_file(path, **kwargs)
        elif format == "csv":
            # Extract coordinates and save as CSV
            df = gdf.copy()
            if 'geometry' in df.columns:
                df['longitude'] = df.geometry.x
                df['latitude'] = df.geometry.y
                df = df.drop(columns=['geometry'])
            df.to_csv(path, index=False, **kwargs)
    else:
        # Regular DataFrame
        if format == "parquet":
            gdf.to_parquet(path, **kwargs)
        else:
            gdf.to_csv(path, index=False, **kwargs)

    return path


def load_locations(
    path: Union[str, Path],
    format: Literal["auto", "parquet", "csv", "geojson", "shapefile"] = "auto",
    **kwargs
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Load gauge/sensor location data.

    Parameters
    ----------
    path : str or Path
        Input file path.
    format : {"auto", "parquet", "csv", "geojson", "shapefile"}, default="auto"
        Input format. "auto" detects from extension.
    **kwargs
        Additional arguments passed to read functions.

    Returns
    -------
    DataFrame or GeoDataFrame
        Location data.

    Examples
    --------
    >>> # Load parquet
    >>> locations = load_locations("locations.parquet")

    >>> # Load GeoJSON
    >>> locations = load_locations("locations.geojson")

    >>> # Load CSV
    >>> locations = load_locations("locations.csv")
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Auto-detect format
    if format == "auto":
        suffix = path.suffix.lower()
        format_map = {
            '.parquet': 'parquet',
            '.pq': 'parquet',
            '.csv': 'csv',
            '.geojson': 'geojson',
            '.json': 'geojson',
            '.shp': 'shapefile',
        }
        format = format_map.get(suffix)
        if format is None:
            raise ValueError(f"Cannot auto-detect format from extension: {suffix}")

    # Load with appropriate format
    if format == "parquet":
        # Try GeoParquet first, fall back to regular parquet
        try:
            return gpd.read_parquet(path, **kwargs)
        except Exception:
            return pd.read_parquet(path, **kwargs)
    elif format in ["geojson", "shapefile"]:
        return gpd.read_file(path, **kwargs)
    elif format == "csv":
        return pd.read_csv(path, **kwargs)


def migrate_csv_to_parquet(
    csv_path: Union[str, Path],
    parquet_path: Optional[Union[str, Path]] = None,
    compression: str = "snappy",
    remove_csv: bool = False,
    **kwargs
) -> Path:
    """
    Migrate CSV file to Parquet format.

    Parameters
    ----------
    csv_path : str or Path
        Path to input CSV file.
    parquet_path : str or Path, optional
        Path to output parquet file. If None, replaces .csv with .parquet.
    compression : str, default="snappy"
        Compression algorithm: "snappy", "gzip", "brotli", "zstd".
    remove_csv : bool, default=False
        If True, delete CSV file after successful migration.
    **kwargs
        Additional arguments passed to pd.read_csv.

    Returns
    -------
    Path
        Path to created parquet file.

    Examples
    --------
    >>> # Migrate single file
    >>> migrate_csv_to_parquet("streamflow.csv")
    >>> # Creates: streamflow.parquet

    >>> # Migrate and remove original
    >>> migrate_csv_to_parquet("streamflow.csv", remove_csv=True)

    >>> # Custom output path
    >>> migrate_csv_to_parquet("old/data.csv", "new/data.parquet")
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Determine output path
    if parquet_path is None:
        parquet_path = csv_path.with_suffix('.parquet')
    else:
        parquet_path = Path(parquet_path)

    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    # Read CSV with intelligent defaults
    csv_kwargs = {'index_col': 0, 'parse_dates': True}
    csv_kwargs.update(kwargs)

    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path, **csv_kwargs)

    # Get file sizes
    csv_size = csv_path.stat().st_size / (1024 * 1024)  # MB

    print(f"Writing Parquet: {parquet_path}")
    df.to_parquet(
        parquet_path,
        engine='pyarrow',
        compression=compression,
        index=True
    )

    parquet_size = parquet_path.stat().st_size / (1024 * 1024)  # MB
    compression_ratio = csv_size / parquet_size if parquet_size > 0 else 0

    print(f"✓ Migration complete!")
    print(f"  CSV size:     {csv_size:.2f} MB")
    print(f"  Parquet size: {parquet_size:.2f} MB")
    print(f"  Compression:  {compression_ratio:.1f}x smaller")

    # Remove CSV if requested
    if remove_csv:
        csv_path.unlink()
        print(f"  Removed: {csv_path}")

    return parquet_path


def migrate_directory(
    directory: Union[str, Path],
    pattern: str = "*.csv",
    compression: str = "snappy",
    remove_csv: bool = False,
    **kwargs
) -> list[Path]:
    """
    Migrate all CSV files in a directory to Parquet.

    Parameters
    ----------
    directory : str or Path
        Directory containing CSV files.
    pattern : str, default="*.csv"
        Glob pattern for CSV files.
    compression : str, default="snappy"
        Compression algorithm.
    remove_csv : bool, default=False
        If True, delete CSV files after migration.
    **kwargs
        Additional arguments passed to pd.read_csv.

    Returns
    -------
    list of Path
        Paths to created parquet files.

    Examples
    --------
    >>> # Migrate all CSVs in a directory
    >>> parquet_files = migrate_directory("data/")

    >>> # Migrate with custom pattern
    >>> parquet_files = migrate_directory("data/", pattern="streamflow_*.csv")
    """
    directory = Path(directory)
    csv_files = sorted(directory.glob(pattern))

    if not csv_files:
        print(f"No CSV files found matching pattern: {pattern}")
        return []

    print(f"Found {len(csv_files)} CSV files to migrate\n")

    parquet_files = []
    for i, csv_file in enumerate(csv_files, 1):
        print(f"[{i}/{len(csv_files)}] {csv_file.name}")
        try:
            parquet_file = migrate_csv_to_parquet(
                csv_file,
                compression=compression,
                remove_csv=remove_csv,
                **kwargs
            )
            parquet_files.append(parquet_file)
            print()
        except Exception as e:
            print(f"  ✗ Error: {e}\n")

    print(f"Successfully migrated {len(parquet_files)}/{len(csv_files)} files")
    return parquet_files
