"""National Water Model (NWM) data utilities.

This module provides functions to download, process, and filter NWM data
for sensor network optimization workflows.
"""

import numpy as np
import pandas as pd
import xarray as xr
import fsspec
import dask
from typing import Union, List, Optional, Tuple
from pathlib import Path


class NWMDataLoader:
    """
    Load and process National Water Model retrospective data from AWS.

    Examples
    --------
    >>> # Filter by HUC and download data
    >>> loader = NWMDataLoader(version="3.0")
    >>> loader.filter_by_huc("1204")  # Texas-Gulf HUC4
    >>> df = loader.download_streamflow(
    ...     start_date="2020-01-01",
    ...     end_date="2022-01-01",
    ...     resample="D"
    ... )
    >>> df.to_csv("streamflow_daily.csv")

    >>> # Or filter by COMID list
    >>> comids = [1234567, 7654321, 9876543]
    >>> loader = NWMDataLoader(version="3.0")
    >>> loader.filter_by_comids(comids)
    >>> df = loader.download_streamflow("2021-01-01", "2022-01-01")
    """

    # NWM dataset URLs
    NWM_URLS = {
        "2.0": "s3://noaa-nwm-retro-v2-zarr-pds",
        "2.1": "s3://noaa-nwm-retrospective-2-1-zarr-pds",
        "3.0": "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr"
    }

    def __init__(
        self,
        version: str = "3.0",
        use_dask: bool = True,
        n_workers: int = 8,
        threads_per_worker: int = 4,
        memory_limit: str = '8GB'
    ):
        """Initialize NWM data loader with configurable Dask parallelization."""
        self.version = version
        self.use_dask = use_dask
        self.comids = None
        self.huc_code = None
        self.ds = None
        self.n_workers = n_workers
        self.threads_per_worker = threads_per_worker
        self.memory_limit = memory_limit

        # Setup Dask if requested
        if use_dask:
            self._setup_dask()

    def _setup_dask(self):
        """Setup Dask local cluster."""
        try:
            from dask.distributed import Client, LocalCluster
            cluster = LocalCluster(
                n_workers=self.n_workers,
                threads_per_worker=self.threads_per_worker,
                memory_limit=self.memory_limit
            )
            self.client = cluster.get_client()
            print(f"✓ Dask cluster started: {self.client.dashboard_link}")
            print(f"  Workers: {self.n_workers}, Threads/worker: {self.threads_per_worker}, Memory/worker: {self.memory_limit}")
        except ImportError:
            print("Warning: dask.distributed not available. Install with: pip install 'dask[distributed]'")
            self.use_dask = False

    def filter_by_huc(
        self,
        huc: str,
        huc_level: Optional[int] = None
    ) -> "NWMDataLoader":
        """Filter COMIDs by Hydrologic Unit Code (HUC)."""
        try:
            from pynhd import nhdplus_vaa
        except ImportError:
            raise ImportError(
                "PyNHD required for HUC filtering. Install with:\n"
                "  pip install pynhd"
            )

        # Auto-detect HUC level
        if huc_level is None:
            huc_level = len(huc)

        print(f"Fetching COMIDs for HUC{huc_level}: {huc}...")
        print(f"  Downloading NHDPlusV2 Value Added Attributes (245 MB, cached after first use)...")

        # Get NHDPlusV2 Value Added Attributes table
        vaa = nhdplus_vaa()

        # Filter by reachcode (first digits correspond to HUC8)
        # For HUC levels < 8, we filter by reachcode prefix
        print(f"  Filtering {len(vaa):,} flowlines by HUC{huc_level}...")

        mask = vaa['reachcode'].astype(str).str.startswith(huc)
        filtered = vaa[mask]

        # Extract COMIDs as integers (NWM uses integer feature_ids)
        self.comids = filtered['comid'].astype(int).tolist()
        self.huc_code = huc

        print(f"\n✓ Found {len(self.comids):,} NHDPlusV2 COMIDs in HUC {huc}")
        print(f"  These COMIDs match NWM v3.0 feature_id values")

        return self

    def filter_by_comids(self, comids: List[Union[int, str]]) -> "NWMDataLoader":
        """Filter by explicit list of COMIDs."""
        self.comids = [int(c) for c in comids]
        print(f"✓ Filtering by {len(self.comids)} COMIDs")
        return self

    def filter_by_shapefile(
        self,
        shapefile: Union[str, Path],
        comid_column: str = "COMID"
    ) -> "NWMDataLoader":
        """Filter by COMIDs from a shapefile."""
        import geopandas as gpd

        gdf = gpd.read_file(shapefile)
        self.comids = gdf[comid_column].astype(int).tolist()
        print(f"✓ Loaded {len(self.comids)} COMIDs from {shapefile}")
        return self

    def filter_by_csv(
        self,
        csv_file: Union[str, Path],
        comid_column: str = "COMID"
    ) -> "NWMDataLoader":
        """Filter by COMIDs from a CSV file."""
        df = pd.read_csv(csv_file)
        self.comids = df[comid_column].astype(int).tolist()
        print(f"✓ Loaded {len(self.comids)} COMIDs from {csv_file}")
        return self

    def export_locations(
        self,
        output_file: Union[str, Path],
        format: str = "csv",
        sample_size: Optional[int] = None,
        use_full_geometries: bool = False
    ) -> Path:
        """Export COMID locations to file."""
        if self.comids is None:
            raise ValueError("No COMIDs specified. Filter COMIDs first.")

        try:
            from pynhd import nhdplus_l48
            import geopandas as gpd
        except ImportError:
            raise ImportError(
                "PyNHD and GeoPandas required. Install with:\n"
                "  pip install pynhd geopandas pyogrio py7zr"
            )

        output_file = Path(output_file)
        comids_to_export = self.comids

        # Sample if requested
        if sample_size and sample_size < len(self.comids):
            import random
            comids_to_export = random.sample(self.comids, sample_size)
            print(f"Sampling {sample_size:,} of {len(self.comids):,} COMIDs...")

        print(f"\nExporting locations for {len(comids_to_export):,} COMIDs...")
        print(f"  Downloading NHDPlus flowline dataset (7.3 GB, cached after first use)...")
        print(f"  This may take 10-30 minutes on first run depending on internet speed...")

        # Download NHDPlus flowlines (cached in ./cache directory)
        try:
            flowlines = nhdplus_l48(
                layer="NHDFlowline_Network",
                sql=f"COMID IN ({','.join(map(str, comids_to_export[:1000]))})"  # Limit query
            )
        except Exception as e:
            # If SQL query fails, download full dataset and filter
            print(f"  Direct query failed, downloading full flowline dataset...")
            flowlines = nhdplus_l48(layer="NHDFlowline_Network")

            # NHDPlus uses uppercase COMID
            comid_col = 'COMID' if 'COMID' in flowlines.columns else 'comid'
            flowlines = flowlines[flowlines[comid_col].isin(comids_to_export)]

        print(f"  ✓ Loaded {len(flowlines):,} flowline geometries")

        # Standardize column name to lowercase
        comid_col = 'COMID' if 'COMID' in flowlines.columns else 'comid'
        if comid_col == 'COMID':
            flowlines = flowlines.rename(columns={'COMID': 'comid'})

        # Calculate centroids if needed
        if format == "csv" or not use_full_geometries:
            print(f"  Calculating centroids...")
            flowlines['longitude'] = flowlines.geometry.centroid.x
            flowlines['latitude'] = flowlines.geometry.centroid.y

        # Save to file
        if format == "csv":
            output_df = flowlines[['comid', 'latitude', 'longitude']].copy()
            output_df.to_csv(output_file, index=False)
            print(f"\n✓ Saved {len(output_df):,} COMID locations to {output_file}")
        elif format == "geojson":
            if not use_full_geometries:
                # Create point geometries from centroids
                flowlines.geometry = flowlines.geometry.centroid
            flowlines[['comid', 'geometry']].to_file(output_file, driver='GeoJSON')
            print(f"\n✓ Saved {len(flowlines):,} COMID geometries to {output_file}")
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'geojson'.")

        return output_file

    def _open_dataset(self, batch_size: int = 5000) -> xr.Dataset:
        """Open NWM Zarr dataset from S3."""
        if self.ds is not None:
            return self.ds

        url = self.NWM_URLS[self.version]
        fs = fsspec.filesystem('s3', anon=True)

        print(f"Opening NWM v{self.version} dataset from S3...")

        if self.version == "3.0":
            # v3.0 has subdirectories
            output_list = fs.ls(url)
            mapper = fs.get_mapper(output_list[1])  # Use channel_rt
        else:
            mapper = fs.get_mapper(url)

        # Open dataset
        self.ds = xr.open_dataset(
            mapper,
            engine='zarr',
            backend_kwargs={'consolidated': True},
            chunks={}
        )

        print(f"✓ Dataset opened. Size: {self.ds['streamflow'].nbytes/1e12:.2f} TB")

        return self.ds

    def download_streamflow(
        self,
        start_date: str,
        end_date: str,
        resample: str = "D",
        resample_method: str = "mean",
        batch_size: int = 5000,
        output_file: Optional[Union[str, Path]] = None,
        format: str = "auto"
    ) -> pd.DataFrame:
        """Download and process streamflow data."""
        if self.comids is None:
            raise ValueError(
                "No COMIDs specified. Use filter_by_huc(), filter_by_comids(), "
                "or filter_by_shapefile() first."
            )

        # Open dataset
        ds = self._open_dataset()

        # Process in batches
        all_dfs = []
        n_batches = (len(self.comids) + batch_size - 1) // batch_size

        print(f"Processing {len(self.comids)} COMIDs in {n_batches} batch(es)...")

        for i in range(0, len(self.comids), batch_size):
            batch_comids = self.comids[i:i+batch_size]
            batch_num = i // batch_size + 1

            print(f"  Batch {batch_num}/{n_batches}: {len(batch_comids)} COMIDs")

            # Filter by COMIDs
            idx = ds.feature_id.isin(batch_comids)

            with dask.config.set(**{'array.slicing.split_large_chunks': False}):
                ds_filtered = ds[['streamflow']].isel(feature_id=idx)

            # Extract time range
            ds_sub = ds_filtered['streamflow'].sel(time=slice(start_date, end_date))

            # Resample to desired frequency
            if resample:
                agg_func = getattr(ds_sub.resample(time=resample), resample_method)
                ds_resampled = agg_func()
            else:
                ds_resampled = ds_sub

            # Convert to DataFrame
            df_batch = ds_resampled.to_dataframe().reset_index()
            df_pivot = df_batch.pivot_table(
                index='time',
                columns='feature_id',
                values='streamflow'
            )

            all_dfs.append(df_pivot)

        # Combine all batches
        df_combined = pd.concat(all_dfs, axis=1)

        # Quality check
        missing_pct = df_combined.isnull().sum() / len(df_combined) * 100
        n_high_missing = (missing_pct > 10).sum()

        print(f"\n✓ Downloaded {df_combined.shape[1]} stations, {df_combined.shape[0]} timesteps")
        print(f"  Date range: {df_combined.index[0]} to {df_combined.index[-1]}")
        print(f"  Stations with >10% missing data: {n_high_missing}")

        if n_high_missing > 0:
            print(f"  Consider removing low-quality stations with:")
            print(f"    df = df.loc[:, missing_pct < 10]")

        # Save if requested
        if output_file:
            from hydrosensenet.io import save_streamflow
            save_streamflow(df_combined, output_file, format=format)
            print(f"  Saved to: {output_file}")

        return df_combined

    def download_multiple_years(
        self,
        years: List[int],
        output_dir: Union[str, Path] = ".",
        resample: str = "D",
        format: str = "parquet",
        **kwargs
    ) -> List[Path]:
        """Download multiple years of data, saving each year separately."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = []

        for i, year in enumerate(years, 1):
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"

            # Determine file extension from format
            if format == "csv":
                ext = ".csv"
            else:  # parquet or auto defaults to parquet
                ext = ".parquet"

            output_file = output_dir / f"streamflow_{year}{ext}"

            print(f"\n{'='*60}")
            print(f"Year {i}/{len(years)}: {year}")
            print(f"{'='*60}")

            df = self.download_streamflow(
                start_date=start_date,
                end_date=end_date,
                resample=resample,
                output_file=output_file,
                format=format,
                **kwargs
            )

            saved_files.append(output_file)

        print(f"\n{'='*60}")
        print(f"✓ All {len(years)} years downloaded!")
        print(f"{'='*60}")
        print(f"Files saved to: {output_dir}")
        print(f"\nTo use with hydro-sensor-network:")
        print(f"  STREAMFLOW_FILES = [")
        for f in saved_files:
            print(f"      Path('{f}'),")
        print(f"  ]")

        return saved_files


def get_huc_info(huc: str) -> dict:
    """
    Get information about a Hydrologic Unit Code.

    Parameters
    ----------
    huc : str
        HUC code (e.g., "12", "1204", "120401").

    Returns
    -------
    dict
        HUC information including name, area, and parent HUC.

    Examples
    --------
    >>> info = get_huc_info("1204")
    >>> print(info['name'])
    'Brazos-Colorado'
    """
    try:
        from pygeohydro import WBD
    except ImportError:
        raise ImportError(
            "PyGeoHydro required. Install with:\n"
            "  pip install pygeohydro"
        )

    huc_level = len(huc)
    wbd = WBD(f"huc{huc_level}")

    gdf = wbd.byids(f"huc{huc_level}", [huc])

    if len(gdf) == 0:
        raise ValueError(f"HUC {huc} not found")

    row = gdf.iloc[0]

    return {
        'huc': huc,
        'level': huc_level,
        'name': row.get('name', 'Unknown'),
        'area_sqkm': row.geometry.area / 1e6,  # Approximate
        'geometry': row.geometry
    }


def list_available_hucs(region: Optional[str] = None, level: int = 2) -> pd.DataFrame:
    """
    List available HUC regions.

    Parameters
    ----------
    region : str, optional
        Filter by region code (e.g., "12" for Texas-Gulf).
    level : int, default=2
        HUC level to list (2, 4, 8, 10, 12).

    Returns
    -------
    pd.DataFrame
        Available HUCs with codes and names.

    Examples
    --------
    >>> # List all HUC2 regions
    >>> hucs = list_available_hucs(level=2)
    >>>
    >>> # List HUC4 subregions in Texas-Gulf
    >>> hucs = list_available_hucs(region="12", level=4)
    """
    try:
        from pygeohydro import WBD
    except ImportError:
        raise ImportError(
            "PyGeoHydro required. Install with:\n"
            "  pip install pygeohydro"
        )

    wbd = WBD(f"huc{level}")

    if region:
        gdf = wbd.bysql(f"huc{level} LIKE '{region}%'")
    else:
        # Get all (may be slow for high levels)
        gdf = wbd.bygeom("conus")

    result = pd.DataFrame({
        'huc': gdf[f'huc{level}'],
        'name': gdf['name'],
        'states': gdf.get('states', 'N/A')
    })

    return result.sort_values('huc').reset_index(drop=True)
